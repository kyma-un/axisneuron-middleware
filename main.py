#!/usr/bin/env python3
"""
BVH middleware: UDP in  ->  ZMQ PUB out  (+ local HTTP server with a live skeleton viewer)

Flow
  [Axis Neuron] --UDP--> [this script] --ZMQ PUB--> subscribers
                              |
                              +--> http://127.0.0.1:8080  (3D skeleton, /bvh, /skeleton, /latest, /status, /stream)

Supported UDP payloads (auto-detected):
  1. Axis Neuron binary BVH stream (64-byte header, tokens 0xDDFF ... 0xEEFF, then floats)
  2. Plain text: one frame per datagram, whitespace-separated floats
  3. Axis Neuron Calculation binary (same tokens, then 16-float bone segments:
     displacement m, velocity m/s, quaternion WXYZ, acceleration g, gyro rad/s)

Right arm (RightShoulder, RightArm, RightForeArm, RightHand):
  Calculation frames use the packet acceleration and gyroscope.
  BVH frames derive world linear acceleration and angular velocity.
  http://127.0.0.1:8080 shows them live. Record from the page or --record-arm.
  Stop writes recordings/right-arm-*.csv (open in Excel). /arm.csv downloads it.

Skeleton shown in the browser:
  * --hierarchy file.bvh  -> uses the HIERARCHY of that file
  * otherwise             -> built-in dummy Axis Neuron skeleton (59 bones, T-pose offsets in cm).
                             Detected automatically from the channel count of the first frame:
                             354/360 channels = with displacement, 180/186 = without (+6 = with reference).

Install:  pip install pyzmq
Run:      python main.py --src-ip 192.168.0.62 --udp-port 7001

ZMQ messages (multipart): [b"bvh", json_bytes]
  json = {"frame": int, "avatar": str, "t": float, "channels": [floats]}
"""
import argparse
import csv
import io
import json
import math
import socket
import struct
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import zmq

HEADER_FMT = "<H4BHBBI32sIIIIH"  # 64 bytes, BVH 1.1
CALC_FMT = "<H4BII32sIIIIH"  # 64 bytes, Calculation
HEADER_SIZE = struct.calcsize(HEADER_FMT)
CALC_HEADER_SIZE = struct.calcsize(CALC_FMT)
TOKEN1, TOKEN2 = 0xDDFF, 0xEEFF
ARM_JOINTS = ("RightShoulder", "RightArm", "RightForeArm", "RightHand")
# 1-based ids from the Axis Neuron manual, stored 0-based.
CALC_BONES = (
    "Hips",
    "RightUpLeg", "RightLeg", "RightFoot",
    "LeftUpLeg", "LeftLeg", "LeftFoot",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "Head", "Neck", "Spine3", "Spine2", "Spine1", "Spine",
)
_ID3 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Parsing of UDP datagrams
# --------------------------------------------------------------------------- #
def _avatar_name(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("ascii", "ignore")


def _parse_bvh_binary(data: bytes):
    if len(data) < HEADER_SIZE:
        return None
    (t1, _v0, _v1, _v2, _v3, count, _disp, _ref, _avatar_idx,
     name, frame_idx, _r0, _r1, _r2, t2) = struct.unpack_from(HEADER_FMT, data, 0)
    if t1 != TOKEN1 or t2 != TOKEN2:
        return None
    n = min(count, (len(data) - HEADER_SIZE) // 4)
    if n <= 0:
        return None
    ch = list(struct.unpack_from(f"<{n}f", data, HEADER_SIZE))
    return {
        "kind": "bvh",
        "frame": frame_idx,
        "avatar": _avatar_name(name),
        "channels": ch,
    }


def _parse_calculation(data: bytes):
    """Calculation binary. None when the body is not 16-float bone segments."""
    if len(data) < CALC_HEADER_SIZE:
        return None
    (t1, _v0, _v1, _v2, _v3, data_count, _avatar_idx, name,
     _r1, _r2, _r3, _r4, t2) = struct.unpack_from(CALC_FMT, data, 0)
    if t1 != TOKEN1 or t2 != TOKEN2 or data_count <= 0:
        return None
    nfloat = (len(data) - CALC_HEADER_SIZE) // 4
    if nfloat in (180, 186, 354, 360):
        return None
    if nfloat == data_count and data_count % 16 == 0:
        nseg = data_count // 16
    elif nfloat == data_count * 16:
        nseg = data_count
    else:
        return None
    if not 0 < nseg <= 128:
        return None
    vals = struct.unpack_from(f"<{nseg * 16}f", data, CALC_HEADER_SIZE)
    segments = []
    for i in range(nseg):
        chunk = vals[i * 16:(i + 1) * 16]
        segments.append({
            "name": CALC_BONES[i] if i < len(CALC_BONES) else f"Bone{i + 1}",
            "displacement": list(chunk[0:3]),
            "velocity": list(chunk[3:6]),
            "quaternion": list(chunk[6:10]),
            "acceleration": list(chunk[10:13]),
            "gyroscope": list(chunk[13:16]),
        })
    return {
        "kind": "calc",
        "frame": None,
        "avatar": _avatar_name(name),
        "segments": segments,
    }


def parse_datagram(data: bytes, text_has_index=False):
    """Return a BVH dict, a Calculation dict, or None."""
    calc = _parse_calculation(data)
    if calc is not None:
        return calc
    binary = _parse_bvh_binary(data)
    if binary is not None:
        return binary
    try:
        parts = data.decode("ascii").replace(",", " ").split()
        vals = [float(p) for p in parts]
    except (UnicodeDecodeError, ValueError):
        return None
    if not vals:
        return None
    frame = None
    if text_has_index:
        frame, vals = int(vals[0]), vals[1:]
    return {"kind": "bvh", "frame": frame, "avatar": "", "channels": vals}


# --------------------------------------------------------------------------- #
# Skeleton: BVH hierarchy parser + built-in dummy Axis Neuron skeleton
# --------------------------------------------------------------------------- #
def parse_hierarchy(text):
    """BVH HIERARCHY text -> flat list of joints (DFS order, end sites included)."""
    toks = text.replace("{", " { ").replace("}", " } ").split()
    joints, stack, pending, i = [], [], None, 0
    while i < len(toks):
        t = toks[i]
        if t in ("ROOT", "JOINT"):
            joints.append(dict(name=toks[i + 1], parent=stack[-1] if stack else -1,
                               offset=[0.0, 0.0, 0.0], channels=[], end=False))
            pending = len(joints) - 1
            i += 2
        elif t == "End":
            joints.append(dict(name=joints[stack[-1]]["name"] + "_end", parent=stack[-1],
                               offset=[0.0, 0.0, 0.0], channels=[], end=True))
            pending = len(joints) - 1
            i += 2
        elif t == "{":
            stack.append(pending)
            i += 1
        elif t == "}":
            stack.pop()
            i += 1
        elif t == "OFFSET":
            joints[stack[-1]]["offset"] = [float(x) for x in toks[i + 1:i + 4]]
            i += 4
        elif t == "CHANNELS":
            n = int(toks[i + 1])
            joints[stack[-1]]["channels"] = toks[i + 2:i + 2 + n]
            i += 2 + n
        else:
            i += 1
    return joints


def _hand(side):
    s = -1 if side == "Right" else 1
    thumb3 = (f"{side}HandThumb3", (3 * s, 0, 0), [])
    thumb2 = (f"{side}HandThumb2", (3.5 * s, 0, 0), [thumb3])
    kids = [(f"{side}HandThumb1", (3 * s, 0, 3), [thumb2])]
    for nm, z in (("Index", 3), ("Middle", 1), ("Ring", -1), ("Pinky", -3)):
        f3 = (f"{side}Hand{nm}3", (2 * s, 0, 0), [])
        f2 = (f"{side}Hand{nm}2", (2.5 * s, 0, 0), [f3])
        f1 = (f"{side}Hand{nm}1", (4 * s, 0, 0), [f2])
        kids.append((f"{side}InHand{nm}", (5 * s, 0, z), [f1]))
    return kids


def _arm(side):
    s = -1 if side == "Right" else 1
    hand = (f"{side}Hand", (25 * s, 0, 0), _hand(side))
    fore = (f"{side}ForeArm", (28 * s, 0, 0), [hand])
    arm = (f"{side}Arm", (14 * s, 0, 0), [fore])
    return (f"{side}Shoulder", (4 * s, 14, 0), [arm])


def _leg(side):
    s = -1 if side == "Right" else 1
    foot = (f"{side}Foot", (0, -42, 0), [])
    leg = (f"{side}Leg", (0, -42, 0), [foot])
    return (f"{side}UpLeg", (9 * s, -4, 0), [leg])


def _default_tree():
    head = ("Head", (0, 6, 0), [])
    neck1 = ("Neck1", (0, 6, 0), [head])
    neck = ("Neck", (0, 14, 0), [neck1])
    spine2 = ("Spine2", (0, 12, 0), [neck, _arm("Right"), _arm("Left")])
    spine1 = ("Spine1", (0, 12, 0), [spine2])
    spine = ("Spine", (0, 10, 0), [spine1])
    return ("Hips", (0, 100, 0), [_leg("Right"), _leg("Left"), spine])


_END_OFFSETS = {"RightFoot": (0, -4, 12), "LeftFoot": (0, -4, 12), "Head": (0, 14, 0)}


def _emit(node, depth, with_disp, root=False):
    name, off, kids = node
    ind = "\t" * depth
    if root or with_disp:
        ch = "6 Xposition Yposition Zposition Yrotation Xrotation Zrotation"
    else:
        ch = "3 Yrotation Xrotation Zrotation"
    lines = [f"{ind}{'ROOT' if root else 'JOINT'} {name}", f"{ind}{{",
             f"{ind}\tOFFSET {off[0]:.4f} {off[1]:.4f} {off[2]:.4f}", f"{ind}\tCHANNELS {ch}"]
    for k in kids:
        lines += _emit(k, depth + 1, with_disp)
    if not kids:
        e = _END_OFFSETS.get(name, tuple(0.7 * v for v in off))
        lines += [f"{ind}\tEnd Site", f"{ind}\t{{", f"{ind}\t\tOFFSET {e[0]:.4f} {e[1]:.4f} {e[2]:.4f}",
                  f"{ind}\t}}"]
    lines.append(f"{ind}}}")
    return lines


def build_default_hierarchy(with_disp):
    return "HIERARCHY\n" + "\n".join(_emit(_default_tree(), 0, with_disp, root=True)) + "\n"


# --------------------------------------------------------------------------- #
# Right-arm kinematics (BVH derivative) and Calculation samples
# --------------------------------------------------------------------------- #
def _rot_m(axis, deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    if axis == "X":
        return (1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c)
    if axis == "Y":
        return (c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c)
    return (c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)


def _mul(a, b):
    o = [0.0] * 9
    for i in range(3):
        for j in range(3):
            o[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j]
    return tuple(o)


def _mv(m, v):
    return (
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    )


def _angvel(prev, curr, dt):
    """World-frame angular velocity (rad/s) from two row-major rotations."""
    if dt <= 1e-6:
        return (None, None, None)
    pt = (prev[0], prev[3], prev[6], prev[1], prev[4], prev[7], prev[2], prev[5], prev[8])
    r = _mul(curr, pt)
    trace = max(-1.0, min(1.0, (r[0] + r[4] + r[8] - 1.0) / 2.0))
    theta = math.acos(trace)
    if theta < 1e-8:
        return ((r[7] - r[5]) / (2.0 * dt), (r[2] - r[6]) / (2.0 * dt), (r[3] - r[1]) / (2.0 * dt))
    scale = theta / (2.0 * math.sin(theta) * dt)
    return ((r[7] - r[5]) * scale, (r[2] - r[6]) * scale, (r[3] - r[1]) * scale)


def fk_pose(joints, channels):
    """World positions (cm) and rotations. Same channel order as the web viewer."""
    total = sum(len(j["channels"]) for j in joints)
    k = max(0, len(channels) - total)
    points, rots = [], []
    for joint in joints:
        pos = list(joint["offset"])
        rot = _ID3
        for ch in joint["channels"]:
            v = channels[k] if k < len(channels) else 0.0
            k += 1
            axis = ch[0].upper()
            if ch.endswith("position"):
                pos["XYZ".index(axis)] += v
            else:
                rot = _mul(rot, _rot_m(axis, v))
        if joint["parent"] < 0:
            points.append(tuple(pos))
            rots.append(rot)
        else:
            parent = joint["parent"]
            moved = _mv(rots[parent], pos)
            points.append((moved[0] + points[parent][0], moved[1] + points[parent][1], moved[2] + points[parent][2]))
            rots.append(_mul(rots[parent], rot))
    return points, rots


def _arm_row(sample):
    units = sample["units"]
    by_name = {j["name"]: j for j in sample["joints"]}
    row = [
        f"{sample['t']:.6f}" if sample.get("t") is not None else "",
        "" if sample.get("frame") is None else sample["frame"],
        sample["source"],
        units["velocity"],
        units["acceleration"],
        units["angular_velocity"],
    ]
    for name in ARM_JOINTS:
        joint = by_name.get(name)
        if joint is None:
            row.extend([""] * 9)
            continue
        for vec in (joint["velocity"], joint["acceleration"], joint["angular_velocity"]):
            for value in vec:
                row.append("" if value is None else f"{value:.6f}")
    return row


def arm_csv_text(rows):
    buf = io.StringIO()
    buf.write("\ufeff")
    writer = csv.writer(buf)
    header = ["t_s", "frame", "source", "velocity_unit", "acceleration_unit", "angular_velocity_unit"]
    for name in ARM_JOINTS:
        for comp in ("vx", "vy", "vz", "ax", "ay", "az", "wx", "wy", "wz"):
            header.append(f"{name}_{comp}")
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue()


def _blank3():
    return (None, None, None)


# --------------------------------------------------------------------------- #
# Shared state
# --------------------------------------------------------------------------- #
class State:
    def __init__(self, hierarchy_text, frame_time, max_frames, record_dir):
        self.hierarchy = hierarchy_text
        self.joints = parse_hierarchy(hierarchy_text) if hierarchy_text else None
        self.frame_time = frame_time
        self.frames = deque(maxlen=max_frames)
        self.latest = None
        self.count = 0
        self.fps = 0.0
        self._last_t = time.time()
        self.cond = threading.Condition()
        self.record_dir = Path(record_dir)
        self.arm = None
        self.arm_recording = False
        self.arm_rows = []
        self.arm_file = None
        self._arm_prev = None

    def _build_default(self, n):
        disp = n in (354, 360)
        if n not in (354, 360, 180, 186):
            print(f"[skel] unexpected channel count {n}; assuming 59 bones without displacement")
        self.hierarchy = build_default_hierarchy(disp)
        self.joints = parse_hierarchy(self.hierarchy)
        print(f"[skel] built-in dummy skeleton ({'with' if disp else 'without'} displacement)")

    def push(self, msg):
        now = time.time()
        with self.cond:
            if msg.get("kind") == "calc":
                self._calc_arm(msg)
            else:
                if self.joints is None:
                    self._build_default(len(msg["channels"]))
                self.latest = msg
                self.frames.append(msg["channels"])
                self._derive_arm(msg)
            self.count += 1
            dt = now - self._last_t
            self._last_t = now
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            self.cond.notify_all()

    def _store_arm(self, sample):
        self.arm = sample
        if self.arm_recording:
            self.arm_rows.append(_arm_row(sample))

    def _calc_arm(self, msg):
        self._arm_prev = None
        by_name = {seg["name"]: seg for seg in msg["segments"]}
        joints = []
        for name in ARM_JOINTS:
            seg = by_name.get(name)
            if seg is None:
                continue
            joints.append({
                "name": name,
                "velocity": seg["velocity"],
                "acceleration": seg["acceleration"],
                "angular_velocity": seg["gyroscope"],
            })
        self._store_arm({
            "source": "calc",
            "t": msg.get("t"),
            "frame": msg.get("frame"),
            "units": {"velocity": "m/s", "acceleration": "g", "angular_velocity": "rad/s"},
            "joints": joints,
        })

    def _derive_arm(self, msg):
        if not self.joints:
            return
        points, rots = fk_pose(self.joints, msg["channels"])
        index = {joint["name"]: i for i, joint in enumerate(self.joints)}
        t = msg["t"]
        prev = self._arm_prev
        pos, vel, rot = {}, {}, {}
        joints = []
        for name in ARM_JOINTS:
            i = index.get(name)
            if i is None:
                continue
            pos[name] = points[i]
            rot[name] = rots[i]
            v = a = w = _blank3()
            if prev and name in prev["pos"]:
                dt = t - prev["t"]
                if 0 < dt <= 0.2:
                    p0 = prev["pos"][name]
                    v = tuple((pos[name][k] - p0[k]) / dt for k in range(3))
                    w = _angvel(prev["rot"][name], rot[name], dt)
                    v0 = prev["vel"].get(name)
                    if v0 and v0[0] is not None:
                        a = tuple((v[k] - v0[k]) / dt for k in range(3))
            vel[name] = v
            joints.append({
                "name": name,
                "velocity": v,
                "acceleration": a,
                "angular_velocity": w,
            })
        self._arm_prev = {"t": t, "pos": pos, "vel": vel, "rot": rot}
        self._store_arm({
            "source": "derived",
            "t": t,
            "frame": msg.get("frame"),
            "units": {"velocity": "cm/s", "acceleration": "cm/s^2", "angular_velocity": "rad/s"},
            "joints": joints,
        })

    def start_arm_record(self):
        with self.cond:
            self.arm_rows = []
            self.arm_file = None
            self.arm_recording = True

    def stop_arm_record(self):
        with self.cond:
            was = self.arm_recording
            self.arm_recording = False
            if not was or not self.arm_rows:
                return self.arm_file
            self.record_dir.mkdir(parents=True, exist_ok=True)
            path = self.record_dir / time.strftime("right-arm-%Y%m%d-%H%M%S.csv")
            path.write_text(arm_csv_text(self.arm_rows), encoding="utf-8")
            self.arm_file = str(path)
            print(f"[arm ] wrote {path} ({len(self.arm_rows)} samples)")
            return self.arm_file

    def arm_json(self):
        with self.cond:
            return {
                "recording": self.arm_recording,
                "samples": len(self.arm_rows),
                "file": self.arm_file,
                "sample": self.arm,
            }

    def arm_csv(self):
        with self.cond:
            return arm_csv_text(self.arm_rows)

    def skeleton_json(self):
        with self.cond:
            if self.joints is None:
                return {"joints": None, "total": 0}
            total = sum(len(j["channels"]) for j in self.joints)
            return {"joints": self.joints, "total": total}

    def bvh_text(self):
        with self.cond:
            frames = list(self.frames)
            head = self.hierarchy.strip() if self.hierarchy else "HIERARCHY\n# (waiting for first frame)"
        out = [head, "MOTION", f"Frames: {len(frames)}", f"Frame Time: {self.frame_time:.6f}"]
        out += [" ".join(f"{v:.4f}" for v in f) for f in frames]
        return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# UDP -> ZMQ thread
# --------------------------------------------------------------------------- #
def udp_loop(args, state, stop):
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 100)
    pub.bind(args.zmq_bind)
    print(f"[zmq ] PUB bound on {args.zmq_bind} (topic 'bvh')")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.bind((args.bind_ip, args.udp_port))
    sock.settimeout(1.0)
    print(f"[udp ] listening on {args.bind_ip}:{args.udp_port}, source filter: {args.src_ip or 'none'}")

    last_handshake = 0.0
    dropped = 0
    while not stop.is_set():
        if args.handshake and time.time() - last_handshake > args.handshake_every:
            try:
                sock.sendto(args.handshake.encode(), (args.src_ip, args.device_port or args.udp_port))
            except OSError as e:
                print(f"[udp ] handshake failed: {e}")
            last_handshake = time.time()
        try:
            data, (ip, _port) = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except OSError:
            break
        if args.src_ip and ip != args.src_ip:
            continue
        msg = parse_datagram(data, args.text_has_index)
        if msg is None:
            dropped += 1
            if dropped % 100 == 1:
                print(f"[udp ] unparseable datagram ({len(data)} bytes) from {ip}; total dropped {dropped}")
            continue
        if msg["frame"] is None:
            msg["frame"] = state.count
        msg["t"] = time.time()
        kind = msg.get("kind", "bvh")
        state.push(msg)
        if kind == "calc":
            payload = {k: msg[k] for k in ("frame", "avatar", "t", "segments")}
            pub.send_multipart([b"calc", json.dumps(payload).encode()])
        else:
            payload = {k: msg[k] for k in ("frame", "avatar", "t", "channels")}
            pub.send_multipart([b"bvh", json.dumps(payload).encode()])
        if args.zmq_raw:
            pub.send_multipart([b"bvh.raw", data])
    sock.close()
    pub.close(0)


# --------------------------------------------------------------------------- #
# Local HTTP server + 3D viewer (no external libraries: canvas 2D + own FK)
# --------------------------------------------------------------------------- #
PAGE = r"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>BVH middleware</title>
<style>
html,body{margin:0;height:100%;background:#111;color:#ddd;font:13px monospace}
#top{position:fixed;top:0;left:0;right:0;padding:8px 12px;background:#000a;z-index:2}
a{color:#6cf}
canvas{display:block;width:100vw;height:100vh}
#warn{color:#fb4}
#arm{position:fixed;right:12px;bottom:12px;z-index:2;background:#000d;padding:10px 12px;border:1px solid #333;max-width:460px}
#arm button{background:#222;color:#eee;border:1px solid #555;padding:2px 8px;margin-right:6px;font:inherit;cursor:pointer}
#arm a{margin-right:8px}
#arm table{border-collapse:collapse;margin-top:6px}
#arm td{padding:1px 8px 1px 0;white-space:nowrap}
#arm .rec{color:#f66}
#arm .src{color:#9cf}
</style>
<div id=top>
 <b>BVH middleware</b> &nbsp;
 <a href="/bvh">BVH</a> | <a href="/skeleton">skeleton</a> | <a href="/latest">latest</a> | <a href="/status">status</a> | <a href="/arm.csv">arm.csv</a>
 &nbsp; <span id=s>waiting for data...</span> <span id=warn></span>
 <div style="opacity:.6">drag = orbit, wheel = zoom, double click = reset</div>
</div>
<canvas id=c></canvas>
<div id=arm>
 <b>Right arm</b> <span id=arm-src class=src>waiting</span>
 <span id=arm-n></span><br>
 <button id=rec type=button>Record</button>
 <button id=stop type=button>Stop</button>
 <a href="/arm.csv">Excel</a>
 <span id=arm-file></span>
 <table id=arm-table></table>
 <div style="opacity:.6;margin-top:4px">calc: accel in g, gyro in rad/s. derived from BVH: accel in cm/s², gyro in rad/s.</div>
</div>
<script>
const cv=document.getElementById('c'), ctx=cv.getContext('2d');
let skel=null, last=null, yaw=0.6, pitch=0.25, zoom=1, dirty=true;

function resize(){cv.width=innerWidth*devicePixelRatio;cv.height=innerHeight*devicePixelRatio;dirty=true}
addEventListener('resize',resize);resize();

function loadSkel(){
  fetch('/skeleton').then(r=>r.json()).then(s=>{ if(s.joints){skel=s;dirty=true;} else setTimeout(loadSkel,1000); })
  .catch(()=>setTimeout(loadSkel,1000));
}
loadSkel();

const es=new EventSource('/stream');
es.onmessage=e=>{
  last=JSON.parse(e.data);
  document.getElementById('s').textContent='frame '+last.frame+' | '+(last.avatar||'-')+' | '+last.channels.length+' ch';
  if(skel){const skip=last.channels.length-skel.total;
    document.getElementById('warn').textContent=(skip<0)?' | WARNING: skeleton needs '+skel.total+' channels':'';}
  dirty=true;
};

// 3x3 row-major helpers
function rotM(a,deg){const r=deg*Math.PI/180,c=Math.cos(r),s=Math.sin(r);
  if(a==='X')return[1,0,0,0,c,-s,0,s,c];
  if(a==='Y')return[c,0,s,0,1,0,-s,0,c];
  return[c,-s,0,s,c,0,0,0,1];}
function mul(a,b){const o=new Array(9);
  for(let i=0;i<3;i++)for(let j=0;j<3;j++)o[i*3+j]=a[i*3]*b[j]+a[i*3+1]*b[3+j]+a[i*3+2]*b[6+j];return o;}
function mv(m,v){return[m[0]*v[0]+m[1]*v[1]+m[2]*v[2],m[3]*v[0]+m[4]*v[1]+m[5]*v[2],m[6]*v[0]+m[7]*v[1]+m[8]*v[2]];}

function fk(ch){
  const J=skel.joints, n=J.length, R=new Array(n), P=new Array(n);
  let k=Math.max(0,ch.length-skel.total);   // leading reference channels are skipped
  for(let i=0;i<n;i++){
    const j=J[i]; let pos=j.offset.slice(), r=[1,0,0,0,1,0,0,0,1];
    for(const c of j.channels){
      const v=ch[k++]||0, ax=c[0].toUpperCase();
      if(c.endsWith('position'))pos['XYZ'.indexOf(ax)]+=v; else r=mul(r,rotM(ax,v));
    }
    if(j.parent<0){R[i]=r;P[i]=pos;}
    else{const pr=R[j.parent],pp=P[j.parent],w=mv(pr,pos);
      R[i]=mul(pr,r);P[i]=[w[0]+pp[0],w[1]+pp[1],w[2]+pp[2]];}
  }
  return P;
}

function render(){
  const W=cv.width,H=cv.height; ctx.clearRect(0,0,W,H);
  if(!skel){ctx.fillStyle='#888';ctx.font=(14*devicePixelRatio)+'px monospace';
    ctx.fillText('waiting for first frame...',20,H/2);return;}
  const ch=last?last.channels:[];
  const P=fk(ch.length?ch:new Array(skel.total).fill(0));
  const cx=P[0][0], cz=P[0][2], cy=90;
  const sc=H/260*zoom, cyaw=Math.cos(yaw),syaw=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  const proj=p=>{let x=p[0]-cx,y=p[1]-cy,z=p[2]-cz;
    let x1=cyaw*x+syaw*z, z1=-syaw*x+cyaw*z;
    let y1=cp*y-sp*z1, z2=sp*y+cp*z1;
    return[W/2+x1*sc, H/2-y1*sc, z2];};
  // floor grid
  ctx.lineWidth=1;ctx.strokeStyle='#2a2a2a';
  for(let g=-300;g<=300;g+=50){
    let a=proj([cx+g,0,cz-300]),b=proj([cx+g,0,cz+300]);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
    a=proj([cx-300,0,cz+g]);b=proj([cx+300,0,cz+g]);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
  }
  const Q=P.map(proj), J=skel.joints;
  for(let i=0;i<J.length;i++){
    const j=J[i]; if(j.parent<0)continue;
    const a=Q[j.parent],b=Q[i];
    const finger=/Hand(Thumb|Index|Middle|Ring|Pinky)|InHand/.test(j.name);
    ctx.strokeStyle=j.name.startsWith('Right')?'#ff9f43':j.name.startsWith('Left')?'#48dbfb':'#eee';
    ctx.lineWidth=(finger?1:3)*devicePixelRatio;
    ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
    if(!j.end){ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(b[0],b[1],(finger?1.2:2.8)*devicePixelRatio,0,7);ctx.fill();}
  }
  ctx.fillStyle='#f55';ctx.beginPath();ctx.arc(Q[0][0],Q[0][1],5*devicePixelRatio,0,7);ctx.fill();
}
(function loop(){ if(dirty){dirty=false;render();} requestAnimationFrame(loop); })();

let drag=null;
cv.onmousedown=e=>drag=[e.clientX,e.clientY];
addEventListener('mouseup',()=>drag=null);
addEventListener('mousemove',e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*0.01;
  pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-drag[1])*0.01));drag=[e.clientX,e.clientY];dirty=true;});
cv.onwheel=e=>{e.preventDefault();zoom*=e.deltaY<0?1.1:0.9;dirty=true;};
cv.ondblclick=()=>{yaw=0.6;pitch=0.25;zoom=1;dirty=true;};

function fmt(v){return (v==null||Number.isNaN(+v))?'—':(+v).toFixed(2)}
function vec(v){return v?v.map(fmt).join('  '):'—'}
function paintArm(d){
  const s=d.sample;
  const src=document.getElementById('arm-src');
  if(!s){src.textContent='waiting';}
  else{
    const u=s.units;
    src.textContent=s.source+'   v '+u.velocity+'   a '+u.acceleration+'   w '+u.angular_velocity;
  }
  const n=document.getElementById('arm-n');
  n.textContent=(d.recording?'  REC  ':'  ')+d.samples+' samples';
  n.className=d.recording?'rec':'';
  const tb=document.getElementById('arm-table');
  tb.innerHTML='<tr><td></td><td>velocity</td><td>acceleration</td><td>angular vel</td></tr>'+
    ((s&&s.joints)||[]).map(j=>'<tr><td>'+j.name.replace('Right','')+'</td><td>'+vec(j.velocity)+'</td><td>'+vec(j.acceleration)+'</td><td>'+vec(j.angular_velocity)+'</td></tr>').join('');
  if(d.file) document.getElementById('arm-file').textContent=d.file.split(/[/\\\\]/).pop();
}
function pollArm(){fetch('/arm').then(r=>r.json()).then(paintArm).catch(()=>{})}
setInterval(pollArm,200); pollArm();
document.getElementById('rec').onclick=()=>fetch('/arm/start',{method:'POST'}).then(pollArm);
document.getElementById('stop').onclick=()=>fetch('/arm/stop',{method:'POST'}).then(r=>r.json()).then(d=>{pollArm(); if(d.file) document.getElementById('arm-file').textContent=d.file.split(/[/\\\\]/).pop();});
</script>"""


def make_handler(state):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype, code=200):
            b = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _path(self):
            return self.path.split("?", 1)[0]

        def _drain(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n:
                self.rfile.read(n)

        def do_POST(self):
            self._drain()
            path = self._path()
            if path == "/arm/start":
                state.start_arm_record()
                self._send(json.dumps(state.arm_json()), "application/json")
            elif path == "/arm/stop":
                path_out = state.stop_arm_record()
                body = state.arm_json()
                body["file"] = path_out
                self._send(json.dumps(body), "application/json")
            else:
                self._send("not found", "text/plain", 404)

        def do_GET(self):
            path = self._path()
            if path == "/":
                self._send(PAGE, "text/html; charset=utf-8")
            elif path == "/bvh":
                self._send(state.bvh_text(), "text/plain; charset=utf-8")
            elif path == "/skeleton":
                self._send(json.dumps(state.skeleton_json()), "application/json")
            elif path == "/latest":
                self._send(json.dumps(state.latest), "application/json")
            elif path == "/arm":
                self._send(json.dumps(state.arm_json()), "application/json")
            elif path == "/arm/start":
                state.start_arm_record()
                self._send(json.dumps(state.arm_json()), "application/json")
            elif path == "/arm/stop":
                state.stop_arm_record()
                self._send(json.dumps(state.arm_json()), "application/json")
            elif path == "/arm.csv":
                body = state.arm_csv().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="right-arm.csv"')
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/status":
                self._send(json.dumps({"frames": state.count, "fps": round(state.fps, 1),
                                       "buffered": len(state.frames),
                                       "arm_recording": state.arm_recording,
                                       "arm_samples": len(state.arm_rows)}), "application/json")
            elif path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                last = -1
                try:
                    while True:
                        with state.cond:
                            state.cond.wait_for(lambda: state.count != last, timeout=15)
                            msg, last = state.latest, state.count
                        payload = f"data: {json.dumps(msg)}\n\n" if msg else ": ping\n\n"
                        self.wfile.write(payload.encode())
                        self.wfile.flush()
                        time.sleep(0.03)  # cap UI rate ~30 Hz
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self._send("not found", "text/plain", 404)

    return H


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-ip", default="", help="only accept datagrams from this IP (also handshake target)")
    ap.add_argument("--bind-ip", default="0.0.0.0")
    ap.add_argument("--udp-port", type=int, default=7001, help="local UDP port to listen on")
    ap.add_argument("--device-port", type=int, default=0, help="remote port for the handshake (default: udp-port)")
    ap.add_argument("--handshake", default="", help="string sent to src-ip to start/keep the stream alive")
    ap.add_argument("--handshake-every", type=float, default=2.0)
    ap.add_argument("--text-has-index", action="store_true", help="text mode: first value is the frame index")
    ap.add_argument("--hierarchy", default="", help="BVH file whose HIERARCHY is used (default: built-in dummy)")
    ap.add_argument("--frame-time", type=float, default=1 / 60)
    ap.add_argument("--buffer-frames", type=int, default=600)
    ap.add_argument("--zmq-bind", default="tcp://*:5556")
    ap.add_argument("--zmq-raw", action="store_true")
    ap.add_argument("--http-host", default="127.0.0.1")
    ap.add_argument("--http-port", type=int, default=8080)
    ap.add_argument("--record-arm", action="store_true",
                    help="start recording the right arm immediately (CSV on exit or Stop)")
    args = ap.parse_args()

    hierarchy = ""
    if args.hierarchy:
        hierarchy = open(args.hierarchy).read().split("MOTION")[0]
    record_dir = Path(__file__).resolve().parent / "recordings"
    state = State(hierarchy, args.frame_time, args.buffer_frames, record_dir)
    if args.record_arm:
        state.start_arm_record()
        print(f"[arm ] recording right arm; CSV will be written to {record_dir}")

    stop = threading.Event()
    threading.Thread(target=udp_loop, args=(args, state, stop), daemon=True).start()

    srv = ThreadingHTTPServer((args.http_host, args.http_port), make_handler(state))
    srv.daemon_threads = True
    print(f"[http] http://{args.http_host}:{args.http_port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.stop_arm_record()
        stop.set()
        srv.shutdown()


if __name__ == "__main__":
    main()
