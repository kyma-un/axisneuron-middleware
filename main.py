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
import json
import socket
import struct
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import zmq

HEADER_FMT = "<H4BHBBI32sIIIIH"  # 64 bytes
HEADER_SIZE = struct.calcsize(HEADER_FMT)
TOKEN1, TOKEN2 = 0xDDFF, 0xEEFF


# --------------------------------------------------------------------------- #
# Parsing of UDP datagrams
# --------------------------------------------------------------------------- #
def parse_datagram(data: bytes, text_has_index=False):
    """Return dict(frame, avatar, channels) or None if not parseable."""
    if len(data) >= HEADER_SIZE:
        (t1, _v0, _v1, _v2, _v3, count, with_disp, with_ref, avatar_idx,
         name, frame_idx, _r0, _r1, _r2, t2) = struct.unpack_from(HEADER_FMT, data, 0)
        if t1 == TOKEN1 and t2 == TOKEN2:
            n = min(count, (len(data) - HEADER_SIZE) // 4)
            ch = list(struct.unpack_from(f"<{n}f", data, HEADER_SIZE))
            return {
                "frame": frame_idx,
                "avatar": name.split(b"\0", 1)[0].decode("ascii", "ignore"),
                "channels": ch,
            }
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
    return {"frame": frame, "avatar": "", "channels": vals}


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
# Shared state
# --------------------------------------------------------------------------- #
class State:
    def __init__(self, hierarchy_text, frame_time, max_frames):
        self.hierarchy = hierarchy_text
        self.joints = parse_hierarchy(hierarchy_text) if hierarchy_text else None
        self.frame_time = frame_time
        self.frames = deque(maxlen=max_frames)
        self.latest = None
        self.count = 0
        self.fps = 0.0
        self._last_t = time.time()
        self.cond = threading.Condition()

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
            if self.joints is None:
                self._build_default(len(msg["channels"]))
            self.latest = msg
            self.count += 1
            self.frames.append(msg["channels"])
            dt = now - self._last_t
            self._last_t = now
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            self.cond.notify_all()

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
        state.push(msg)
        pub.send_multipart([b"bvh", json.dumps(msg).encode()])
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
</style>
<div id=top>
 <b>BVH middleware</b> &nbsp;
 <a href="/bvh">BVH</a> | <a href="/skeleton">skeleton</a> | <a href="/latest">latest</a> | <a href="/status">status</a>
 &nbsp; <span id=s>waiting for data...</span> <span id=warn></span>
 <div style="opacity:.6">drag = orbit, wheel = zoom, double click = reset</div>
</div>
<canvas id=c></canvas>
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

        def do_GET(self):
            if self.path == "/":
                self._send(PAGE, "text/html; charset=utf-8")
            elif self.path == "/bvh":
                self._send(state.bvh_text(), "text/plain; charset=utf-8")
            elif self.path == "/skeleton":
                self._send(json.dumps(state.skeleton_json()), "application/json")
            elif self.path == "/latest":
                self._send(json.dumps(state.latest), "application/json")
            elif self.path == "/status":
                self._send(json.dumps({"frames": state.count, "fps": round(state.fps, 1),
                                       "buffered": len(state.frames)}), "application/json")
            elif self.path == "/stream":
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
    args = ap.parse_args()

    hierarchy = ""
    if args.hierarchy:
        hierarchy = open(args.hierarchy).read().split("MOTION")[0]
    state = State(hierarchy, args.frame_time, args.buffer_frames)

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
        stop.set()
        srv.shutdown()


if __name__ == "__main__":
    main()
