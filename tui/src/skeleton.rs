//! BVH hierarchy and forward kinematics, matching the viewer in `main.py`.
//!
//! The built-in tree is the Axis Neuron dummy: 59 joints, centimetre offsets,
//! T-pose. End sites are drawn but consume no channels. A frame longer than
//! the skeleton (360 or 186) has a leading reference block; those floats are
//! skipped, same as the browser viewer.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    Xpos,
    Ypos,
    Zpos,
    Xrot,
    Yrot,
    Zrot,
}

impl Channel {
    pub fn is_position(self) -> bool {
        matches!(self, Self::Xpos | Self::Ypos | Self::Zpos)
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Xpos => "Xpos",
            Self::Ypos => "Ypos",
            Self::Zpos => "Zpos",
            Self::Xrot => "Xrot",
            Self::Yrot => "Yrot",
            Self::Zrot => "Zrot",
        }
    }

    pub fn unit(self) -> &'static str {
        if self.is_position() {
            "cm"
        } else {
            "deg"
        }
    }

    fn parse(name: &str) -> Result<Self, String> {
        match name.to_ascii_lowercase().as_str() {
            "xposition" => Ok(Self::Xpos),
            "yposition" => Ok(Self::Ypos),
            "zposition" => Ok(Self::Zpos),
            "xrotation" => Ok(Self::Xrot),
            "yrotation" => Ok(Self::Yrot),
            "zrotation" => Ok(Self::Zrot),
            other => Err(format!("unknown channel {other}")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Joint {
    pub name: String,
    pub parent: i32,
    pub offset: [f64; 3],
    pub channels: Vec<Channel>,
    pub end_site: bool,
}

#[derive(Clone, Debug)]
pub struct Skeleton {
    pub joints: Vec<Joint>,
    /// True when every joint carries XYZ position channels (354 / 360 frames).
    pub with_displacement: bool,
}

#[derive(Clone, Debug)]
pub struct SolvedFrame {
    pub points: Vec<[f64; 3]>,
    /// Channel values consumed by each joint, in hierarchy order.
    pub locals: Vec<Vec<f64>>,
    /// Leading floats ignored (Axis Neuron reference block).
    pub skipped: usize,
}

impl Skeleton {
    pub fn axis_neuron(with_displacement: bool) -> Self {
        let mut joints = Vec::new();
        flatten(&default_tree(), -1, with_displacement, true, &mut joints);
        Self {
            joints,
            with_displacement,
        }
    }

    pub fn channel_count(&self) -> usize {
        self.joints.iter().map(|j| j.channels.len()).sum()
    }

    pub fn selectable(&self) -> Vec<usize> {
        self.joints
            .iter()
            .enumerate()
            .filter(|(_, j)| !j.end_site)
            .map(|(i, _)| i)
            .collect()
    }

    pub fn find(&self, name: &str) -> Option<usize> {
        self.joints.iter().position(|j| j.name == name)
    }

    /// Pose the hierarchy. Missing channels read as 0. Extra leading channels
    /// are the reference block and are not applied.
    pub fn solve(&self, channels: &[f64]) -> SolvedFrame {
        let total = self.channel_count();
        let skipped = channels.len().saturating_sub(total);
        let mut cursor = skipped;
        let mut points = Vec::with_capacity(self.joints.len());
        let mut locals = Vec::with_capacity(self.joints.len());
        let mut rots: Vec<Mat3> = Vec::with_capacity(self.joints.len());

        for joint in &self.joints {
            let mut pos = joint.offset;
            let mut rot = Mat3::id();
            let mut local = Vec::with_capacity(joint.channels.len());
            for ch in &joint.channels {
                let v = channels.get(cursor).copied().unwrap_or(0.0);
                cursor += 1;
                local.push(v);
                match ch {
                    Channel::Xpos => pos[0] += v,
                    Channel::Ypos => pos[1] += v,
                    Channel::Zpos => pos[2] += v,
                    _ => rot = rot.mul(Mat3::rot(*ch, v)),
                }
            }
            if joint.parent < 0 {
                points.push(pos);
                rots.push(rot);
            } else {
                let p = joint.parent as usize;
                let moved = rots[p].mul_vec(pos);
                points.push([
                    moved[0] + points[p][0],
                    moved[1] + points[p][1],
                    moved[2] + points[p][2],
                ]);
                rots.push(rots[p].mul(rot));
            }
            locals.push(local);
        }

        SolvedFrame {
            points,
            locals,
            skipped,
        }
    }
}

/// `HIERARCHY` text, optionally followed by a `MOTION` section, which is ignored.
pub fn parse_hierarchy(text: &str) -> Result<Skeleton, String> {
    let text = text.split("MOTION").next().unwrap_or(text);
    let spaced = text.replace('{', " { ").replace('}', " } ");
    let toks: Vec<&str> = spaced.split_whitespace().collect();

    let mut joints = Vec::new();
    let mut stack: Vec<i32> = Vec::new();
    let mut pending: i32 = -1;
    let mut i = 0;

    while i < toks.len() {
        let t = toks[i];
        if t == "ROOT" || t == "JOINT" {
            let name = toks.get(i + 1).ok_or("truncated joint name")?.to_string();
            let parent = stack.last().copied().unwrap_or(-1);
            joints.push(Joint {
                name,
                parent,
                offset: [0.0; 3],
                channels: Vec::new(),
                end_site: false,
            });
            pending = joints.len() as i32 - 1;
            i += 2;
        } else if t == "End" {
            let parent = *stack.last().ok_or("End Site outside a joint")?;
            let name = format!("{}_end", joints[parent as usize].name);
            joints.push(Joint {
                name,
                parent,
                offset: [0.0; 3],
                channels: Vec::new(),
                end_site: true,
            });
            pending = joints.len() as i32 - 1;
            i += 2;
        } else if t == "{" {
            if pending < 0 {
                return Err("brace without a joint".into());
            }
            stack.push(pending);
            i += 1;
        } else if t == "}" {
            stack.pop().ok_or("unbalanced '}'")?;
            i += 1;
        } else if t == "OFFSET" {
            let top = *stack.last().ok_or("OFFSET outside a joint")? as usize;
            joints[top].offset = [
                parse_f64(&toks, i + 1)?,
                parse_f64(&toks, i + 2)?,
                parse_f64(&toks, i + 3)?,
            ];
            i += 4;
        } else if t == "CHANNELS" {
            let top = *stack.last().ok_or("CHANNELS outside a joint")? as usize;
            let n: usize = toks
                .get(i + 1)
                .ok_or("missing channel count")?
                .parse()
                .map_err(|_| format!("bad channel count '{}'", toks[i + 1]))?;
            let mut chans = Vec::with_capacity(n);
            for k in 0..n {
                let name = toks.get(i + 2 + k).ok_or("truncated channel list")?;
                chans.push(Channel::parse(name)?);
            }
            joints[top].channels = chans;
            i += 2 + n;
        } else {
            i += 1;
        }
    }

    if joints.is_empty() {
        return Err("no joints in hierarchy".into());
    }
    let with_displacement = joints
        .iter()
        .skip(1)
        .any(|j| j.channels.iter().any(|c| c.is_position()));
    Ok(Skeleton {
        joints,
        with_displacement,
    })
}

fn parse_f64(toks: &[&str], i: usize) -> Result<f64, String> {
    let t = toks.get(i).ok_or("truncated OFFSET")?;
    t.parse().map_err(|_| format!("bad offset '{t}'"))
}

fn pose_channels(six: bool) -> Vec<Channel> {
    if six {
        vec![
            Channel::Xpos,
            Channel::Ypos,
            Channel::Zpos,
            Channel::Yrot,
            Channel::Xrot,
            Channel::Zrot,
        ]
    } else {
        vec![Channel::Yrot, Channel::Xrot, Channel::Zrot]
    }
}

struct Node {
    name: String,
    offset: [f64; 3],
    children: Vec<Node>,
}

fn node(name: impl Into<String>, offset: [f64; 3], children: Vec<Node>) -> Node {
    Node {
        name: name.into(),
        offset,
        children,
    }
}

fn side_sign(side: &str) -> f64 {
    if side == "Right" {
        -1.0
    } else {
        1.0
    }
}

fn fingers(side: &str) -> Vec<Node> {
    let s = side_sign(side);
    let mut kids = vec![node(
        format!("{side}HandThumb1"),
        [3.0 * s, 0.0, 3.0],
        vec![node(
            format!("{side}HandThumb2"),
            [3.5 * s, 0.0, 0.0],
            vec![node(
                format!("{side}HandThumb3"),
                [3.0 * s, 0.0, 0.0],
                vec![],
            )],
        )],
    )];
    for (name, z) in [
        ("Index", 3.0),
        ("Middle", 1.0),
        ("Ring", -1.0),
        ("Pinky", -3.0),
    ] {
        kids.push(node(
            format!("{side}InHand{name}"),
            [5.0 * s, 0.0, z],
            vec![node(
                format!("{side}Hand{name}1"),
                [4.0 * s, 0.0, 0.0],
                vec![node(
                    format!("{side}Hand{name}2"),
                    [2.5 * s, 0.0, 0.0],
                    vec![node(
                        format!("{side}Hand{name}3"),
                        [2.0 * s, 0.0, 0.0],
                        vec![],
                    )],
                )],
            )],
        ));
    }
    kids
}

fn arm(side: &str) -> Node {
    let s = side_sign(side);
    let hand = node(format!("{side}Hand"), [25.0 * s, 0.0, 0.0], fingers(side));
    let fore = node(format!("{side}ForeArm"), [28.0 * s, 0.0, 0.0], vec![hand]);
    let upper = node(format!("{side}Arm"), [14.0 * s, 0.0, 0.0], vec![fore]);
    node(format!("{side}Shoulder"), [4.0 * s, 14.0, 0.0], vec![upper])
}

fn leg(side: &str) -> Node {
    let s = side_sign(side);
    let foot = node(format!("{side}Foot"), [0.0, -42.0, 0.0], vec![]);
    let shin = node(format!("{side}Leg"), [0.0, -42.0, 0.0], vec![foot]);
    node(format!("{side}UpLeg"), [9.0 * s, -4.0, 0.0], vec![shin])
}

fn default_tree() -> Node {
    let head = node("Head", [0.0, 6.0, 0.0], vec![]);
    let neck1 = node("Neck1", [0.0, 6.0, 0.0], vec![head]);
    let neck = node("Neck", [0.0, 14.0, 0.0], vec![neck1]);
    let spine2 = node(
        "Spine2",
        [0.0, 12.0, 0.0],
        vec![neck, arm("Right"), arm("Left")],
    );
    let spine1 = node("Spine1", [0.0, 12.0, 0.0], vec![spine2]);
    let spine = node("Spine", [0.0, 10.0, 0.0], vec![spine1]);
    node(
        "Hips",
        [0.0, 100.0, 0.0],
        vec![leg("Right"), leg("Left"), spine],
    )
}

fn end_offset(name: &str, offset: [f64; 3]) -> [f64; 3] {
    match name {
        "RightFoot" | "LeftFoot" => [0.0, -4.0, 12.0],
        "Head" => [0.0, 14.0, 0.0],
        _ => [offset[0] * 0.7, offset[1] * 0.7, offset[2] * 0.7],
    }
}

fn flatten(node: &Node, parent: i32, with_displacement: bool, root: bool, out: &mut Vec<Joint>) {
    let index = out.len() as i32;
    out.push(Joint {
        name: node.name.clone(),
        parent,
        offset: node.offset,
        channels: pose_channels(root || with_displacement),
        end_site: false,
    });
    if node.children.is_empty() {
        out.push(Joint {
            name: format!("{}_end", node.name),
            parent: index,
            offset: end_offset(&node.name, node.offset),
            channels: Vec::new(),
            end_site: true,
        });
    } else {
        for child in &node.children {
            flatten(child, index, with_displacement, false, out);
        }
    }
}

/// Row-major 3x3. Column vectors, so `mul` applies the right-hand matrix first.
#[derive(Clone, Copy)]
struct Mat3([f64; 9]);

impl Mat3 {
    fn id() -> Self {
        Self([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    }

    fn mul(self, b: Self) -> Self {
        let a = self.0;
        let b = b.0;
        let mut o = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                o[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
            }
        }
        Self(o)
    }

    fn mul_vec(self, v: [f64; 3]) -> [f64; 3] {
        let m = self.0;
        [
            m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
            m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
        ]
    }

    fn rot(axis: Channel, deg: f64) -> Self {
        let r = deg.to_radians();
        let (c, s) = (r.cos(), r.sin());
        match axis {
            Channel::Xrot => Self([1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c]),
            Channel::Yrot => Self([c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c]),
            Channel::Zrot => Self([c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]),
            _ => Self::id(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f64; 3], b: [f64; 3]) {
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-6, "{a:?} != {b:?}");
        }
    }

    #[test]
    fn builtin_matches_axis_neuron_channel_counts() {
        let rot = Skeleton::axis_neuron(false);
        let disp = Skeleton::axis_neuron(true);
        assert_eq!(rot.selectable().len(), 59);
        assert_eq!(disp.selectable().len(), 59);
        assert_eq!(rot.joints.len(), 59 + 13);
        assert_eq!(rot.channel_count(), 180);
        assert_eq!(disp.channel_count(), 354);
        assert!(rot
            .joints
            .iter()
            .filter(|j| j.end_site)
            .all(|j| j.channels.is_empty()));

        let names: Vec<_> = rot.joints.iter().take(6).map(|j| j.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "Hips",
                "RightUpLeg",
                "RightLeg",
                "RightFoot",
                "RightFoot_end",
                "LeftUpLeg"
            ]
        );
        assert_eq!(rot.joints[0].channels.len(), 6);
        assert_eq!(rot.joints[1].channels.len(), 3);
        assert_eq!(disp.joints[1].channels.len(), 6);
    }

    #[test]
    fn t_pose_places_the_body() {
        let skel = Skeleton::axis_neuron(false);
        let solved = skel.solve(&vec![0.0; 180]);
        close(solved.points[skel.find("Hips").unwrap()], [0.0, 100.0, 0.0]);
        close(solved.points[skel.find("Head").unwrap()], [0.0, 160.0, 0.0]);
        close(
            solved.points[skel.find("Head_end").unwrap()],
            [0.0, 174.0, 0.0],
        );
        close(
            solved.points[skel.find("RightFoot").unwrap()],
            [-9.0, 12.0, 0.0],
        );
        close(
            solved.points[skel.find("RightFoot_end").unwrap()],
            [-9.0, 8.0, 12.0],
        );
        close(
            solved.points[skel.find("LeftHand").unwrap()],
            [71.0, 148.0, 0.0],
        );
        close(
            solved.points[skel.find("RightHand").unwrap()],
            [-71.0, 148.0, 0.0],
        );
        assert_eq!(solved.skipped, 0);
    }

    #[test]
    fn root_yaw_and_reference_prefix() {
        let skel = Skeleton::axis_neuron(false);
        let mut channels = vec![0.0; 180];
        channels[3] = 90.0;
        let solved = skel.solve(&channels);
        close(
            solved.points[skel.find("RightUpLeg").unwrap()],
            [0.0, 96.0, 9.0],
        );

        let mut prefixed = vec![1.0; 6];
        prefixed.extend_from_slice(&channels);
        let solved = skel.solve(&prefixed);
        assert_eq!(solved.skipped, 6);
        close(
            solved.points[skel.find("RightUpLeg").unwrap()],
            [0.0, 96.0, 9.0],
        );
        assert_eq!(skel.channel_count(), 180);
        assert_eq!(prefixed.len(), 186);
    }

    #[test]
    fn displacement_frame_uses_the_same_root_layout() {
        let skel = Skeleton::axis_neuron(true);
        let mut channels = vec![0.0; 360];
        channels[9] = 90.0;
        let solved = skel.solve(&channels);
        assert_eq!(solved.skipped, 6);
        close(
            solved.points[skel.find("RightUpLeg").unwrap()],
            [0.0, 96.0, 9.0],
        );
    }

    #[test]
    fn parser_reads_hierarchy_and_stops_at_motion() {
        let text = r#"
HIERARCHY
ROOT Hips
{
    OFFSET 0.0 0.0 0.0
    CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation
    JOINT Spine
    {
        OFFSET 0.0 10.0 0.0
        CHANNELS 3 Yrotation Xrotation Zrotation
        End Site
        {
            OFFSET 0.0 5.0 0.0
        }
    }
}
MOTION
Frames: 1
Frame Time: 0.016667
1 2 3 4 5 6 7 8 9
"#;
        let skel = parse_hierarchy(text).unwrap();
        assert_eq!(skel.joints.len(), 3);
        assert!(!skel.with_displacement);
        assert_eq!(skel.joints[1].name, "Spine");
        assert_eq!(skel.joints[1].parent, 0);
        assert_eq!(skel.joints[2].name, "Spine_end");
        assert!(skel.joints[2].end_site);
        let solved = skel.solve(&[0.0; 9]);
        close(solved.points[0], [0.0, 0.0, 0.0]);
        close(solved.points[1], [0.0, 10.0, 0.0]);
        close(solved.points[2], [0.0, 15.0, 0.0]);
    }
}
