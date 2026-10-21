//! Ratatui view: status, the posed skeleton, and the live channel readout.

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Style, Stylize};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Circle, Line as Bone};
use ratatui::widgets::{Block, BorderType, Paragraph, Sparkline};
use ratatui::Frame;

use crate::skeleton::{Channel, Joint, Skeleton, SolvedFrame};
use crate::stream::BvhFrame;

pub struct Update {
    pub frame: Option<BvhFrame>,
    pub count: u64,
    pub fps: f64,
    pub connected: bool,
    pub fatal: Option<String>,
    pub bad_payloads: u64,
}

struct View {
    yaw: f64,
    pitch: f64,
    zoom: f64,
    show_fingers: bool,
    /// Index into [`Skeleton::selectable`], not into the flat joint list.
    selected: usize,
    paused: bool,
}

impl Default for View {
    fn default() -> Self {
        Self {
            yaw: 0.6,
            pitch: 0.25,
            zoom: 1.0,
            show_fingers: true,
            selected: 0,
            paused: false,
        }
    }
}

pub struct App {
    endpoint: String,
    bind: bool,
    connected: bool,
    fatal: Option<String>,
    bad_payloads: u64,
    fps: f64,
    recv_count: u64,
    seen: u64,
    frame: Option<BvhFrame>,
    skeleton: Option<Skeleton>,
    solved: Option<SolvedFrame>,
    hierarchy_locked: bool,
    warning: Option<String>,
    view: View,
    hips_mm: VecDeque<i32>,
}

impl App {
    pub fn new(endpoint: String, bind: bool, skeleton: Option<Skeleton>) -> Self {
        Self {
            hierarchy_locked: skeleton.is_some(),
            endpoint,
            bind,
            connected: false,
            fatal: None,
            bad_payloads: 0,
            fps: 0.0,
            recv_count: 0,
            seen: 0,
            frame: None,
            skeleton,
            solved: None,
            warning: None,
            view: View::default(),
            hips_mm: VecDeque::new(),
        }
    }

    pub fn apply(&mut self, up: Update) {
        self.connected = up.connected;
        self.fatal = up.fatal;
        self.bad_payloads = up.bad_payloads;
        self.fps = up.fps;
        self.recv_count = up.count;
        if self.view.paused || up.count == self.seen {
            return;
        }
        self.seen = up.count;
        if let Some(frame) = up.frame {
            self.ingest(frame);
        }
    }

    pub fn on_key(&mut self, key: KeyEvent) -> bool {
        let n = self
            .skeleton
            .as_ref()
            .map(|s| s.selectable().len())
            .unwrap_or(0);
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => return true,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return true,
            KeyCode::Left => self.view.yaw -= 0.08,
            KeyCode::Right => self.view.yaw += 0.08,
            KeyCode::Up => self.view.pitch = (self.view.pitch + 0.05).clamp(-1.3, 1.3),
            KeyCode::Down => self.view.pitch = (self.view.pitch - 0.05).clamp(-1.3, 1.3),
            KeyCode::Char('+') | KeyCode::Char('=') => {
                self.view.zoom = (self.view.zoom * 1.12).min(6.0);
            }
            KeyCode::Char('-') | KeyCode::Char('_') => {
                self.view.zoom = (self.view.zoom / 1.12).max(0.25);
            }
            KeyCode::Char('f') => self.view.show_fingers = !self.view.show_fingers,
            KeyCode::Char(' ') => self.view.paused = !self.view.paused,
            KeyCode::Tab | KeyCode::Char(']') => {
                if n > 0 {
                    self.view.selected = (self.view.selected + 1) % n;
                }
            }
            KeyCode::BackTab | KeyCode::Char('[') => {
                if n > 0 {
                    self.view.selected = (self.view.selected + n - 1) % n;
                }
            }
            KeyCode::Char('1') => {
                self.view.yaw = 0.0;
                self.view.pitch = 0.0;
            }
            KeyCode::Char('3') => {
                self.view.yaw = std::f64::consts::FRAC_PI_2;
                self.view.pitch = 0.0;
            }
            KeyCode::Char('5') => {
                self.view.yaw = 0.0;
                self.view.pitch = 1.15;
            }
            KeyCode::Char('7') => {
                self.view.yaw = 0.6;
                self.view.pitch = 0.25;
            }
            KeyCode::Char('r') => {
                let selected = self.view.selected;
                let paused = self.view.paused;
                let show_fingers = self.view.show_fingers;
                self.view = View::default();
                self.view.selected = selected;
                self.view.paused = paused;
                self.view.show_fingers = show_fingers;
            }
            _ => {}
        }
        false
    }

    pub fn draw(&self, frame: &mut Frame) {
        let area = frame.area();
        let rows = Layout::vertical([
            Constraint::Length(3),
            Constraint::Fill(1),
            Constraint::Length(3),
        ])
        .split(area);

        frame.render_widget(self.header(), rows[0]);
        frame.render_widget(self.footer(area.width), rows[2]);

        let cols = if area.width >= 88 {
            Layout::horizontal([Constraint::Fill(1), Constraint::Length(40)]).split(rows[1])
        } else {
            Layout::vertical([Constraint::Fill(2), Constraint::Length(14)]).split(rows[1])
        };
        self.draw_stage(frame, cols[0]);
        self.draw_side(frame, cols[1]);
    }

    fn ingest(&mut self, frame: BvhFrame) {
        if !frame.channels.is_empty() && !self.hierarchy_locked {
            let disp = matches!(frame.channels.len(), 354 | 360);
            let have = self.skeleton.as_ref().map(|s| s.with_displacement);
            if have != Some(disp) {
                self.skeleton = Some(Skeleton::axis_neuron(disp));
            }
        }
        if let Some(skel) = &self.skeleton {
            if !frame.channels.is_empty() {
                self.warning = channel_warning(skel, frame.channels.len(), self.hierarchy_locked);
                let solved = skel.solve(&frame.channels);
                if let Some(p) = solved.points.first() {
                    self.hips_mm.push_back((p[1] * 10.0).round() as i32);
                    while self.hips_mm.len() > 90 {
                        self.hips_mm.pop_front();
                    }
                }
                let n = skel.selectable().len();
                if n == 0 {
                    self.view.selected = 0;
                } else if self.view.selected >= n {
                    self.view.selected %= n;
                }
                self.solved = Some(solved);
            }
        }
        self.frame = Some(frame);
    }

    fn mode_word(&self) -> &'static str {
        if self.bind {
            "bind"
        } else {
            "connect"
        }
    }

    fn header(&self) -> Paragraph<'static> {
        let (label, fg, bg) = self.status_colors();
        let mut spans = vec![
            Span::styled(format!(" {label} "), Style::new().fg(fg).bg(bg).bold()),
            Span::raw("  "),
        ];
        if self.recv_count >= 2 {
            spans.push(Span::styled(
                format!("{:>5.1}", self.fps),
                Style::new().fg(Color::Green).bold(),
            ));
            spans.push(Span::styled(" fps", Style::new().fg(Color::DarkGray)));
        } else {
            spans.push(Span::styled("  —  fps", Style::new().fg(Color::DarkGray)));
        }

        if let Some(frame) = &self.frame {
            let age = match age_secs(frame.t) {
                Some(s) if s < 10.0 => format!("{:>4.0} ms", s * 1000.0),
                Some(s) => format!("{s:>4.1} s"),
                None => "   —".to_string(),
            };
            spans.push(Span::raw("   "));
            spans.push(Span::styled(age, Style::new().fg(Color::White)));
            spans.push(Span::raw("   "));
            spans.push(Span::styled(
                format!("#{}", frame.frame),
                Style::new().fg(Color::White).bold(),
            ));
            let avatar = if frame.avatar.is_empty() {
                "—".to_string()
            } else {
                frame.avatar.clone()
            };
            spans.push(Span::raw("   "));
            spans.push(Span::styled(avatar, Style::new().fg(Color::Cyan)));
            if let (Some(skel), Some(solved)) = (&self.skeleton, &self.solved) {
                spans.push(Span::raw("   "));
                spans.push(Span::styled(
                    layout_label(skel, frame.channels.len(), solved.skipped),
                    Style::new().fg(Color::White),
                ));
            }
        }

        spans.push(Span::styled(
            format!("   {} {}", self.endpoint, self.mode_word()),
            Style::new().fg(Color::DarkGray),
        ));

        Paragraph::new(Line::from(spans)).block(chrome())
    }

    fn status_colors(&self) -> (&'static str, Color, Color) {
        if self.frame.is_none() && self.fatal.is_some() {
            ("ERR", Color::White, Color::Red)
        } else if self.view.paused {
            ("PAUSE", Color::Black, Color::Yellow)
        } else if self.frame.is_none() {
            ("WAIT", Color::White, Color::Rgb(55, 58, 68))
        } else if age_secs(self.frame.as_ref().map(|f| f.t).unwrap_or(0.0)).unwrap_or(0.0) > 1.0 {
            ("STALE", Color::Black, Color::Yellow)
        } else {
            ("LIVE", Color::Black, Color::Rgb(80, 200, 120))
        }
    }

    fn footer(&self, width: u16) -> Paragraph<'static> {
        let fingers = if self.view.show_fingers { "on" } else { "off" };
        let help = if width < 100 {
            format!("arrows orbit   -+ zoom   tab joint   f:{fingers}   q quit")
        } else {
            format!(
                "arrows orbit   -/+ zoom   tab joint   f fingers:{fingers}   1 front  3 side  5 top  7 3/4   space pause   r reset   q quit"
            )
        };
        Paragraph::new(Line::from(help).fg(Color::DarkGray))
            .alignment(Alignment::Center)
            .block(chrome())
    }

    fn draw_stage(&self, frame: &mut Frame, area: Rect) {
        let selected = self.solved.as_ref().and_then(|_| self.selected_name());
        let title = match &selected {
            Some(name) => format!("skeleton · {name}"),
            None => "skeleton".to_string(),
        };
        let mut block = titled(title);
        if self.solved.is_some() {
            block = block.title_bottom(self.camera_line().centered());
        }

        let Some(skel) = &self.skeleton else {
            frame.render_widget(self.waiting(block), area);
            return;
        };
        let Some(solved) = &self.solved else {
            frame.render_widget(self.waiting(block), area);
            return;
        };
        if solved.points.is_empty() {
            frame.render_widget(self.waiting(block), area);
            return;
        }

        let plot_w = area.width.saturating_sub(2).max(1) as f64;
        let plot_h = area.height.saturating_sub(2).max(1) as f64;
        let bounds = view_bounds(plot_w, plot_h, self.view.zoom);
        let origin = solved.points[0];
        let scene = scene(
            skel,
            solved,
            Shot {
                origin,
                yaw: self.view.yaw,
                pitch: self.view.pitch,
                bounds,
                show_fingers: self.view.show_fingers,
                selected: selected.as_deref(),
            },
        );
        let grid = floor_lines(origin, self.view.yaw, self.view.pitch);

        let canvas = Canvas::default()
            .block(block)
            .marker(Marker::Braille)
            .x_bounds([bounds[0], bounds[1]])
            .y_bounds([bounds[2], bounds[3]])
            .paint(move |ctx| {
                for (x1, y1, x2, y2) in &grid {
                    ctx.draw(&Bone {
                        x1: *x1,
                        y1: *y1,
                        x2: *x2,
                        y2: *y2,
                        color: Color::Rgb(42, 46, 56),
                    });
                }
                for item in &scene.bones {
                    ctx.draw(&Bone {
                        x1: item.x1,
                        y1: item.y1,
                        x2: item.x2,
                        y2: item.y2,
                        color: item.color,
                    });
                }
                for dot in &scene.dots {
                    ctx.draw(&Circle {
                        x: dot.x,
                        y: dot.y,
                        radius: dot.r,
                        color: dot.color,
                    });
                }
                if let Some(label) = &scene.label {
                    ctx.print(
                        label.x,
                        label.y,
                        Line::from(label.text.clone()).fg(Color::Yellow),
                    );
                }
            });
        frame.render_widget(canvas, area);
    }

    fn waiting(&self, block: Block<'static>) -> Paragraph<'static> {
        let text = if let Some(err) = &self.fatal {
            format!(
                "\nZMQ error\n\n{err}\n\n{}\n{}",
                self.endpoint,
                self.mode_word()
            )
        } else if !self.connected {
            format!(
                "\nconnecting\n\n{}\n{}\ntopic bvh",
                self.endpoint,
                self.mode_word()
            )
        } else {
            format!(
                "\nwaiting for frames\n\n{}\n{}\ntopic bvh\n\npython main.py --udp-port 7001",
                self.endpoint,
                self.mode_word()
            )
        };
        Paragraph::new(text)
            .alignment(Alignment::Center)
            .fg(Color::DarkGray)
            .block(block)
    }

    fn camera_line(&self) -> Line<'static> {
        let yaw = wrap_deg(self.view.yaw);
        let pitch = self.view.pitch.to_degrees();
        Line::from(format!(
            "yaw {yaw:>4.0}°   pitch {pitch:>4.0}°   {:>3.1}x",
            self.view.zoom
        ))
        .fg(Color::DarkGray)
    }

    fn draw_side(&self, frame: &mut Frame, area: Rect) {
        let block = titled("pose");
        let inner = block.inner(area);
        frame.render_widget(block, area);
        if inner.width == 0 || inner.height == 0 {
            return;
        }
        if self.hips_mm.len() < 2 || inner.height < 8 {
            frame.render_widget(self.pose_paragraph(), inner);
            return;
        }
        let parts = Layout::vertical([Constraint::Fill(1), Constraint::Length(4)]).split(inner);
        frame.render_widget(self.pose_paragraph(), parts[0]);
        let y = self
            .solved
            .as_ref()
            .and_then(|s| s.points.first())
            .map(|p| p[1])
            .unwrap_or(0.0);
        let label = Paragraph::new(Line::from(format!("hips Y {y:.1} cm")).fg(Color::DarkGray));
        let bits = Layout::vertical([Constraint::Length(1), Constraint::Fill(1)]).split(parts[1]);
        frame.render_widget(label, bits[0]);
        let data = spark_data(&self.hips_mm);
        frame.render_widget(
            Sparkline::default()
                .data(&data)
                .style(Style::new().fg(Color::Cyan)),
            bits[1],
        );
    }

    fn pose_paragraph(&self) -> Paragraph<'static> {
        Paragraph::new(self.pose_lines())
    }

    fn pose_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![
            section("STREAM"),
            kv("endpoint", self.endpoint.clone()),
            kv("socket", self.mode_word().to_string()),
            kv("received", self.recv_count.to_string()),
        ];
        if self.bad_payloads > 0 {
            lines.push(Line::from(format!("bad json {}", self.bad_payloads)).fg(Color::Yellow));
        }
        if let Some(w) = &self.warning {
            lines.push(Line::from(w.clone()).fg(Color::Yellow));
        }
        if let Some(err) = &self.fatal {
            if self.frame.is_some() {
                lines.push(Line::from(err.clone()).fg(Color::Red));
            }
        }

        let Some(skel) = &self.skeleton else {
            lines.push(Line::from(""));
            lines.push(Line::from("skeleton follows the first frame").fg(Color::DarkGray));
            return lines;
        };
        let Some(solved) = &self.solved else {
            lines.push(Line::from(""));
            lines.push(Line::from("waiting for a pose").fg(Color::DarkGray));
            return lines;
        };

        lines.push(Line::from(""));
        lines.push(section("ROOT"));
        if let Some(p) = solved.points.first() {
            lines.push(kv("world", triplet(*p)));
        }
        if let (Some(joint), Some(local)) = (skel.joints.first(), solved.locals.first()) {
            if let Some(v) =
                triplet_of(joint, local, &[Channel::Xpos, Channel::Ypos, Channel::Zpos])
            {
                lines.push(kv("XYZ", triplet(v)));
            }
            if let Some(v) =
                triplet_of(joint, local, &[Channel::Yrot, Channel::Xrot, Channel::Zrot])
            {
                lines.push(kv("YXZ", triplet(v)));
            }
        }

        lines.push(Line::from(""));
        lines.push(section("BODY"));
        for (label, name, color) in [
            ("head", "Head", Color::White),
            ("R hand", "RightHand", Color::Rgb(255, 159, 67)),
            ("L hand", "LeftHand", Color::Rgb(72, 219, 251)),
            ("R foot", "RightFoot", Color::Rgb(255, 159, 67)),
            ("L foot", "LeftFoot", Color::Rgb(72, 219, 251)),
        ] {
            if let Some(i) = skel.find(name) {
                if let Some(p) = solved.points.get(i) {
                    lines.push(world_row(label, color, *p));
                }
            }
        }

        let selectable = skel.selectable();
        if let Some(&idx) = selectable.get(self.view.selected) {
            let joint = &skel.joints[idx];
            lines.push(Line::from(""));
            lines.push(section("JOINT"));
            lines.push(kv(
                "name",
                format!(
                    "{}  {}/{}",
                    joint.name,
                    self.view.selected + 1,
                    selectable.len()
                ),
            ));
            let parent = if joint.parent >= 0 {
                skel.joints
                    .get(joint.parent as usize)
                    .map(|j| j.name.clone())
                    .unwrap_or_else(|| "—".into())
            } else {
                "—".into()
            };
            lines.push(kv("parent", parent));
            let local = solved.locals.get(idx).map(Vec::as_slice).unwrap_or(&[]);
            for (ch, v) in joint.channels.iter().zip(local.iter()) {
                lines.push(kv(ch.label(), format!("{v:7.1} {}", ch.unit())));
            }
            if let Some(p) = solved.points.get(idx) {
                lines.push(kv("world", triplet(*p)));
                if joint.parent >= 0 {
                    if let Some(pp) = solved.points.get(joint.parent as usize) {
                        lines.push(kv("bone", format!("{:7.1} cm", dist(*p, *pp))));
                    }
                }
            }
        }
        lines
    }

    fn selected_name(&self) -> Option<String> {
        let skel = self.skeleton.as_ref()?;
        let idx = *skel.selectable().get(self.view.selected)?;
        Some(skel.joints[idx].name.clone())
    }
}

fn spark_data(samples: &VecDeque<i32>) -> Vec<u64> {
    let min = samples.iter().copied().min().unwrap_or(0);
    samples
        .iter()
        .map(|v| v.saturating_sub(min) as u64)
        .collect()
}

fn channel_warning(skel: &Skeleton, n: usize, custom: bool) -> Option<String> {
    let total = skel.channel_count();
    if n < total {
        return Some(format!("stream has {n} channels, skeleton wants {total}"));
    }
    if !custom && !matches!(n, 180 | 186 | 354 | 360) {
        return Some(format!(
            "unexpected {n} channels; assuming a rotation-only skeleton"
        ));
    }
    None
}

fn layout_label(skel: &Skeleton, n: usize, skipped: usize) -> String {
    let mode = if skel.with_displacement {
        "displacement"
    } else {
        "rotation"
    };
    if n < skel.channel_count() {
        format!("{n} ch, need {}", skel.channel_count())
    } else if skipped > 0 {
        format!("{n} ch · {mode} + {skipped} ref")
    } else {
        format!("{n} ch · {mode}")
    }
}

fn age_secs(t: f64) -> Option<f64> {
    if t <= 0.0 {
        return None;
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_secs_f64();
    Some((now - t).max(0.0))
}

fn section(title: &str) -> Line<'static> {
    Line::from(Span::styled(
        title.to_string(),
        Style::new().fg(Color::Cyan).bold(),
    ))
}

fn kv(label: &str, value: String) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<8} "), Style::new().fg(Color::DarkGray)),
        Span::styled(value, Style::new().fg(Color::White)),
    ])
}

fn world_row(label: &str, color: Color, p: [f64; 3]) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<8} "), Style::new().fg(color)),
        Span::styled(triplet(p), Style::new().fg(Color::White)),
    ])
}

fn triplet(p: [f64; 3]) -> String {
    format!("{:>7.1} {:>7.1} {:>7.1}", p[0], p[1], p[2])
}

fn triplet_of(joint: &Joint, local: &[f64], wanted: &[Channel]) -> Option<[f64; 3]> {
    let mut out = [0.0; 3];
    for (i, w) in wanted.iter().enumerate() {
        let pos = joint.channels.iter().position(|c| c == w)?;
        out[i] = *local.get(pos)?;
    }
    Some(out)
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

fn wrap_deg(rad: f64) -> f64 {
    let mut d = rad.to_degrees() % 360.0;
    if d > 180.0 {
        d -= 360.0;
    }
    if d < -180.0 {
        d += 360.0;
    }
    d
}

fn chrome() -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(Color::Rgb(70, 74, 84)))
}

fn titled(title: impl Into<String>) -> Block<'static> {
    chrome().title(Line::from(format!(" {} ", title.into())).bold())
}

/// Half-extents, in centimetres, of the default camera. The body is about
/// 170 cm tall and the arms span about 140 cm.
const REST_HALF_W: f64 = 120.0;
const REST_HALF_H: f64 = 110.0;

/// xmin, xmax, ymin, ymax. Cell aspect is about 1:2, so the world window is
/// stretched to match and then grown until the body fits.
fn view_bounds(width: f64, height: f64, zoom: f64) -> [f64; 4] {
    let zoom = zoom.clamp(0.25, 6.0);
    let mut half_w = REST_HALF_W / zoom;
    let aspect = (height.max(1.0) * 2.0) / width.max(1.0);
    let mut half_h = half_w * aspect;
    let need_h = REST_HALF_H / zoom;
    if half_h < need_h {
        let scale = need_h / half_h;
        half_w *= scale;
        half_h *= scale;
    }
    let cy = -8.0;
    [-half_w, half_w, cy - half_h, cy + half_h]
}

struct BoneSeg {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    z: f64,
    color: Color,
}

struct Dot {
    x: f64,
    y: f64,
    z: f64,
    r: f64,
    color: Color,
}

struct Label {
    x: f64,
    y: f64,
    text: String,
}

struct Scene {
    bones: Vec<BoneSeg>,
    dots: Vec<Dot>,
    label: Option<Label>,
}

#[derive(Clone, Copy)]
struct Shot<'a> {
    origin: [f64; 3],
    yaw: f64,
    pitch: f64,
    bounds: [f64; 4],
    show_fingers: bool,
    selected: Option<&'a str>,
}

fn scene(skel: &Skeleton, solved: &SolvedFrame, shot: Shot<'_>) -> Scene {
    let Shot {
        origin,
        yaw,
        pitch,
        bounds,
        show_fingers,
        selected,
    } = shot;
    let half_w = (bounds[1] - bounds[0]) / 2.0;
    let plot_cells = 80.0;
    let cm_per_cell = (half_w * 2.0 / plot_cells).max(1.0);
    let mut bones = Vec::new();
    let mut dots = Vec::new();

    for (i, joint) in skel.joints.iter().enumerate() {
        if joint.parent < 0 {
            continue;
        }
        let Some(child) = solved.points.get(i) else {
            continue;
        };
        let Some(parent_pt) = solved.points.get(joint.parent as usize) else {
            continue;
        };
        let parent_name = skel
            .joints
            .get(joint.parent as usize)
            .map(|j| j.name.as_str())
            .unwrap_or("");
        if !bone_visible(&joint.name, parent_name, show_fingers, selected) {
            continue;
        }
        let a = project(*parent_pt, origin, yaw, pitch);
        let b = project(*child, origin, yaw, pitch);
        let selected_bone = selected == Some(joint.name.as_str()) || selected == Some(parent_name);
        bones.push(BoneSeg {
            x1: a[0],
            y1: a[1],
            x2: b[0],
            y2: b[1],
            z: (a[2] + b[2]) * 0.5,
            color: if selected_bone {
                Color::Yellow
            } else {
                bone_color(&joint.name)
            },
        });
    }

    for (i, joint) in skel.joints.iter().enumerate() {
        if joint.end_site {
            continue;
        }
        if !node_visible(&joint.name, show_fingers, selected) {
            continue;
        }
        let Some(p) = solved.points.get(i) else {
            continue;
        };
        let q = project(*p, origin, yaw, pitch);
        let finger = is_finger(&joint.name);
        let (r, color) = if selected == Some(joint.name.as_str()) {
            (cm_per_cell * 0.9, Color::Yellow)
        } else if i == 0 {
            (cm_per_cell * 1.15, Color::Red)
        } else if finger {
            (cm_per_cell * 0.35, bone_color(&joint.name))
        } else {
            (cm_per_cell * 0.55, bone_color(&joint.name))
        };
        dots.push(Dot {
            x: q[0],
            y: q[1],
            z: q[2],
            r,
            color,
        });
    }

    bones.sort_by(|a, b| a.z.partial_cmp(&b.z).unwrap_or(std::cmp::Ordering::Equal));
    dots.sort_by(|a, b| a.z.partial_cmp(&b.z).unwrap_or(std::cmp::Ordering::Equal));

    let label = selected.and_then(|name| {
        let i = skel.find(name)?;
        let p = solved.points.get(i)?;
        let q = project(*p, origin, yaw, pitch);
        Some(Label {
            x: q[0] + cm_per_cell * 1.4,
            y: q[1],
            text: name.to_string(),
        })
    });

    Scene { bones, dots, label }
}

fn floor_lines(origin: [f64; 3], yaw: f64, pitch: f64) -> Vec<(f64, f64, f64, f64)> {
    let y = 0.0;
    let (ox, oz) = (origin[0], origin[2]);
    let s = 50.0;
    let seg = |a: [f64; 3], b: [f64; 3]| {
        let p = project(a, origin, yaw, pitch);
        let q = project(b, origin, yaw, pitch);
        (p[0], p[1], q[0], q[1])
    };
    vec![
        seg([ox - s, y, oz], [ox + s, y, oz]),
        seg([ox, y, oz - s], [ox, y, oz + s]),
        seg([ox - s, y, oz - s], [ox + s, y, oz - s]),
        seg([ox + s, y, oz - s], [ox + s, y, oz + s]),
        seg([ox + s, y, oz + s], [ox - s, y, oz + s]),
        seg([ox - s, y, oz + s], [ox - s, y, oz - s]),
    ]
}

fn project(p: [f64; 3], origin: [f64; 3], yaw: f64, pitch: f64) -> [f64; 3] {
    let x = p[0] - origin[0];
    let y = p[1] - origin[1];
    let z = p[2] - origin[2];
    let (sy, cy) = yaw.sin_cos();
    let x1 = cy * x + sy * z;
    let z1 = -sy * x + cy * z;
    let (sp, cp) = pitch.sin_cos();
    [x1, cp * y - sp * z1, sp * y + cp * z1]
}

fn is_finger(name: &str) -> bool {
    let name = name.strip_suffix("_end").unwrap_or(name);
    name.contains("InHand")
        || name.contains("HandThumb")
        || name.contains("HandIndex")
        || name.contains("HandMiddle")
        || name.contains("HandRing")
        || name.contains("HandPinky")
}

fn bone_color(name: &str) -> Color {
    let finger = is_finger(name);
    if name.starts_with("Right") {
        if finger {
            Color::Rgb(176, 96, 32)
        } else {
            Color::Rgb(255, 159, 67)
        }
    } else if name.starts_with("Left") {
        if finger {
            Color::Rgb(32, 130, 150)
        } else {
            Color::Rgb(72, 219, 251)
        }
    } else {
        Color::Rgb(230, 230, 230)
    }
}

fn bone_visible(child: &str, parent: &str, show_fingers: bool, selected: Option<&str>) -> bool {
    if show_fingers || !is_finger(child) {
        return true;
    }
    let Some(sel) = selected else {
        return false;
    };
    child == sel || parent == sel || child.strip_suffix("_end") == Some(sel)
}

fn node_visible(name: &str, show_fingers: bool, selected: Option<&str>) -> bool {
    if show_fingers || !is_finger(name) {
        return true;
    }
    selected == Some(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn unix_now() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
    }

    fn buffer_text(buf: &ratatui::buffer::Buffer) -> String {
        let w = buf.area().width as usize;
        let mut out = String::new();
        for (i, cell) in buf.content().iter().enumerate() {
            if i > 0 && i % w == 0 {
                out.push('\n');
            }
            out.push_str(cell.symbol());
        }
        out
    }

    fn posed() -> App {
        let mut app = App::new(
            "tcp://127.0.0.1:5556".into(),
            false,
            Some(Skeleton::axis_neuron(false)),
        );
        app.apply(Update {
            frame: Some(BvhFrame {
                frame: 42,
                avatar: "Actor".into(),
                t: unix_now(),
                channels: vec![0.0; 180],
            }),
            count: 8,
            fps: 60.0,
            connected: true,
            fatal: None,
            bad_payloads: 0,
        });
        app
    }

    #[test]
    fn waiting_screen_names_the_socket() {
        let app = App::new("tcp://127.0.0.1:5556".into(), false, None);
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
        terminal.draw(|f| app.draw(f)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(text.contains("connecting"), "{text}");
        assert!(text.contains("tcp://127.0.0.1:5556"), "{text}");
        assert!(text.contains("topic bvh"), "{text}");
    }

    #[test]
    fn t_pose_renders_pose_readout() {
        let app = posed();
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
        terminal.draw(|f| app.draw(f)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(text.contains("LIVE"), "{text}");
        assert!(text.contains("#42"), "{text}");
        assert!(text.contains("Actor"), "{text}");
        assert!(text.contains("180 ch · rotation"), "{text}");
        assert!(text.contains("Hips"), "{text}");
        assert!(text.contains("100.0"), "{text}");
        assert!(text.contains("-71.0"), "{text}");
        assert!(text.contains("71.0"), "{text}");
    }

    #[test]
    fn keys_orbit_and_quit() {
        let mut app = posed();
        let yaw = app.view.yaw;
        assert!(!app.on_key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE)));
        assert!(app.view.yaw < yaw);
        assert!(!app.on_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE)));
        assert!(!app.view.show_fingers);
        assert!(app.on_key(KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE)));
    }

    #[test]
    fn view_keeps_cell_aspect() {
        let b = view_bounds(100.0, 50.0, 1.0);
        let half_w = (b[1] - b[0]) / 2.0;
        let half_h = (b[3] - b[2]) / 2.0;
        let aspect = (50.0 * 2.0) / 100.0;
        assert!((half_h / half_w - aspect).abs() < 1e-9);
        assert!(half_h >= REST_HALF_H - 1e-6);
    }

    #[test]
    fn pause_holds_the_displayed_frame() {
        let mut app = posed();
        assert!(!app.on_key(KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE)));
        app.apply(Update {
            frame: Some(BvhFrame {
                frame: 99,
                avatar: "Other".into(),
                t: unix_now(),
                channels: vec![0.0; 180],
            }),
            count: 9,
            fps: 30.0,
            connected: true,
            fatal: None,
            bad_payloads: 0,
        });
        assert_eq!(app.frame.as_ref().unwrap().frame, 42);
        assert!((app.fps - 30.0).abs() < 1e-9);
    }
}
