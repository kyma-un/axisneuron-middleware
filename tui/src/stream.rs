//! SUB socket for the middleware publisher.
//!
//! `main.py` binds a PUB socket and sends multipart `[b"bvh", json]`.
//! This side connects (or binds, if asked) and keeps only the latest frame.
//! `bvh.raw` shares the `bvh` prefix, so the topic frame is matched exactly.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct BvhFrame {
    #[serde(default)]
    pub frame: i64,
    #[serde(default)]
    pub avatar: String,
    #[serde(default)]
    pub t: f64,
    #[serde(default)]
    pub channels: Vec<f64>,
}

#[derive(Debug, Default)]
pub struct Live {
    pub frame: Option<BvhFrame>,
    pub count: u64,
    pub fps: f64,
    pub connected: bool,
    pub fatal: Option<String>,
    pub bad_payloads: u64,
}

pub fn bvh_body(parts: &[Vec<u8>]) -> Option<&[u8]> {
    match parts {
        [topic, body, ..] if topic.as_slice() == b"bvh" => Some(body.as_slice()),
        _ => None,
    }
}

pub fn pump(endpoint: &str, bind_socket: bool, live: Arc<Mutex<Live>>, stop: Arc<AtomicBool>) {
    let ctx = zmq::Context::new();
    let sock = match ctx.socket(zmq::SUB) {
        Ok(sock) => sock,
        Err(e) => {
            lock(&live).fatal = Some(format!("socket: {e}"));
            return;
        }
    };
    if let Err(e) = configure(&sock, endpoint, bind_socket) {
        lock(&live).fatal = Some(e);
        return;
    }
    lock(&live).connected = true;

    let mut fps = 0.0;
    let mut last: Option<Instant> = None;
    while !stop.load(Ordering::Relaxed) {
        match sock.recv_multipart(0) {
            Ok(parts) => {
                let Some(body) = bvh_body(&parts) else {
                    continue;
                };
                match serde_json::from_slice::<BvhFrame>(body) {
                    Ok(frame) => {
                        let now = Instant::now();
                        if let Some(prev) = last {
                            let dt = now.duration_since(prev).as_secs_f64();
                            if dt > 1e-4 {
                                fps = 0.9 * fps + 0.1 / dt;
                            }
                        }
                        last = Some(now);
                        let mut g = lock(&live);
                        g.count = g.count.saturating_add(1);
                        g.fps = fps;
                        g.frame = Some(frame);
                        g.fatal = None;
                    }
                    Err(e) => {
                        let mut g = lock(&live);
                        g.bad_payloads = g.bad_payloads.saturating_add(1);
                        if g.frame.is_none() {
                            g.fatal = Some(format!("json: {e}"));
                        }
                    }
                }
            }
            Err(zmq::Error::EAGAIN) | Err(zmq::Error::EINTR) => {}
            Err(e) => {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                lock(&live).fatal = Some(format!("recv: {e}"));
                thread::sleep(Duration::from_millis(200));
            }
        }
    }
}

fn configure(sock: &zmq::Socket, endpoint: &str, bind_socket: bool) -> Result<(), String> {
    sock.set_rcvhwm(8).map_err(|e| format!("rcvhwm: {e}"))?;
    sock.set_linger(0).map_err(|e| format!("linger: {e}"))?;
    sock.set_subscribe(b"bvh")
        .map_err(|e| format!("subscribe: {e}"))?;
    if bind_socket {
        sock.bind(endpoint)
            .map_err(|e| format!("bind {endpoint}: {e}"))?;
    } else {
        sock.connect(endpoint)
            .map_err(|e| format!("connect {endpoint}: {e}"))?;
    }
    sock.set_rcvtimeo(200)
        .map_err(|e| format!("rcvtimeo: {e}"))?;
    Ok(())
}

fn lock(live: &Mutex<Live>) -> MutexGuard<'_, Live> {
    live.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topic_filter_is_exact() {
        assert!(bvh_body(&[b"bvh".to_vec(), b"{}".to_vec()]).is_some());
        assert!(bvh_body(&[b"bvh.raw".to_vec(), b"{}".to_vec()]).is_none());
        assert!(bvh_body(&[b"bvh".to_vec()]).is_none());
    }

    #[test]
    fn pump_keeps_the_latest_bvh_json() {
        let ctx = zmq::Context::new();
        let publisher = ctx.socket(zmq::PUB).unwrap();
        publisher.set_linger(0).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = publisher
            .get_last_endpoint()
            .unwrap()
            .expect("endpoint is utf-8");

        let live = Arc::new(Mutex::new(Live::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let worker = {
            let live = Arc::clone(&live);
            let stop = Arc::clone(&stop);
            thread::spawn(move || pump(&endpoint, false, live, stop))
        };

        let raw = br#"{"frame":1,"avatar":"nope","t":1,"channels":[1]}"#;
        let body = br#"{"frame":7,"avatar":"Actor","t":1.5,"channels":[0.0,1.0]}"#;
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            publisher.send(b"bvh.raw".as_slice(), zmq::SNDMORE).unwrap();
            publisher.send(raw.as_slice(), 0).unwrap();
            publisher.send(b"bvh".as_slice(), zmq::SNDMORE).unwrap();
            publisher.send(body.as_slice(), 0).unwrap();
            thread::sleep(Duration::from_millis(30));
            if lock(&live).count >= 1 {
                break;
            }
        }

        let got = lock(&live).frame.clone();
        stop.store(true, Ordering::Relaxed);
        let _ = worker.join();

        let frame = got.expect("subscriber received no bvh frame");
        assert_eq!(frame.frame, 7);
        assert_eq!(frame.avatar, "Actor");
        assert_eq!(frame.channels, vec![0.0, 1.0]);
    }
}
