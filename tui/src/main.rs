//! Terminal viewer for the Axis Neuron BVH middleware.
//!
//! `main.py` binds a ZMQ PUB socket (`tcp://*:5556` by default) and publishes
//! multipart messages `[b"bvh", json]`. This process subscribes with CONNECT.
//! Pass `--bind` only when the publisher is the side that connects.

mod skeleton;
mod stream;
mod ui;

use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use clap::Parser;
use ratatui::crossterm::event::{self, Event, KeyEventKind};

use skeleton::parse_hierarchy;
use stream::{pump, Live};
use ui::{App, Update};

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Terminal skeleton for the Axis Neuron BVH ZMQ stream",
    after_help = "\
The middleware binds PUB on tcp://*:5556 and publishes topic bvh.
This viewer connects to tcp://127.0.0.1:5556. Use --bind only if the
publisher connects to you instead of binding.

  python main.py --udp-port 7001
  cargo run --manifest-path tui/Cargo.toml"
)]
struct Args {
    /// ZMQ endpoint. Connecting requires a host (127.0.0.1), not *.
    #[arg(long, default_value = "tcp://127.0.0.1:5556")]
    endpoint: String,

    /// Bind the SUB socket instead of connecting to the middleware.
    #[arg(long)]
    bind: bool,

    /// BVH file whose HIERARCHY poses the skeleton.
    /// Without this, the built-in Axis Neuron dummy (59 joints) is chosen
    /// from the first frame: 354/360 channels include displacement, otherwise
    /// rotation only. A longer frame's leading floats are a reference block.
    #[arg(long)]
    hierarchy: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let preset = load_hierarchy(args.hierarchy.as_deref())?;

    let stop = Arc::new(AtomicBool::new(false));
    let live = Arc::new(Mutex::new(Live::default()));
    let worker = {
        let stop = Arc::clone(&stop);
        let live = Arc::clone(&live);
        let endpoint = args.endpoint.clone();
        let bind = args.bind;
        thread::Builder::new()
            .name("zmq".into())
            .spawn(move || pump(&endpoint, bind, live, stop))?
    };

    let mut app = App::new(args.endpoint, args.bind, preset);
    let mut terminal = ratatui::init();
    let result = run(&mut terminal, &mut app, &live);
    ratatui::restore();
    stop.store(true, Ordering::Relaxed);
    let _ = worker.join();
    result?;
    Ok(())
}

fn load_hierarchy(
    path: Option<&std::path::Path>,
) -> Result<Option<skeleton::Skeleton>, Box<dyn std::error::Error>> {
    let Some(path) = path else {
        return Ok(None);
    };
    let text = fs::read_to_string(path)?;
    let skel = parse_hierarchy(&text).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: {e}", path.display()),
        )
    })?;
    Ok(Some(skel))
}

fn run(
    terminal: &mut ratatui::DefaultTerminal,
    app: &mut App,
    live: &Mutex<Live>,
) -> io::Result<()> {
    loop {
        let update = {
            let g = live.lock().unwrap_or_else(|e| e.into_inner());
            Update {
                frame: g.frame.clone(),
                count: g.count,
                fps: g.fps,
                connected: g.connected,
                fatal: g.fatal.clone(),
                bad_payloads: g.bad_payloads,
            }
        };
        app.apply(update);
        terminal.draw(|f| app.draw(f))?;

        if event::poll(Duration::from_millis(33))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Release && app.on_key(key) {
                    break;
                }
            }
        }
    }
    Ok(())
}
