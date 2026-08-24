// No console window on a release launch — the app is a tray/window app, not
// a CLI. Debug builds keep the console so the `[mummu-serve]` startup lines
// (adapter inventory, device policy, listening addresses) stay visible.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! mummu-app — the Windows desktop shell around `mummu-serve`.
//!
//! Why it exists: in the WSL2 container only CUDA ever reached a GPU. The
//! AMD integrated adapter was invisible to the container entirely, and every
//! kernel launch that did reach the NVIDIA card paid the GPU-PV round trip.
//! Hosting the same axum server in a native Windows process puts wgpu in
//! front of the real Vulkan drivers, so both adapters are enumerable and the
//! paravirtualization layer is gone. Nothing about the server changes: this
//! process runs `mummu_serve`'s routers, unmodified, over the same
//! process-wide model slot, so every endpoint and wire format is identical
//! to the container's.
//!
//! The window is not a second UI. It points at `http://127.0.0.1:<port>/`,
//! which is the very `ui.html` the server has always served — one source of
//! truth, and what the browser sees is what the app sees.
//!
//! Configuration (env):
//!
//! | var                     | default          | meaning                          |
//! |-------------------------|------------------|----------------------------------|
//! | `MUMMU_APP_ADDR`        | `127.0.0.1:8095` | native API + UI bind address     |
//! | `MUMMU_APP_OLLAMA_ADDR` | `127.0.0.1:11435`| ollama shim bind (`off` disables)|
//! | `MUMMU_MODELS_DIR`      | `./models`       | model store                      |
//! | `MUMMU_BACKEND`         | auto             | `cuda` \| `wgpu`/`gpu` \| `cpu`  |
//! | `MUMMU_FORCE_CPU`       | unset            | `1` pins the CPU backend         |
//!
//! `MUMMU_ADDR` / `MUMMU_OLLAMA_ADDR` are honored as fallbacks so a host that
//! already exports the server's variables keeps working, but the `APP_`
//! spellings win: the defaults deliberately differ (loopback here, `0.0.0.0`
//! for the headless binary), and a desktop app that silently published a
//! model runner to the LAN because some unrelated shell had `MUMMU_ADDR` set
//! would be a nasty surprise. Backend selection is *not* re-implemented here
//! — `MUMMU_BACKEND` stays authoritative inside the engine, which is the
//! only thing that knows which adapters exist.

use std::net::IpAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tauri::async_runtime::JoinHandle;
use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Manager, RunEvent, WebviewUrl, WebviewWindowBuilder};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

/// Loopback by default: the app must never publish a model runner to the
/// LAN just by being launched. Reaching it from outside (Caddy, see
/// README.md) is an explicit `MUMMU_APP_ADDR=0.0.0.0:8095` decision.
const DEFAULT_ADDR: &str = "127.0.0.1:8095";
const DEFAULT_OLLAMA_ADDR: &str = "127.0.0.1:11435";

/// The running server: a trigger to stop it and the task to wait on. Both
/// are `Option` because shutdown is one-shot — `RunEvent::Exit` fires once,
/// but the tray's Quit path can reach it by more than one route.
#[derive(Default)]
struct Server {
    shutdown: Mutex<Option<oneshot::Sender<()>>>,
    task: Mutex<Option<JoinHandle<std::io::Result<()>>>>,
}

impl Server {
    /// Ask the listeners to drain and wait for them, briefly. A generation
    /// in flight holds the model slot, not the accept loop, so this is a
    /// short wait in practice; the timeout is there so a wedged task can
    /// never keep the process alive after the user asked it to close.
    fn stop(&self) {
        if let Some(tx) = self.shutdown.lock().unwrap_or_else(|e| e.into_inner()).take() {
            let _ = tx.send(());
        }
        let task = self.task.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(task) = task {
            let drained = tauri::async_runtime::block_on(async {
                tokio::time::timeout(Duration::from_secs(5), task).await
            });
            if drained.is_err() {
                eprintln!("[mummu-app] server did not drain in 5s — exiting anyway");
            }
        }
    }
}

fn main() {
    // Our own runtime rather than tauri's implicit one, so the flavor is
    // stated rather than inherited: mummu is CPU-bound across cores and
    // parks whole threads in `spawn_blocking` (hub downloads, device
    // probes, model loads). A current-thread runtime would deadlock the
    // first time a blocking load ran while a request was streaming.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("mummu-app")
        .build()
        .expect("tokio runtime");
    tauri::async_runtime::set(runtime.handle().clone());

    let server = Arc::new(Server::default());
    let setup_server = Arc::clone(&server);

    let app = tauri::Builder::default()
        .setup(move |app| {
            start(app.handle(), &setup_server)?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to build the Mummu app");

    app.run(move |_handle, event| {
        // Closing the last window exits (tauri's default) and lands here.
        // Stopping the server on the way out is what frees the model's
        // VRAM and releases port 8095 for the next launch.
        if matches!(event, RunEvent::Exit) {
            server.stop();
        }
    });
}

/// Bind, serve, and open the window onto whatever port we actually got.
fn start(app: &AppHandle, server: &Arc<Server>) -> Result<(), Box<dyn std::error::Error>> {
    let root = mummu_serve::prepare_models_root()?;

    // Enumerating adapters spins up wgpu and takes a noticeable moment on a
    // two-GPU host. Off the main thread, so the window is not held hostage
    // to a device probe; the log line lands a beat later.
    std::thread::spawn(move || mummu_serve::log_device_policy(&root));

    // Bind before the window is created, not after: with the listener
    // already bound, the webview's first request queues in the accept
    // backlog instead of racing the server task and failing with a
    // connection-refused page.
    let addr = env_addr("MUMMU_APP_ADDR", "MUMMU_ADDR", DEFAULT_ADDR);
    let api = tauri::async_runtime::block_on(mummu_serve::bind(&addr))?;
    let url = local_url(&api)?;

    let shim_addr = env_addr("MUMMU_APP_OLLAMA_ADDR", "MUMMU_OLLAMA_ADDR", DEFAULT_OLLAMA_ADDR);
    let shim = if mummu_serve::shim_disabled(&shim_addr) {
        None
    } else {
        // A shim that can't bind (a real ollama already on the port, a
        // second copy of this app) is not fatal — the same call the
        // headless binary makes, with the same "keep serving" answer.
        match tauri::async_runtime::block_on(mummu_serve::bind(&shim_addr)) {
            Ok(listener) => Some(listener),
            Err(e) => {
                eprintln!("[mummu-app] ollama shim: {e} — shim disabled");
                None
            }
        }
    };

    let (tx, rx) = oneshot::channel::<()>();
    let task = tauri::async_runtime::spawn(async move {
        mummu_serve::serve_on(api, shim, async move {
            // A dropped sender means the app is tearing down without a
            // clean `stop()`; treat it as a stop request either way.
            let _ = rx.await;
        })
        .await
    });
    *server.shutdown.lock().unwrap_or_else(|e| e.into_inner()) = Some(tx);
    *server.task.lock().unwrap_or_else(|e| e.into_inner()) = Some(task);

    WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url.parse()?))
        .title("Mummu")
        .inner_size(1180.0, 820.0)
        .min_inner_size(520.0, 420.0)
        .build()?;

    tray(app)?;
    Ok(())
}

/// Show / Hide / Quit in the notification area. Best-effort: a host with the
/// tray unavailable logs and runs windowed, rather than refusing to start.
fn tray(app: &AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    let show = MenuItem::with_id(app, "show", "Show", true, None::<&str>)?;
    let hide = MenuItem::with_id(app, "hide", "Hide", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&show, &hide, &quit])?;

    let mut builder = TrayIconBuilder::with_id("mummu")
        .tooltip("Mummu")
        .menu(&menu)
        .show_menu_on_left_click(true)
        .on_menu_event(|app: &AppHandle, event| {
            let Some(window) = app.get_webview_window("main") else {
                return;
            };
            match event.id.as_ref() {
                "show" => {
                    let _ = window.show();
                    let _ = window.unminimize();
                    let _ = window.set_focus();
                }
                "hide" => {
                    let _ = window.hide();
                }
                // Exiting runs the `RunEvent::Exit` arm in `main`, which
                // drains the listeners — quitting from the tray stops the
                // server exactly like closing the window does.
                "quit" => app.exit(0),
                _ => {}
            }
        });
    if let Some(icon) = app.default_window_icon() {
        builder = builder.icon(icon.clone());
    }
    if let Err(e) = builder.build(app) {
        eprintln!("[mummu-app] tray unavailable: {e}");
    }
    Ok(())
}

/// The URL to point the window at. A wildcard bind (`0.0.0.0`, for Caddy)
/// is not an address you can *connect* to, so the window follows loopback
/// while the listener stays wide.
fn local_url(listener: &TcpListener) -> std::io::Result<String> {
    let addr = listener.local_addr()?;
    let host = match addr.ip() {
        IpAddr::V4(v4) if v4.is_unspecified() => "127.0.0.1".to_owned(),
        IpAddr::V6(v6) if v6.is_unspecified() => "[::1]".to_owned(),
        IpAddr::V4(v4) => v4.to_string(),
        IpAddr::V6(v6) => format!("[{v6}]"),
    };
    Ok(format!("http://{host}:{}/", addr.port()))
}

/// `primary` if set at all, else `fallback` if set and non-empty, else the
/// built-in default.
///
/// The asymmetry is deliberate. An empty value is meaningful in the primary
/// slot — `MUMMU_APP_OLLAMA_ADDR=` is one of the spellings `shim_disabled`
/// reads as "off" — but in the fallback slot it is almost always an
/// inherited-and-cleared variable, and letting that win would turn a stray
/// empty `MUMMU_ADDR` into a failed bind at startup.
fn env_addr(primary: &str, fallback: &str, default: &str) -> String {
    if let Ok(value) = std::env::var(primary) {
        return value;
    }
    match std::env::var(fallback) {
        Ok(value) if !value.trim().is_empty() => value,
        _ => default.to_owned(),
    }
}
