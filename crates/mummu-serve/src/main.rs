//! The `mummu-serve` binary: read the environment, hand it to the library.
//!
//! Everything that used to live here is now `mummu_serve` the library (see
//! `lib.rs`) so the Tauri desktop shell can run the same routers in-process.
//! This file is only the env-to-argument translation and the process exit
//! code, and it is deliberately the *only* place that reads `MUMMU_ADDR` /
//! `MUMMU_OLLAMA_ADDR` — a library that reached for the environment behind
//! its caller's back would give the app a second, invisible config surface.
//!
//! Configuration (env): `MUMMU_ADDR` (default `0.0.0.0:8095`),
//! `MUMMU_OLLAMA_ADDR` (default `0.0.0.0:11435`; `off` disables),
//! `MUMMU_MODELS_DIR` (default `./models`), `MUMMU_BACKEND`,
//! `MUMMU_FORCE_CPU`.

use std::process::ExitCode;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> ExitCode {
    // Raise the Windows system timer to 1 ms for this process. This was
    // theory six of the ~28 ms/layer readback stall (two default quanta,
    // 2 x 15.625 ms — a suspicious fit) and it measurably changed NOTHING:
    // the stall turned out to be the GPU fence surfacing at the first CPU
    // touch of wgpu's deferred-mapped readback bytes (see mapped-wait-probe
    // and the drain in nn/moe.rs). Kept anyway: any real timed park in the
    // wgpu poll path rounds to 1 ms instead of 15.6 ms under it, and the
    // only cost is slightly higher idle power.
    #[cfg(windows)]
    {
        #[link(name = "winmm.dll", kind = "raw-dylib", modifiers = "+verbatim")]
        unsafe extern "system" {
            fn timeBeginPeriod(u_period: u32) -> u32;
        }
        // SAFETY: plain WinMM call, documented to accept 1; the matching
        // timeEndPeriod is deliberately skipped — the raised resolution
        // should last exactly as long as the process.
        unsafe {
            timeBeginPeriod(1);
        }
    }

    // Demote the gemm herd before flex ever touches the global rayon pool:
    // the pool's spinners at NORMAL priority starved every microsecond-
    // scale service thread in the process — cubecl's unnamed poll thread
    // (which signals readback-map completions; measured 26.5 -> 8.7 ms of
    // per-layer fence latency once only the named DSD threads were
    // boosted), the device servers, and the drain workers. BELOW_NORMAL
    // for the herd inverts that: services preempt, and an uncontended box
    // still gives the pool every core.
    {
        #[cfg(windows)]
        fn demote_current_thread() {
            #[link(name = "kernel32.dll", kind = "raw-dylib", modifiers = "+verbatim")]
            unsafe extern "system" {
                fn GetCurrentThread() -> isize;
                fn SetThreadPriority(handle: isize, priority: i32) -> i32;
            }
            // SAFETY: plain kernel32 calls on the current thread's pseudo
            // handle; -1 = THREAD_PRIORITY_BELOW_NORMAL.
            unsafe {
                SetThreadPriority(GetCurrentThread(), -1);
            }
        }
        #[cfg(not(windows))]
        fn demote_current_thread() {}
        if let Err(e) = rayon::ThreadPoolBuilder::new()
            .start_handler(|_| demote_current_thread())
            .build_global()
        {
            eprintln!("[mummu-serve] rayon pool was already initialized ({e}); gemm herd keeps default priority");
        }
    }

    let addr = std::env::var("MUMMU_ADDR")
        .unwrap_or_else(|_| mummu_serve::DEFAULT_ADDR.into());
    let shim_addr = std::env::var("MUMMU_OLLAMA_ADDR")
        .unwrap_or_else(|_| mummu_serve::DEFAULT_OLLAMA_ADDR.into());

    let root = mummu_serve::prepare_models_root().expect("models dir must be creatable");
    mummu_serve::log_device_policy(&root);

    match mummu_serve::serve(&addr, Some(&shim_addr)).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[mummu-serve] {e}");
            ExitCode::FAILURE
        }
    }
}
