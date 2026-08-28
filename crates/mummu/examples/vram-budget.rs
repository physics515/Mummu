//! Sanity-check `backend::video_memory()` against reality.
//!
//! The query reaches DXGI through hand-written vtable offsets, so a wrong
//! slot count would call a different function and return plausible-looking
//! nonsense. Printing it next to the adapter's total, and next to whatever
//! nvidia-smi says, is how that gets caught.
fn main() {
    let inv = mummu::backend::inventory();
    for g in &inv.gpus {
        println!(
            "adapter: {} total={:?} MiB",
            g.name,
            g.vram_bytes.map(|b| b >> 20)
        );
    }
    match mummu::vram::memory() {
        None => println!(
            "
NVML unavailable"
        ),
        Some(m) => println!(
            "
NVML global: total {:>6} MiB  used {:>6} MiB  free {:>6} MiB",
            m.total >> 20,
            m.used >> 20,
            m.free >> 20
        ),
    }
    match mummu::backend::video_memory() {
        None => println!("\nno budget available on this platform/driver"),
        Some(v) => println!(
            "\nbudget      {:>6} MiB\ncurrent use {:>6} MiB\nheadroom    {:>6} MiB",
            v.budget >> 20,
            v.current_usage >> 20,
            v.headroom() >> 20
        ),
    }
}
