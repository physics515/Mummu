fn main() {
    // Reads `tauri.conf.json`, embeds `ui/` as the fallback frontend, and
    // attaches the Windows resources (icon, manifest). The manifest asks for
    // `asInvoker` — the app binds a loopback port and writes under the
    // user's own profile, so it must never trip a UAC prompt.
    tauri_build::build();
}
