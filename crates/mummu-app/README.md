# mummu-app

A Tauri v2 desktop shell that hosts `mummu-serve` **natively on the Windows
host**, replacing the WSL2 Docker deployment.

It is not a rewrite and not a second UI. The Rust side starts the existing
axum server in-process (`mummu_serve::serve_on`) and the window points at
`http://127.0.0.1:8095/`, which is the same `ui.html` the server has always
served. Every endpoint, every wire format, and the ollama-compatibility shim
are byte-for-byte what the container served.

## Why native

Inside the WSL2 container only CUDA ever reached a GPU:

- The **AMD Radeon integrated GPU is invisible** to the container. wgpu can
  only enumerate what the guest's driver stack exposes, and mesa's `dzn`
  (Vulkan-on-D3D12) is the only Vulkan that reaches WSL2 — which is also why
  the Docker image has to build without `fusion` and `vulkan-spirv` (dzn
  panics under burn-fusion's stream ordering).
- Every kernel launch that *did* reach the RTX 4070 Ti SUPER paid the
  **GPU-PV round trip**, a per-launch cost that decode — thousands of tiny
  launches per response — is maximally sensitive to.

On the host, both adapters are real Vulkan devices, the paravirtualization
layer is gone, and this crate builds with mummu-serve's **default** feature
set (`fusion` + `vulkan-spirv`) — the parity-validated configuration the
container had to give up.

Backend selection is unchanged and still lives in the engine:
`MUMMU_BACKEND` (`cuda` | `wgpu`/`gpu` | `cpu`) and `MUMMU_FORCE_CPU` are
authoritative, and unset still means "probe the adapters". Nothing here
hardcodes a device.

## Running

```powershell
cargo run -p mummu-app                  # debug: keeps a console for the logs
cargo build -p mummu-app --release      # release: windowed, no console
cargo tauri build                       # MSI/NSIS installers (run in crates/mummu-app)
```

On launch the app binds its listeners, opens a window on the local server,
and installs a tray icon with **Show / Hide / Quit**. Closing the window (or
Quit) drains both listeners and stops the server, which is what frees the
resident model's VRAM and releases the port.

It runs entirely as the invoking user: **no admin rights**, no service
install, no firewall rule as long as the bind stays on loopback.

## Configuration

| Env                     | Default            | What                                                     |
| ----------------------- | ------------------ | -------------------------------------------------------- |
| `MUMMU_APP_ADDR`        | `127.0.0.1:8095`   | native API + chat UI bind address                        |
| `MUMMU_APP_OLLAMA_ADDR` | `127.0.0.1:11435`  | ollama shim bind; `off` / `disabled` / `0` / empty = off |
| `MUMMU_MODELS_DIR`      | `./models`         | model cache root (registry layout)                       |
| `MUMMU_BACKEND`         | unset (auto)       | `cuda` \| `wgpu`/`gpu` \| `cpu`                          |
| `MUMMU_FORCE_CPU`       | unset              | `1` pins the CPU backend (wins over all)                 |

Everything else in [mummu-serve's README](../mummu-serve/README.md#configuration)
— `MUMMU_QUANT`, `MUMMU_GPU_BUDGET_GB`, `MUMMU_PACK`, `MUMMU_TIERS`, the FFN
skip knobs — applies unchanged; the app reads none of them itself.

`MUMMU_ADDR` / `MUMMU_OLLAMA_ADDR` are honored as fallbacks so a host that
already exports the headless server's variables keeps working, but the
`MUMMU_APP_*` spellings win. The defaults differ **on purpose**: the headless
binary defaults to `0.0.0.0` because it lives in a container with its own
network namespace, while a desktop app that published a model runner to the
LAN merely by being double-clicked would be a nasty surprise. Widening the
bind is an explicit decision — and it is exactly the decision the Caddy
integration below requires.

Because `mummu_serve::serve_on` takes bound listeners rather than addresses,
`MUMMU_APP_ADDR=127.0.0.1:0` is also valid: the OS picks a free port and the
window opens on whatever it got.

## Caddy integration

The public hostname `mummu.basicautomation.io` is fronted by the Caddy
container in `D:\Docker Containers\caddy` (`caddy_file/Caddyfile`, mounted at
`/etc/caddy`). Today it proxies to `wireguard-client`, the container the
Dockerized mummu-serve shares a network namespace with:

```caddyfile
############################################################
# Mummu (mummu-serve — Rust/Burn model runner + chat UI)
############################################################
mummu.basicautomation.io {
	# Ollama-compatible shim under /ollama/ — handle_path strips the
	# prefix, so /ollama/api/tags reaches the shim as /api/tags.
	redir /ollama /ollama/
	handle_path /ollama/* {
		reverse_proxy http://wireguard-client:11435
	}
	handle {
		reverse_proxy http://wireguard-client:8095
	}
	import cf_tls
}

############################################################
# Mummu ollama-compatible API shim (NDJSON, /api/tags etc.)
############################################################
mummu-api.basicautomation.io {
	reverse_proxy http://wireguard-client:11435
	import cf_tls
}
```

Once the server moves to the host, `wireguard-client` no longer has anything
listening on 8095/11435 — Caddy has to reach **out of Docker to the Windows
host** instead.

**This repo does not make that change.** `D:\Docker Containers` is out of
scope here; what follows is the exact edit to apply there by hand.

### 1. Widen the app's bind

A loopback listener is unreachable from a container. Set, in the environment
the app is launched from (a shortcut's target, a Task Scheduler action, or a
`setx` for the user):

```
MUMMU_APP_ADDR=0.0.0.0:8095
MUMMU_APP_OLLAMA_ADDR=0.0.0.0:11435
```

Windows Firewall will prompt on first bind (or add the rules ahead of time —
this needs an elevated shell **once**, and the app itself still never does):

```powershell
New-NetFirewallRule -DisplayName "mummu 8095" -Direction Inbound -Action Allow `
  -Protocol TCP -LocalPort 8095 -Profile Private
New-NetFirewallRule -DisplayName "mummu 11435" -Direction Inbound -Action Allow `
  -Protocol TCP -LocalPort 11435 -Profile Private
```

### 2. Caddyfile: `wireguard-client` → `host.docker.internal`

Replace both upstream hosts in the two blocks above — four `reverse_proxy`
lines in total:

```diff
 mummu.basicautomation.io {
 	redir /ollama /ollama/
 	handle_path /ollama/* {
-		reverse_proxy http://wireguard-client:11435
+		reverse_proxy http://host.docker.internal:11435
 	}
 	handle {
-		reverse_proxy http://wireguard-client:8095
+		reverse_proxy http://host.docker.internal:8095
 	}
 	import cf_tls
 }

 mummu-api.basicautomation.io {
-	reverse_proxy http://wireguard-client:11435
+	reverse_proxy http://host.docker.internal:11435
 	import cf_tls
 }
```

Nothing else in either block changes. In particular the `handle_path`
prefix-stripping stays as-is — the shim still expects `/api/tags`, not
`/ollama/api/tags` — and `import cf_tls` is untouched.

### 3. Make sure `host.docker.internal` resolves

Docker Desktop provides it to containers automatically, so on this host the
diff above is usually all that is needed. If Caddy logs
`dial tcp: lookup host.docker.internal: no such host`, pin it explicitly in
`D:\Docker Containers\caddy\docker-compose.yaml` — the service already has an
`extra_hosts` block, so this is one more line:

```yaml
                extra_hosts:
                        - "unifi-controller:192.168.1.1"
                        - "host.docker.internal:host-gateway"
```

Then `docker compose up -d` in that directory. (A fixed LAN address for the
Windows host works too and is more robust across Docker restarts, at the cost
of hardcoding an IP: `reverse_proxy http://192.168.1.x:8095`.)

### 4. Reload and verify

```powershell
docker exec -w /etc/caddy caddy caddy reload           # picks up the Caddyfile
curl.exe -sS https://mummu.basicautomation.io/api/health
curl.exe -sS https://mummu-api.basicautomation.io/api/tags
```

`/api/health` should now report the host's adapters — both the NVIDIA card
and the AMD iGPU — where the container only ever reported one.

### Streaming note

Both surfaces stream (SSE on `/api/chat`, NDJSON on the shim). Caddy does not
buffer proxied responses, and the server already sends
`X-Accel-Buffering: no`, so no `flush_interval` tuning is needed. If a
future intermediary does buffer, `reverse_proxy` grows
`flush_interval -1`.

### Retiring the container

The Dockerized runner is the `mummu` service in
`D:\Docker Containers\compose\ai.yaml`. It runs with
`network_mode: service:wireguard-client`, which is exactly why Caddy
addresses it as `wireguard-client:8095` — the container has no network
identity of its own. Once the host app is answering through Caddy, stop it:

```powershell
docker compose -f "D:\Docker Containers\compose\ai.yaml" stop mummu
```

Three of its environment settings need a decision on the host rather than a
straight copy:

- **`MUMMU_BACKEND: cuda` — drop it.** CUDA was the *only* correct GPU path
  inside the container: no NVIDIA Vulkan ICD reaches Docker Desktop/WSL2, and
  the mesa `dzn` (Vulkan-on-D3D12) route computes garbage on GiB-sized weight
  buffers. On the host, unset means "probe the adapters", which is the whole
  point of the move — the parity-validated wgpu/Vulkan stack takes over and
  both GPUs are visible. Set it again only to A/B the two backends, and only
  in a `--features cuda` build.
- **`MUMMU_GPU_BUDGET_GB: "9"` — keep it.** The reasoning is unchanged and if
  anything stronger: the 16 GB card is shared with the desktop, and now with
  this app's own webview as well.
- **`MUMMU_MODELS_DIR: /models` — repoint it.** `/models` is a *bind* mount,
  not a named volume, so the weights already live on the host at whatever
  `MUMMU_DATA_DIR` resolves to for that stack (`docker inspect -f
  '{{range .Mounts}}{{.Source}}{{end}}' mummu` prints the path while the
  container still exists). Point `MUMMU_MODELS_DIR` straight at it: the
  on-disk registry layout is identical, so every installed model is picked up
  as-is — nothing to copy, nothing to re-download.

## Layout

| Path                       | What                                                             |
| -------------------------- | ---------------------------------------------------------------- |
| `src/main.rs`              | the shell: bind, serve, window, tray, graceful shutdown           |
| `tauri.conf.json`          | Tauri config; `app.windows` is empty — the window is built in code so it can be pointed at the port that was actually bound |
| `ui/`                      | the bundled fallback page, shown only if the local server is unreachable; the real chat UI is served by mummu-serve |
| `capabilities/default.json`| core permissions only — the chat UI uses no Tauri IPC             |
| `icons/`                   | generated with `cargo tauri icon`                                 |
| `gen/`                     | tauri-build output, git-ignored                                   |
