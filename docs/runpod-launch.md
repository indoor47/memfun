# RunPod launch kit (swarm runs on a rented GPU)

`scripts/runpod/` rents one GPU pod on RunPod, serves a model with SGLang from a network
volume, and wires it to memfun and the dashboard on the Mac. It works with any GPU type:
B300 is the default, and H200 (or anything else) is one flag. Everything is Python stdlib.
Run it with the repo's Python 3.12 (`uv run python ...`); the macOS system `python3` (3.9)
is too old.

**Nothing in this kit spends money unless you pass both `--no-dry-run` and
`--yes-spend-money`.** Never launch without the owner's go.

## Files

| file | what it does |
|---|---|
| `scripts/runpod/runpod_kit.py` | library and CLI: `gpus`, `launch`, `spend`, `watch`, `tunnel`, `watchdog`, `status`, `logs`, `terminate` |
| `scripts/runpod/launch_pod.py` | `launch` by default; `launch_pod.py <subcommand> ...` runs any other subcommand |
| `scripts/runpod/watchdog.py`, `tunnel.py` | shortcuts for `watchdog` and `tunnel` |
| `scripts/runpod/pod_boot.sh` | in-pod boot (sent base64 in the pod env): hard stop, preflight diagnostics, sshd, weights check/download, SGLang (log mirrored to stdout) |
| `scripts/runpod/prepull_weights.sh` | download weights onto the network volume, then verify them |
| `scripts/runpod/verify_weights.py` | check every file against the Hub's sha256 / git blob sha1 |
| `tests/test_runpod_launch.py` | hermetic tests (fake transport, fake clock, spy that raises on mutation) |

## Safety model

- **Read-only by construction.** Every read goes through `ReadOnlyTransport`, which refuses
  any GraphQL document containing `mutation` (or that is not a query) and any non-GET REST
  call. `gpus`, `status`, `spend`, `tunnel`, `watch` and the `launch` dry run (the default)
  only ever get that transport. The mutating transport is built only after both flags pass.
- **Spend guards on the real path:** prints the plan and estimated max cost first; refuses
  if the estimate exceeds `--max-cost-usd` (default $40), if RunPod reports no stock, if the
  GPU is not `AVAILABLE` in the volume's data center, or if there is no SSH public key.
  Inputs are validated before any transport exists: `--max-cost-usd` finite and in
  (0, 10000], `--gpu-count` an integer 1..16, `--max-minutes` an integer 1..720 (so `nan`,
  `inf`, `0` and negatives are refused), and a price from the API that is 0, negative, NaN
  or infinite refuses the launch. `--model-repo`/`--served-name` must look like `org/name`
  over `[A-Za-z0-9._-]` (no `..`, no absolute paths); the boot script re-checks them.
- **API key:** read from `$RUNPOD_API_KEY` or `~/.config/runpod/key` (warns if the file is
  not chmod 600). It must be one line of `[A-Za-z0-9_-]`; anything else (a comment line,
  spaces, CR, NUL) is refused with a message that does not show the value, so a malformed
  key can never reach an HTTP header (where `http.client` would quote it in an exception).
  It only goes into the `Authorization` header. Every line the kit prints and every error
  it raises passes through a redactor that also hides key prefixes of 6+ characters and
  inner runs of 8+; text is redacted before it is truncated. Every exception (not only
  HTTP errors) is converted to a redacted one-line error without the original exception
  chain, and the CLI never prints a raw traceback. ssh gets an environment without the
  RunPod key or any `*KEY*`/`*TOKEN*`/`*SECRET*`/`*PASSWORD*` variable. The launch payload is checked so it cannot carry the key, and
  `watch` strips `RUNPOD_API_KEY` from the environment it gives memfun. **The account key is
  never sent into the pod.**

## Three hard stops

1. **Server side:** the pod is created with `terminateAfter = now + --max-minutes` (a
   `DateTime` field of `PodFindAndDeployOnDemandInput` in the GraphQL spec). The spec gives
   no description of its semantics, so it counts as best effort. `--no-server-terminate-after`
   leaves it out if RunPod rejects the field.
2. **In the pod (`pod_boot.sh`):** the first thing the boot script does is start a timer for
   min(`--max-minutes` from boot, the absolute deadline). When it fires, the pod deletes
   itself with the in-pod `runpodctl remove pod $RUNPOD_POD_ID` (the in-pod CLI is the
   older one; `runpodctl pod delete` is the fallback). This uses the **Pod-scoped** API key
   that RunPod injects as `RUNPOD_API_KEY` in every pod, not the account key. The same path
   runs when the weights are missing, unverified or cannot be downloaded (after a short
   `--fail-grace-minutes`, default 2, so the logs can be read), and 10 minutes
   (`--crash-grace-minutes`) after SGLang exits. The deadline itself has no grace, and it
   still fires during either grace period.
   *Limit:* if `runpodctl` is missing or fails, the fallback can only stop SGLang and exit
   PID 1. That frees the GPU from work but **may not stop billing** (RunPod can restart the
   container). That is why layers 1 and 3 exist.
3. **Local watchdog (on the Mac):** `launch` stays in the foreground after creating the pod.
   It polls the pod, prints live spend (the API's `costPerHr` times elapsed), and terminates
   the pod at the deadline, on Ctrl-C, on SIGTERM/SIGHUP (closed terminal), on an unexpected
   error, or after 5 consecutive failed polls. Termination is retried (6 tries with backoff)
   and then **verified**: the pod query must return nothing or `TERMINATED`. If it cannot be
   verified, it prints a loud warning to terminate the pod in the console and exits 3. For a
   pod launched some other way: `launch_pod.py watchdog --pod-id <id> --max-minutes N`.
   **No unwatched window:** from the create call until the watchdog takes over, any error,
   Ctrl-C or SIGTERM terminates the pod. Each pod gets a unique name
   (`<--name>-<10 hex>`); if the create call itself fails or times out after RunPod made
   the pod, the kit finds it by that name (read-only `myself { pods }`) and terminates it.
   If that lookup also fails, it prints a loud warning with the pod name and the manual
   `runpodctl pod delete` / console steps (exit 4). A network volume the kit created for a
   failed launch is never deleted; the kit prints a warning that it keeps billing storage.

Network volumes survive termination (weights and runs stay); container disk does not.

## Verified facts (2026-11-06)

| fact | source |
|---|---|
| B300 `gpuTypeId` = `NVIDIA B300 SXM6 AC` (288 GB, secure $8.99/h, community $6.94/h); H200 = `NVIDIA H200` (secure $5.29) | read-only `gpuTypes` query through the kit (`gpus`) |
| B300 secure-cloud availability per data center at ~14:40 CET: `EUR-IS-1` AVAILABLE (network volumes yes); `EU-NL-1`, `US-WA-2` UNAVAILABLE. By ~15:50 `EUR-IS-1` was UNAVAILABLE too: stock moves within minutes, so re-check right before launch | `lowestPrice.gpuTypeDatacenters` (`gpus --where`) |
| GraphQL: `podFindAndDeployOnDemand`, `podTerminate` (no selection set), `pod(input:{podId})`, `gpuTypes`, `myself { clientBalance networkVolumes }`; input has `networkVolumeId`, `dataCenterIds`, `terminateAfter`, `ports`, `env [{key,value}]`, `dockerArgs` | [docs.runpod.io GraphQL manage pods](https://docs.runpod.io/sdks/graphql/manage-pods), [graphql-spec.runpod.io](https://graphql-spec.runpod.io/) |
| Network volume create: `POST https://rest.runpod.io/v1/networkvolumes {name, size, dataCenterId}` | [REST reference](https://docs.runpod.io/api-reference/network-volumes/POST/networkvolumes) |
| In-pod env: `RUNPOD_POD_ID` "Unique Pod identifier", `RUNPOD_API_KEY` "Pod-scoped API key" | [Pod environment variables](https://docs.runpod.io/pods/templates/environment-variables) |
| Every Pod "comes preinstalled with `runpodctl` and a Pod-scoped API key"; delete: `runpodctl pod delete $RUNPOD_POD_ID` (older: `runpodctl remove pod`). **Observed in a pod (2026-11-06): the preinstalled CLI is the older one** (`Error: unknown command "pod"`, then `runpodctl remove pod` worked), so `pod_boot.sh` tries `remove pod` first | [runpodctl overview](https://docs.runpod.io/runpodctl/overview), [Manage Pods](https://docs.runpod.io/pods/manage-pods) |
| Full SSH needs TCP 22 exposed and sshd running in the container. RunPod only starts sshd for its own templates; a custom image/start command must start it itself (the SGLang image has none, so `pod_boot.sh` installs and starts it: ~8 s observed). `portMappings`/`publicIp` stay empty while the pod is initializing, image pull included | [Use SSH](https://docs.runpod.io/pods/configuration/use-ssh) |
| SGLang image `lmsysorg/sglang:v0.5.23-cu130-runtime` (pushed 2026-10-30); the docs recommend the cu130 images for B300/GB300 (SM103) / CUDA 13 | [Docker Hub tags](https://hub.docker.com/r/lmsysorg/sglang/tags), [SGLang install](https://sgl-project.github.io/get_started/install.html) |
| SGLang default `--port` 30000; `--enable-metrics` "Enable log prometheus metrics." (default off); metrics are served at `/metrics` on the serving port | [Server arguments](https://docs.sglang.io/advanced_features/server_arguments.html), [Production metrics](https://docs.sglang.io/references/production_metrics.html) |
| The `lmsysorg/sglang:v0.5.23-cu130-runtime` image has **neither `hf` nor `huggingface-cli`** (this made the first B300 launch self-terminate at boot); `pod_boot.sh` falls back to `huggingface_hub.snapshot_download` | pod logs, RTX PRO 6000 diagnostic pod, 2026-11-06 |
| Pod log stream: `GET https://api.runpod.io/v2/pods/{id}/logs` (SSE, Bearer auth; `source=container\|system`, `tail<=5000`, `since`, `Last-Event-ID`); returns 404 once the pod is removed, so logs vanish with the pod | used by hand on the diagnostic pod, 2026-11-06 |
| `Qwen/Qwen3.8-27B-FP8` exists, not gated, 81 files, 30.9 GB (sha `017b9c7a`) | HF API `GET /api/models/Qwen/Qwen3.8-27B-FP8` |

**Verify before launch:** the image tag (re-check Docker Hub), that SGLang supports the
model's architecture in that tag, and that the `terminateAfter` format is accepted (if the
create call errors on it, nothing is created; retry with `--no-server-terminate-after`).

## Run it

### 0. Prerequisites

- RunPod balance above the estimated max cost (the plan prints both).
- An SSH key pair; pass the **public** key with `--ssh-public-key ~/.ssh/<key>.pub` (the
  default `~/.ssh/id_ed25519.pub` may not exist) and the private key to `tunnel`/`watch`
  with `--identity`.

### 1. Pick a data center and create the network volume

```bash
uv run python scripts/runpod/launch_pod.py gpus --filter B300
uv run python scripts/runpod/launch_pod.py gpus --where "NVIDIA B300 SXM6 AC"
```

Create the network volume (for example `memfun-weights`, 80 GB) in a data center that shows
`AVAILABLE` and supports network volumes, in the RunPod console (Storage). A pod that uses a
network volume is pinned to that data center. (The kit can also create it at launch with
`--volume-name new --data-center DC`, but then the weights are not pre-pulled.)

### 2. Pre-pull the weights (before renting the GPU)

Disk: the FP8 weights are ~30.9 GB; `prepull_weights.sh` wants ~40 GB free; 80 GB leaves room
for runs. Start the cheapest CPU pod in the **same data center** with the volume attached at
`/workspace` (console), copy `prepull_weights.sh` and `verify_weights.py` onto it, then:

```bash
bash prepull_weights.sh Qwen/Qwen3.8-27B-FP8          # resumable: just re-run on failure
REVISION=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a bash prepull_weights.sh   # pin a revision
bash prepull_weights.sh --verify-only Qwen/Qwen3.8-27B-FP8   # re-check later
```

It downloads with `hf download` (or `huggingface-cli download`) into
`/workspace/models/Qwen/Qwen3.8-27B-FP8`, retries, then checks every file against the Hub's
sha256 (LFS) or git blob sha1 and writes `.memfun_verified`. **The volume is ready when the
script prints `VERIFIED` and `du -sh` shows ~29 GiB.** Terminate the CPU pod. The GPU pod
refuses to serve without `.memfun_verified` and terminates itself straight away. The
marker must name the same repo, and when `--model-revision` is a 40-hex commit sha the
marker's sha must equal it; an in-pod pull uses `--revision` of that pin. (Pulling on
the GPU pod is possible with `--pull-in-pod`, but costs GPU minutes and only gets
the Hub client's own integrity checks. In the pod the download uses, in order: `hf`,
`huggingface-cli`, `python3 -c 'huggingface_hub.snapshot_download(...)'`, and as a last
resort `pip install huggingface_hub` then the Python API; 3 resumable attempts, a progress
line every 20 s, and a clear reason plus the tail of `pull.log` if all fail.)

### 3. Dry run (default, read-only)

```bash
uv run python scripts/runpod/launch_pod.py --volume-name memfun-weights \
    --ssh-public-key ~/.ssh/id_ed25519.pub --max-minutes 120
uv run python scripts/runpod/launch_pod.py --gpu-type "NVIDIA H200" --volume-name memfun-weights ...
```

It resolves the GPU type (exact id or exact display name, never a fuzzy match), checks stock
and price (overall and in the volume's data center), and prints the plan, the **estimated max
cost** (the higher of list and live price x GPUs x minutes), the redacted launch payload, and
a run card labelled `DRY RUN`.

### 4. Launch (spends money)

```bash
uv run python scripts/runpod/launch_pod.py --volume-name memfun-weights \
    --ssh-public-key ~/.ssh/id_ed25519.pub --max-minutes 120 --no-dry-run --yes-spend-money
```

It prints the run card:

```
=== run card ===
pod id     : <id>
GPU        : 1 x NVIDIA B300 SXM6 AC
rate       : $8.99/h (reported by RunPod for this pod)
hard stop  : 2026-11-06 16:40:00 CET local (in 120 min, auto-terminate)
live spend : uv run python scripts/runpod/launch_pod.py spend --pod-id <id>
tunnel     : uv run python scripts/runpod/launch_pod.py tunnel --pod-id <id> --run
watch      : uv run python scripts/runpod/launch_pod.py watch --pod-id <id> -- memfun chat
env        : export MEMFUN_INFERENCE_METRICS_URL=http://127.0.0.1:18000/metrics
             export MEMFUN_POD_RATE_USD=8.99
dashboard: watch in the Swarm app
```

Then this terminal becomes the watchdog. **Leave it running**; Ctrl-C here terminates the
pod. `spend` (read-only) prints `costPerHr x uptime` at any time.

### 5. Watch: tunnel, metrics, dashboard (second terminal)

Start order: **pod (step 4) -> tunnel -> wrapper -> memfun dashboard -> iPhone app.**
`watch` does the middle three:

```bash
uv run python scripts/runpod/launch_pod.py watch --pod-id <id> --identity ~/.ssh/id_ed25519 \
    [--local-port 18000] [-- memfun chat]
```

1. Refuses local port 8434 (the owner's local server) and any port already in use.
2. Reads the pod's public SSH host/port from the API (read-only) and opens
   `ssh -N -L 127.0.0.1:<local>:127.0.0.1:30000` in its own session, so Ctrl-C in memfun does
   not kill it.
3. Readiness, read-only, with retries (`--ready-retries 60 --ready-interval 10`):
   `GET /metrics` and `GET /v1/models` through the tunnel. It warns if the served name is
   missing. A 404 on `/metrics` means SGLang is running without `--enable-metrics`.
4. Starts the memfun command (default `memfun chat`, which starts the dashboard on 0.0.0.0)
   with `MEMFUN_INFERENCE_METRICS_URL=http://127.0.0.1:<local>/metrics` and
   `MEMFUN_POD_RATE_USD=<the pod's costPerHr from the API>` set. The rate is never
   hardcoded; if the API reports none, the variable is left unset. It also sets
   `MEMFUN_SGLANG_DUMMY_KEY=unused` for the `[llm]` block below.
5. On macOS it runs `caffeinate -dimsu -w <memfun pid>`, so the Mac stays awake until memfun
   exits. Elsewhere it prints a notice; `--no-caffeinate` turns it off.
6. When memfun exits, it closes the tunnel. It does **not** stop the pod; the watchdog does.
   Ctrl-C belongs to memfun (the wrapper keeps waiting); SIGTERM/SIGHUP to the wrapper stops
   memfun, the tunnel and caffeinate and exits 143, and the pod keeps running under the
   watchdog and in-pod stop.

**Host keys.** RunPod reuses public IP:port pairs across pods, so the kit never touches
`~/.ssh/known_hosts`. ssh runs with `StrictHostKeyChecking=accept-new` against a per-pod
`UserKnownHostsFile` (a temp dir for `watch` and `tunnel --run`, deleted on exit). That is
trust on first use: the first connection to a fresh pod accepts its host key unverified,
which is the usual trade-off for ephemeral cloud pods (the host key is generated at boot and
not published anywhere the kit could check). The tunnel only carries localhost traffic to
SGLang and authenticates you by key, so the exposure is a man-in-the-middle on the first
connection reading or changing model traffic.

Starting memfun yourself instead: `tunnel --pod-id <id> --run` and paste the printed
`export ...` lines.

**memfun config: Qwen everywhere.** For the first run every memfun role (orchestrator,
decomposer, workers, auditors, reviewer) uses the same model served by SGLang on the pod,
and no cloud API key is needed. One `[llm]` block in the project `memfun.toml` does it,
since every role inherits it (`tunnel`, `watch` and `launch` print this block):

```toml
[llm]
provider = "openai"
base_url = "http://127.0.0.1:18000/v1"
model = "qwen3.8-27b-fp8"                 # the --served-name alias
api_key_env = "MEMFUN_SGLANG_DUMMY_KEY"   # NAME of a dummy env var; SGLang needs no key
```

Once per-role models land (`[llm.roles.<role>]`, PR 4, not merged yet), single roles can
be overridden; for example Claude Opus 5.5 for orchestrator/decomposer as a later
comparison arm.

**Dashboard.** With `MEMFUN_INFERENCE_METRICS_URL` set, the dashboard server proxies the
SGLang Prometheus endpoint (`/api/server-metrics`), and the inference-server strip shows
requests running and queued, KV-cache usage, prefix-cache hit rate and generation
throughput (`sglang:num_running_reqs`, `num_queue_reqs`, `token_usage`, `cache_hit_rate`,
`gen_throughput`). `MEMFUN_POD_RATE_USD` is for the pod cost meter; reading it is a
separate dashboard change. **iPhone app:** the app wraps the dashboard on the Mac, and
non-local clients need the dashboard token (`$MEMFUN_DASHBOARD_TOKEN` or
`~/.config/memfun/dashboard_token`; `memfun dashboard --remote` creates it and prints the LAN
URL). See the "Remote access: token auth" section of
`packages/memfun-cli/src/memfun_cli/dashboard/server.py` (PR #49).

### 6. Teardown

The watchdog terminates the pod at the deadline, or press Ctrl-C in the launch terminal. It
prints `verified: pod <id> is gone`; if it prints `COULD NOT VERIFY`, terminate the pod in
the console. Network volume storage is billed until you delete the volume.

**To stop a pod on purpose, use `terminate`, not a signal:**

```bash
uv run python scripts/runpod/launch_pod.py terminate --pod-id <id> --yes
```

It needs `--yes`, accepts only a well-formed pod id, terminates only that pod (retried),
verifies it, prints the final state, and exits 0 **only** when the pod is verified gone;
otherwise 3. "Gone" means absent from the account's `myself { pods }` listing or listed
with `desiredStatus` TERMINATED. A pod listed with any other status is NOT gone: RunPod
can keep a pod `RUNNING` (and billing) after its container has exited, so a frozen uptime
or a stopped container proves nothing. If the listing call fails, `pod(input)` is the
fallback; an unknown answer never counts as gone.

Stopping the `launch` process with SIGTERM is **not** a substitute. The kit does handle it
(the first SIGTERM/SIGHUP starts termination; later ones are ignored, and SIGTERM/SIGHUP/
SIGINT are ignored while termination runs; a final safety net in `launch` and `watchdog`
terminates any created pod that is not verified gone), but a driver that kills a process
group (for example `uv run` plus its child) depends on all of that and on the process not
being SIGKILLed. Drivers and operators should call `terminate` and check its exit code.

Termination is bounded: all termination passes of one process share an 8-minute budget
(each API call is capped at 30 s, including DNS hangs). While it runs, SIGTERM/SIGHUP/
SIGINT do not interrupt it; each one prints `termination of pod <id> in progress, Ns of
budget left; kill -9 <pid> to abandon`. When the budget is spent the kit prints
`COULD NOT VERIFY` with the exact `terminate --pod-id <id> --yes` command and exits 3.
From the moment the create request is sent until the pod id is known (or known not to
exist), signals are only recorded, so a Ctrl-C/SIGHUP there cannot orphan a pod. A pod
absent from the listing counts as gone only if `pod(input)` also returns null or a second
listing at least 10 s later agrees; HTTP errors are never read as "gone".

## Debugging a launch

The pod's stdout/stderr (container log) and RunPod's own events (system log: image pull,
container create/remove) are only readable **while the pod exists**: after removal the log
endpoint returns 404. So `launch` (real path) and `watch` follow the log stream into a
local file from the moment the pod exists, and the run card prints its path:

```
log file   : ~/.cache/memfun/runpod-logs/pod-<id>.log  (captured locally; survives the pod)
```

The file is created `0600` with `O_EXCL|O_NOFOLLOW` (an existing file or planted symlink is
refused; the next free `pod-<id>.N.log` is used); every line is redacted. Capture failures
never affect the watchdog. `--log-dir` changes the directory. Read the log by hand:

```bash
uv run python scripts/runpod/launch_pod.py logs --pod-id <id>              # backlog, then exit
uv run python scripts/runpod/launch_pod.py logs --pod-id <id> --follow     # until the pod is gone
uv run python scripts/runpod/launch_pod.py logs --pod-id <id> --source system --tail 500
uv run python scripts/runpod/launch_pod.py logs --pod-id <id> --follow --out pod.log --raw
```

Read-only (GET); image-pull progress lines (`<12 hex> Extracting` ...) are hidden unless
`--raw` (a count is printed); 404 means the pod is gone (exit 0); 429 waits `Retry-After`;
reconnects resume with `Last-Event-ID`; without `--follow` it stops after `--timeout` s.

What to look for (`[memfun-boot]` lines come from `pod_boot.sh`):

| pattern | meaning |
|---|---|
| system log `still fetching image` / layer lines, no container lines yet | image pull. 14.5 GB: ~1.5 min on one host, **~10 min** on the B300 host. The tunnel reports "no public TCP 22 mapping yet: still initializing" meanwhile; that is normal |
| `hard stop armed: Ns from now` | boot script started; N is what is left of the launch-time deadline |
| `---- preflight ----` block | GPU/driver, disk, python packages (sglang, torch, transformers, huggingface_hub, flashinfer), tools present, huggingface.co reachability, env var **names** (never values) |
| `sshd started` | tunnel possible from now (once RunPod publishes the port mapping) |
| `download attempt k/3 via <method>`, `download progress: <size>` | in-pod weights pull (`--pull-in-pod`) |
| `no way to download weights: missing: ...` / `all 3 download attempts ... failed` + `pull.log` tail | weights could not be fetched; the pod stays `--fail-grace-minutes` then removes itself |
| SGLang's own output (mirrored from `sglang.log`), then `SGLang exited rc=N` | SGLang crashed; `--crash-grace-minutes` to ssh in, then removal |
| `self-terminate: <reason>` + `---- last 40 lines of ...log ----` | why the pod is going away, with the tail of `pull/apt/sglang.log` |
| `terminated via 'runpodctl remove pod'` | the pod removed itself (pod-scoped key) |

**Image pull and download time eat the deadline.** The in-pod deadline and `terminateAfter`
both count from **launch** time, not from container start. In `--no-volume` mode the image
pull (1.5 to ~10 min observed) plus the ~31 GB weights download (3-5 min) both come out of
`--max-minutes` before SGLang serves anything, so give `--max-minutes` at least 15 minutes
of margin over the time you need SGLang up.

**SGLang's log is public to anyone with the pod's logs:** `pod_boot.sh` mirrors the full
`sglang.log` to stdout, so it lands in RunPod's container log and in the local capture
file. Nothing secret is in it today (SGLang runs without `--api-key`); if one is ever
added, keep it out of SGLang's printed arguments or stop the mirror.

**Hangs are bounded:** every long step in `pod_boot.sh` (download attempts, pip, apt,
`runpodctl`, `df`) runs in the background under a timeout and the script `wait`s for it,
so the TERM trap and the deadline always act (bash defers traps behind a foreground
command). A download attempt is also killed when `du -sk` of the model dir has not grown
for 5 minutes (stall) or after min(time left, 25 min); it is then retried (resumable). If
the deadline's `runpodctl` removal fails, PID 1 is told to stop all processes and exit.
Every other step after the deadline is armed that could block (python/tool probes,
`free`, `df`, `ssh-keygen`, `sshd`, `import` probes, `du`/`find` on the volume, `mkdir`/`rm`
of the model dir) is bounded the same way and is never run inside a command substitution
in the main shell, so the TERM/USR1 traps always run within seconds. A download attempt runs
in its own process group; a stall/cap kill sends TERM to the whole group, then KILL after
10 s (no `pgrep` needed). Progress counts a size change OR any file modified under the model
dir or the HF/xet caches, so a preallocated file being filled is not a stall.
A hung network volume cannot block removal either: removal is tried before any log dump,
every read of `RUN_DIR` (log tails, the weights marker) is bounded, files the script reads
back live under `/tmp`, and log files are opened by the child process, never by PID 1.
SGLang, downloads and the deadline timer each run in their own process group, and
nothing outlives PID 1.
If `runpodctl` reports success but PID 1 still runs 120 s later, the deadline path stops
the container anyway (a 0 exit is not proof the pod is gone).

**Worst case past the deadline** (defaults, `runpodctl` hanging every time): 5 attempts x
2 forms x 60 s, plus 4 x 10 s between attempts, plus 15 s SGLang grace, plus 10 s for
PID 1 to stop ~= 11 min; `runpodctl` now gets default signal handling, so each timeout
costs 60 s, not 65 s. PID 1 is told to stop with SIGUSR1 (no marker file, so a read-only
run dir cannot make it run a second removal loop). `terminateAfter` and the local
watchdog remain the other two stops.

`shellcheck` was not available when this was written; the script is covered by the
hermetic bash tests only.

**SSH on a custom image:** RunPod starts sshd only for its own templates; with a custom
start command (as here) the container must start it, which `pod_boot.sh` does (installs
`openssh-server` with apt if missing, key-only auth). The port mapping appears only after
the container is running.

## Smoke tasks and grading

The smoke task set for the first run is `evals/swarm/smoke/` (validate it with
`uv run python evals/swarm/validate_bundles.py`). Grade agent output only the way its
README's "Grading rules" section describes: owned files over a clean `repo/`, hidden tests
run from a separate grader dir with a pinned rootdir/ini and no conftest from the agent tree.

## Ports

- Local tunnel port: default 18000. **8434 is always refused.** The tunnel binds 127.0.0.1 only.
- In the pod: SGLang listens on 127.0.0.1:30000 (`--sglang-port`) and is reachable only
  through SSH. Only TCP 22 is exposed (`ports = "22/tcp"`).

## Not covered / known limits

- RunPod billing details (start-up time, volume storage) are not in the cost estimate. Live
  spend uses the API's `costPerHr`, and `spend` uses pod uptime.
- `availableGpuCounts` came back empty in the live query, so stock checks use `stockStatus`
  and the per-data-center `availability` enum.
- The kit does not create CPU pods for pre-pulling; use the console.
