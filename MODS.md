# Branch Mods: Complete Index

This branch is sunnypilot staging plus a content-ported mod set for a comma four (mici). Upstream
force-pushes staging with rewritten history, so the mods are never rebased: a daily GitHub workflow
overlays the managed files onto the current upstream head as a fresh merge commit, gated by
`.github/vbsm-base.json` and a battery of verify guards (marker presence, pyflakes undefined-name
sweep, file-mode parity vs upstream). Any upstream drift in a managed file fails the port loudly
for a hand-merge — auto-merging modified driving code is how incidents happen.

Deep dives: [VBSM.md](VBSM.md) (blind spot monitor), [CHESTNUT.md](CHESTNUT.md) (eGPU debugging log).

## The mods

### 1. Camera blind spot monitor (vbsm)
A fine-tuned YOLOv10n watches the rear side windows through the cabin camera and fires the stock
"car in blindspot" chime/indicator — vehicle detection for a car with no factory BSM. Full design,
calibration data, and refuted approaches in [VBSM.md](VBSM.md).
Files: `vision_bsm.py` (daemon), `card.py` (carState injection), `selfdrived.py` (chime, marker
`VBSM`), `process_config.py` (process entries), `toggles.py` (settings), `augmented_road_view.py`
(cabin preview + chevrons).

### 2. Privacy guards — `VBSM_PRIVACY`
In-cabin footage never leaves the device: `athenad.py` refuses on-demand uploads and clip creation
for driver-camera files, covering both the comma and sunnylink remote-procedure sockets (they share
a dispatcher). Background uploaders never sent camera footage to begin with; these guards close the
two on-demand paths. Honest failures, not faked successes.

### 3. Process reliability — `VBSM_RESTART`, `VBSM_WATCHDOG`
- `process.py`: upstream's manager never restarts a process that dies mid-session — one crash means
  the process (and, for the driving model, openpilot engagement) is gone until reboot. The manager
  now reaps a dead child and rebuilds it: 3 restarts per session, 10 s apart, then it parks with its
  crash files. Field-proven on the driving model, microphone, and sound daemons.
- `ui_watchdog.py`: detects a UI that is alive but no longer rendering (frame-beacon based, exact
  proctitle match) and kills it for the manager to rebuild. Grew three GPU duties over time — see §4.

### 4. eGPU (chestnut) integration — `VBSM_GPU_*`
The enclosure runs from the 12 V accessory outlet, which shaped everything (measured ~0.45 Ω supply
path). Full forensic history in [CHESTNUT.md](CHESTNUT.md).
- **Power cap** (`VBSM_GPU_PPT`, `modeld_v2/modeld.py`): 80 W SMU package-power limit applied
  *before* the model transfer; bounds the boost transients that browned out the supply. Tunable via
  `/data/vbsm_gpu_ppt_w`. The driving model draws 24–34 W — the cap costs nothing.
- **Load fallback ladder** (`VBSM_GPU_FALLBACK`, `modeld_v2/modeld.py`): a failed or wedged eGPU
  load exits for a manager respawn (in-process fallbacks die with the GIL when USB wedges); first
  failure gets a free retry, the second vetoes the boot (`/dev/shm/vbsm_usbgpu_veto`, cleared each
  reboot) and the respawn loads the Qualcomm bundle slot in ~2 s. Lock-contention failures capture
  the lock holder via `fuser` at the moment of failure.
- **Watchdog GPU duties** (`VBSM_GPU_KICK`, `ui_watchdog.py`): restarts a modeld that booted before
  the enclosure enumerated (standstill + disengaged only, gated on the GPU slot holding a bundle and
  `UsbGpuActive` false); SIGKILLs a load wedged past 90 s (a GIL-held process ignores everything
  else); vetoes the boot when a GPU-active modeld dies.
- **HUD status** (`VBSM_GPU_HUD`, `ui_state.py`): upstream's eGPU icon (pulsing = loading, green =
  big model live, orange = SoC fallback) never rendered on bundle installs; a cached bundle pkl now
  satisfies its gate.
- **Parked power-off** (`VBSM_GPU_IDLE`, `hardwared.py` + `chestnut_power.py`): the enclosure holds
  12 V after some parks and idles at 25–40 W straight off the car battery. After 120 s of offroad the
  GPU rails are cut via the firmware's F3 switch (hardware-validated both directions: 2 A → 1 mA,
  restore retrains the link first try); restored on the offroad→onroad edge. Opt out:
  `/data/vbsm_no_gpu_idle_off`. The privileged CLI runs under `python -B` — a root interpreter
  writing bytecode into the tree silently breaks the updater's git clean.

### 5. Driver HUD — `VBSM_HUD`
- **Persistent set speed** (`hud_renderer.py`): the cruise set-speed no longer fades 2.5 s after a
  change; it stays up whenever engaged. (Trade-off: the driver-monitoring emoji shares that slot and
  hides while engaged.)
- **Gap-profile chip** (`hud_renderer.py` + `augmented_road_view.py` + `selfdrived.py`): top-right
  blue-bar indicator — one bar = aggressive, two = standard, three = relaxed — reading the
  personality live from selfdrived's own state. Tap to cycle; selfdrived adopts external personality
  changes on its periodic check (upstream reads the param only at boot and from the wheel button)
  and fires the stock personality-changed alert as feedback.

## Managed files (16) and markers

| File | Mods | Markers |
|---|---|---|
| `VBSM.md`, `CHESTNUT.md` | documentation | — |
| `openpilot/sunnypilot/vision_bsm.py` | §1 (additive file) | — |
| `openpilot/sunnypilot/ui_watchdog.py` | §3, §4 (additive file) | `VBSM_WATCHDOG`, `VBSM_GPU_KICK` |
| `openpilot/sunnypilot/chestnut_power.py` | §4 (additive file) | `VBSM_GPU_IDLE` |
| `openpilot/system/manager/process.py` | §3 | `VBSM_RESTART` |
| `openpilot/system/manager/process_config.py` | §1, §3 process entries | — |
| `openpilot/selfdrive/car/card.py` | §1 | — |
| `openpilot/selfdrive/selfdrived/selfdrived.py` | §1 chime, §5 personality re-read | `VBSM`, `VBSM_HUD` |
| `openpilot/selfdrive/ui/mici/layouts/settings/toggles.py` | §1 settings | `BigConfigControl` |
| `openpilot/selfdrive/ui/mici/onroad/augmented_road_view.py` | §1 preview, §5 tap | `BSM_STATE_PATH`, `VBSM_HUD` |
| `openpilot/selfdrive/ui/mici/onroad/hud_renderer.py` | §5 | `VBSM_HUD` |
| `openpilot/selfdrive/ui/ui_state.py` | §4 HUD gate | `VBSM_GPU_HUD` |
| `openpilot/system/athena/athenad.py` | §2 | `VBSM_PRIVACY` |
| `openpilot/sunnypilot/modeld_v2/modeld.py` | §4 cap + fallback | `VBSM_GPU_FALLBACK`, `VBSM_GPU_PPT` |
| `openpilot/system/hardware/hardwared.py` | §4 idle power | `VBSM_GPU_IDLE` |

Retired: `VBSM_COMPAT` (a modeld_v2 unpacking shim, superseded when upstream fixed the API
properly).

## Tunables and switches

| Control | Effect |
|---|---|
| `/data/vision_bsm.json` | blind spot monitor enable + config (its presence gates the daemon) |
| `/data/vbsm_gpu_ppt_w` | GPU power cap in watts (default 80, clamp 40–220, 0 disables) |
| `/data/vbsm_no_gpu_idle_off` | opt out of the parked GPU power-off |
| `/dev/shm/vbsm_usbgpu_veto` | per-boot GPU veto (set automatically on failures; clears at reboot) |

## Operational notes

- Deploys: never kill the updater (the restart policy respawns it and the fresh cycle invalidates
  the consistency marker — a silent skip); instead HUP it and poll for the expected staged head
  *and* the marker in one on-device command, then reboot.
- A root python importing tree modules writes root-owned `__pycache__` that blocks all subsequent
  updates; clean with `find -user root` if updates fail on `git clean`.
- The device RTC resets on offline boots: every boot's first minutes log into the same stale
  window, and crash files written pre-NTP carry stale names *and* mtimes.
