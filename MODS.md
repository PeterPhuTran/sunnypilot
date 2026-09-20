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

Its three outputs are independently switchable from the on-device toggles page, because they suit
different drivers at different times: **icons on screen** (the upstream `BlindSpot` param — which
also stops the steering wheel yielding its slot), **chime on signal** (`chime` in the JSON, with
`chime_always` as its sub-option: chime on entry rather than only on a signal), and the **window
view on signal** (`camera_view`). The window view depends only on `camera_view` (plus the calibrated
zones): with the monitor itself (`enabled`) off the daemon idles without a model and the view is a plain
mirror of the signalled side -- no chevrons, chime or icons, and no CPU spent on inference (2026-09-11:
the model is shelved this way while the SoC runs the big-model warp; flip `enabled` to bring it back). Fork settings live in that JSON rather than in params because the
branch ships a prebuilt `libparams_c.so`: a new key added to `params_keys.h` is never compiled, so
it would raise `UnknownKeyName` on-device.

### 2. Privacy guards — `VBSM_PRIVACY`
In-cabin footage never leaves the device: `athenad.py` refuses on-demand uploads and clip creation
for driver-camera files, covering both the comma and sunnylink remote-procedure sockets (they share
a dispatcher). Background uploaders never sent camera footage to begin with; these guards close the
two on-demand paths. Honest failures, not faked successes.

### 2b. Parked power — `VBSM_PARK`
- `power_monitoring.py` (managed from 2026-09-08): the parked energy allowance is read from
  `/data/vbsm_park_budget_wh` (whole Wh, clamped 5..30; absent = stock 30). It is the only lever that
  changes what the device takes from the 12V battery per park (a weak hybrid AGM resting ~55% SoC;
  ~56 Wh/day simulated at stock). `should_shutdown()` makes the identical decision as upstream
  (bench: 6144-point grid) but keeps a term-by-term record of WHY.
- `hardwared.py`: when the (2-tick) shutdown decision fires it appends one JSON line to
  `/data/vbsm_shutdowns.jsonl` — reason (timer / voltage / budget / force), balance and Wh used,
  LPF + instant rail voltage, offroad seconds, both clocks (wall can be stale-from-boot) — bounded to
  the last 300 and never allowed to block the shutdown. Until now a short park could not be told
  apart as timer, voltage or budget.

- Sync-aware park (2026-09-11): while the home Pi is pulling footage it touches `/dev/shm/vbsm_sync_active`
  once per batch; a marker younger than 30 min suspends the budget and timer rules so a park never ends
  mid-sync. The 11.8 V rule and ForcePowerDown are untouched. The record carries `sync_active`.
- 2026-09-19 `VBSM_PARKWATCH` handback fix (`pandad.py`): every window end used to relaunch pandad through
  stock's crash ladder — `count` was already odd, so `recover_internal_panda()` put the panda in DFU and
  reflashed it after every park (~10 s unmonitored, 13x in two days; on the even laps `Panda.list()` was
  empty 10 ms after the reset pulse and the next lap recovered anyway). `_supervise()` now returns True on
  the window path and `main()` answers with the plain reset stock does before every launch (re-inits the
  SPI slave, wipes the window's can-speed/power-save writes; the C++ pandad's first heartbeat re-arms the
  heartbeat check either way) plus a ≤6 s wait for the app to enumerate, no count bump; the recovery
  ladder is reached only if nothing enumerates (or `flash_panda` raises, as in stock). `_parkwatch_run()`
  takes the known serial (no DFU probe on the bus), returns its end reason, skips the panda writes on a
  failed link (that was the second "cleanup failed" traceback), otherwise leaves the panda in SILENT +
  power save (stock's parked state, so a manager-exit window end cannot leave it in noOutput at SoM
  power-off), logs the in-window elapsed time on failure, and a failed window is retried once after a
  reset when ≥60 s remain. A manager exit is honoured during the enumeration wait and before the relaunch.
  Log lines: `pandad.flash_and_connect` carries `handback=true/false`; a retry logs `parkwatch retry`.
  Acceptance: no "Panda in DFU mode found" / "Done flashing" after a `parkwatch window end`; relaunch in a
  few seconds with `handback=true` and `count` unchanged.
- 2026-09-20 (`VBSM_PARK` calibration): the BMS fallback only sees the SoM (2.6–3.2 W) while the whole
  device draws 4.1–4.8 W at the panda, so the "10 Wh" knob was spending ~17 Wh per park (four 3.7 h parks ran
  the balance to the boot floor in the 09-19 review). `park_power_draw()` now scales the BMS reading by
  `PARK_SOM_TO_DEVICE = 1.6` and floors at `PARK_DRAW_FLOOR_W = 4.5`; with the knob at 10 Wh a budget-ended
  park is ~2.2 h. `MIN_ON_TIME_S` 3600 → 600: upstream's hour is for a fresh install's registration, here it
  blocked every park rule for an hour after each updater/deploy reboot while parked (09-16: 82 min awake).
  The shutdown record gains `raw_w` (the unscaled sensor reading) next to the integrated `draw_w`, so the
  calibration stays checkable. Acceptance: budget-ended parks around 2.2 h (`used_wh` ≈ 10 at `offroad_s`
  ≈ 7,500–8,500 with `raw_w` 2.5–3.9), and a park after a parked reboot ends within the hour instead of
  at `monotonic_s` ≈ 3,600. Old records carry the raw BMS value in `draw_w`; split any trend at this commit.
- 2026-09-08: the budget integrator was inert on the comma four (`get_current_power_draw()` reads a hwmon node that does not exist there, so 0 W). `park_power_draw()` now falls back to the SoM BMS reading (~2.7 W idle, a lower bound of the whole device), then a 3 W floor; the shutdown record carries `draw_w` / `draw_source`. First real record: 9.1 h parked, used 0.0 Wh, ended by the 11.8 V voltage rule.

### 3. Process reliability — `VBSM_RESTART`, `VBSM_WATCHDOG`, `VBSM_EXIT`
- `process.py`: upstream's manager never restarts a process that dies mid-session — one crash means
  the process (and, for the driving model, openpilot engagement) is gone until reboot. The manager
  now reaps a dead child and rebuilds it: 5 restarts per DRIVE, 10 s apart, then it parks with its
  crash files. Field-proven on the driving model, microphone, and sound daemons. The budget resets
  on the offroad→onroad edge in `ensure_running()`: it read "per session" but nothing ever reset it,
  and this device stays up for days across ignition cycles — so it drained silently and then a
  single crash stranded the process for every later drive, the exact failure this exists to prevent.
  A crash *loop* is still bounded within a drive by the cap and `MIN_RESTART_GAP_S`.
- `ui_watchdog.py`: detects a UI that is alive but no longer rendering (frame-beacon based, exact
  proctitle match) and kills it for the manager to rebuild. Grew three GPU duties over time — see §4.
- `VBSM_EXIT` (2026-09-20): two processes were SIGKILLed at every ignition edge. `modeld_v2/modeld.py`:
  the manager's SIGINT lands mid-frame and the normal interpreter exit then walks tinygrad's USB/AMD
  teardown (atexit + GC), which outlives the 5 s grace — so every car-off logged `sending signal 2` then
  `sending signal 9 to modeld_tinygrad`; the `KeyboardInterrupt` handler now does `time.sleep(0.2)` +
  `os._exit(0)` like the wedge path. Skipping the teardown is safe: the usbdevfs claim and lock die with
  the process, hardwared cuts the rails 120 s into the park (a power cycle clears everything), and a
  re-open inside those 120 s finds `SCRATCH_REG6` set and takes tinygrad's own full-reset path (mode1
  reset, ~1–2 s slower `open_ms`) — the same path a ui_watchdog SIGKILL kick already produces. Two
  known costs: the GPU is not put at DPM level 0 for those ≤120 s (slightly higher idle draw until
  rails-off), and a SIGINT that lands inside a synchronous libusb transfer is only honoured when the
  transfer returns, so the odd `signal 9` can still appear — expect the common case, not 100 %.
  `process_config.py`: upstream's `backup_manager` blocks in `rk.keep_time()` inside `asyncio.run` and
  never sees SIGINT, so it was SIGKILLed after 5 s at every drive start; it is now `sigkill=True`, the
  same outcome without the stall (upstream fix would be `await asyncio.sleep(1.0)` in its main loop).
  Acceptance: at car-off `modeld_tinygrad is dead with 0` and no `signal 9` for it; at drive start a
  single `sending signal 9 to backup_manager` with no preceding `signal 2`.

#### Port note — 2026-09-07 rebase onto sunnypilot `40d6afd3` (v2026.003.000)
Upstream squashed `staging` (no common ancestor with the previous base `45515f72`), so this was a
hand-merge of six managed files. Substantive changes to the fork layer:
- `modeld.py`: upstream now keeps the SoC model resident and swaps to it **in-process** on an eGPU
  exception (no restart, no no-model window). The fork's exit-for-respawn path is retired; the
  veto/strike bookkeeping (`/dev/shm/vbsm_usbgpu_veto`, `vbsm_gpu_hangs`) stays so `ui_watchdog`'s
  drive-scoped veto and mid-drive retry keep working. The PPT cap now runs inside upstream's
  `load_big()` before the transfer. Load failures: a *wedged* loader (thread still alive after the
  timeout) still exits for process replacement; a *clean* load failure falls through to upstream's
  in-process SoC path. The load-fail counter is now cleared on a successful load. Upstream's
  `ChestnutModelError` param is set/cleared as upstream does.
- `hardwared.py`: upstream replaced the bare USB-id tuples with `is_chestnut_usb_id()` and added
  `ChestnutStatus`; the rail switch's presence test goes through the helper (real device only).
  `Offroad_ChestnutUncompiled` is suppressed while the active runner is tinygrad: it keys on the
  stock big model's compiled pkl, which bundle users never have (upstream's UI already exempts the
  tinygrad runner; its status.py does not).
- `home.py`, `augmented_road_view.py`: upstream's `TextAlignment` enums replace `rl.GuiTextAlignment`;
  upstream added USB/loading chestnut icons beside the fork's voltage label.

## 4. eGPU (chestnut) integration — `VBSM_GPU_*`
The enclosure runs from the 12 V accessory outlet, which shaped everything (measured ~0.45 Ω supply
path). Full forensic history in [CHESTNUT.md](CHESTNUT.md).
- **Power cap** (`VBSM_GPU_PPT`, `modeld_v2/modeld.py`): 80 W SMU package-power limit applied
  *before* the model transfer; bounds the boost transients that browned out the supply. Tunable via
  `/data/vbsm_gpu_ppt_w`. The driving model draws 24–34 W — the cap costs nothing.
- **Load fallback ladder** (`VBSM_GPU_FALLBACK`, `modeld_v2/modeld.py`): a failed or wedged eGPU
  load exits for a manager respawn (in-process fallbacks die with the GIL when USB wedges); first
  failure gets a free retry, the second vetoes the boot (`/dev/shm/vbsm_usbgpu_veto`, cleared each
  reboot) and the respawn loads the Qualcomm bundle slot in ~2 s. Lock-contention failures capture
  the lock holder via `fuser` at the moment of failure. A hang *during* a run vetoes immediately
  with no free retry — a device that already loaded and ran for minutes is not failing on a
  cold-start transient — and modeld writes that veto itself before replacing its process, so a
  respawn can never re-attempt a browned-out GPU (previously the veto arrived from the watchdog
  0.9 s *after* the manager had already respawned). Mid-run vetoes are scoped to the DRIVE, not
  the boot: the device stays up across car restarts, so ui_watchdog clears a first-strike veto
  once offroad (making the "Restart the car to retry" alert true). Within a drive the eGPU also
  gets ONE retry: once the car rail has held the charging band (>=13.0 V for 60 s, the state the
  eGPU has demonstrably run tens of minutes in) the hang veto is cleared and the existing GPU kick
  reloads modeld — but only through its standstill + disengaged gate, so the model is never taken
  away from a moving car. The kick and retry budgets reset each drive, since the device stays up
  for days and boot-scoped counters would strand the big model until a manual reboot;
  `/dev/shm/vbsm_gpu_hangs` stays boot-scoped as a backstop (6 strikes, checked by both the
  drive-end clear and the retry gate) against a dying rail earning retries forever. Vetoes are
  classified by their payload prefix: only `hang` (a mid-run death) is clearable or retryable —
  `load`, written by the load ladder and by a loader wedged past the deadline, stays boot-scoped,
  because a device that never came up has not shown it can run on this rail.
  Note the deliberate asymmetry: rail voltage is used to *permit a retry*, never to *pre-emptively
  veto* — as a veto it was refuted (route af ran 37 min with 869 samples below 12.5 V).
- **No device probe at import** (`VBSM_GPU_READY`, `modeld_v2/modeld.py`): upstream's `compile_modeld.py` declares
  `device: str = Device.DEFAULT` as a default argument, so importing modeld_v2 makes tinygrad probe every backend
  (AMD first) before `main()` runs. On a cold boot that probe opens the AMD device before the PCIe link is up, fails
  silently, and leaks the lock fd and the libusb claim -- the 2026-09-13 "lock error 7 ms in, no other holder"
  case. modeld now pins `DEV=QCOM` before tinygrad is imported (verified: importing the module opens nothing);
  every tensor in modeld_v2 names its device explicitly and the eGPU is opened deliberately in `load_big()`.
- **Link readiness before the first open** (`VBSM_GPU_READY`, `modeld_v2/modeld.py`): the enclosure's custom
  firmware boots with PCIe off, and tinygrad's AMD open powers it on and checks the link once, with no wait. At a cold
  boot the link is often still training, so the first open fails -- and, verified on-device 2026-09-14, that failure
  leaks both tinygrad's lock fd and the libusb claim inside the process, so no later open in that process can ever
  succeed ("Failed to acquire lock", then "Resource busy"). modeld therefore reads the bridge's supply voltage and
  LTSSM straight over usbdevfs before the first open (the same read-only control transfers as `chestnut_power.py
  status`; own fd, no tinygrad, no lock). With 12 V present and the link down it writes the PCIe power bit (0xF3=1)
  once, right after the `ChestnutLoading` param so link training overlaps the vision-stream wait, re-sends it only if
  the LTSSM is still in Detect 10 s later, and polls at 2 Hz until L0 (0x78) or the budget ends: 20 s, and in any
  case 22 s after `main()` entry, so probe + the 60 s loader budget + the SoC load stay inside ui_watchdog's 90 s
  deadline (the watchdog matches `.modeld` in the cmdline, which only appears at `setproctitle()` inside `main()`, and
  polls every 2 s, so its clock starts at or after modeld's; e8b7e968 counted from process start and let a slow
  import shrink the link budget).
  Outcomes: `ready` opens; `no_12v` (three reads under 5 V, a dead outlet) and `absent` (six failed reads) skip the
  open without a strike toward the boot veto, so the kick may retry later; `timeout` (12 V present, link never
  trained) skips the open and counts a strike; `probe_error` opens anyway. Events: `chestnut preflight` (state at
  start: DEV, opened devices, lock fds, supply, LTSSM, whether the power bit was written and the LTSSM right after),
  `chestnut link ready` / `chestnut link not ready` (reason, reads, F3 writes in the wait and in total, wait, age since
  main and since exec, supply, LTSSM), `eGPU load skipped; link not ready`, and `open_ms` on
  `chestnut ppt limit` (duration of the first open). The first revision (e0c253de: 1 Hz `flash.link_up()` polling)
  wrote 0xF3=1 on every probe, live link included, and its 20 s could overrun the watchdog deadline; it was replaced
  before its first drive. First drive on e8b7e968 (2026-09-14 20:08 and 20:23 PT): both starts warm (hardwared's
  rails-on had trained the link), preflight LTSSM 0x78, ready on the first read, open 4.4 s / 1.9 s, big model from the
  first process in 26 s / 23 s, 100 % big frames, 0 % drops; the cold-boot link-down path is still unobserved.
- **Lock-contention retry, revised** (`VBSM_GPU_LOCK_RETRY`, `modeld_v2/modeld.py`): every failed open logs
  `eGPU open failed` with each sub-exception's type@file:line, whether it failed inside tinygrad's flock, the lock
  fds this process held before and after, and the elapsed time. One retry (2 s later) is attempted only when the
  failure was raised inside `flock_acquire` itself AND the process held no lock fd beforehand -- an external holder,
  the unexplained 2026-09-13 case; the unlocked fd tinygrad leaves behind is closed first. Any failure after the
  flock leaks the flock and the libusb claim and is never retried. The holder capture runs `sudo fuser -v` so
  root-owned holders are visible. Bench 2026-09-14 (car off, rails off): a failed open classifies as not retryable
  with one leaked fd; a second open in that process fails at the flock on its own fd and is refused; with an
  external holder the retry passes the flock once the holder is gone. The 2026-09-13 ladder (retry on any lock text)
  could never recover: the failing process was blocking on its own leaked descriptor. The e0c253de rule ("retry if no
  leaked fd") could never fire either: a flock refusal still leaves an unlocked fd behind.
- **Late link probe, readiness-gated kick** (`VBSM_GPU_LATE`, `modeld_v2/modeld.py` + `ui_watchdog.py`): on
  2026-09-16 a cold boot found the bridge enumerated at 5000 Mb/s from the first second but answering no control
  transfer for ~30 s (modeld's preflight and its own INA telemetry both read nothing until then); the probe called
  that `absent` after six reads in 2.5 s and the drive ran on the SoC. `absent` now requires the bridge to be missing
  from sysfs; a bridge that is listed but silent keeps the whole budget and ends as `unreadable` (no strike, like
  `absent`). After any skip (`no_12v`, `absent`, `unreadable`, `timeout`) modeld keeps reading the bridge at 1 Hz
  from a daemon thread (the same read-only usbdevfs transfers, own fd; `fcntl.ioctl` releases the GIL), writes the
  PCIe power bit with the preflight's resend rule at most three times, and writes `/dev/shm/vbsm_gpu_link_ready`
  once it has seen 12 V and L0 (events `chestnut late probe sees 12v`, `chestnut late f3`, `chestnut link ready
  late`). `/dev/shm/vbsm_gpu_link_wait` marks the skip; every modeld process clears both at start. ui_watchdog's
  automatic kick waits for the ready marker whenever the wait marker exists, so a kick can never buy a second skip.
- **Kick cooldown clock** (`VBSM_GPU_KICK`, `ui_watchdog.py`): `last_kick` started at 0.0 while `now` is
  `time.monotonic()`, which starts near zero at boot, so the 120 s cooldown read as "kicked at boot" and refused
  every kick in the first two minutes of uptime -- exactly what a cold boot into onroad offers. 2026-09-16: 40 s
  standstill from route second 9, cruise main off, every other gate open, no kick; the driver rebooted. Now `None`
  until the first kick.
- **Driver-requested retry** (`VBSM_GPU_KICK_REQUEST`, `ui_watchdog.py`): `/dev/shm/vbsm_gpu_kick_request` asks
  for the kick at the next standstill with openpilot disengaged, without the cruise-main gate and without waiting
  for the ready marker (the fresh process re-probes and writes the power bit itself). Requests older than 10 min
  are discarded, never queued. Vetoes and the two-kicks-per-drive budget still apply; a reboot stays the last resort.
- **Chestnut icon gestures** (`VBSM_GPU_HUD_TAP`, `hud_renderer.py` + `augmented_road_view.py`): everything the
  driver needs is on the device. The orange (failed) chestnut icon stays on screen while disengaged -- the only
  time a retry can fire -- and fades when engaged as before so the dmoji gets the slot back. Tap it: the retry
  marker above is written and the icon pulses slowly until ui_watchdog consumes it; tap again to cancel. Hold it
  for 2 s (the icon grows while the finger rests on it) and let go: `DoReboot`, the same param the settings page
  sets, so the manager reboots even onroad. The gesture is decided in one function (`chestnut_touch_action`,
  returns `reboot` / `request` / `cancel` / `''`) that swallows every exception, because an exception in the
  road-view touch handler takes the whole UI down. Taps on a green or loading icon do nothing; a swipe is never a
  tap. Bench (device, headless, 2026-09-16): tap on orange writes then cancels the marker, off-icon / swipe /
  green taps do nothing, 1.5 s hold does nothing, 2.1 s hold sets DoReboot in any state, an icon that is not drawn
  has no target, a broken state object is swallowed.
- **Kick only when the driver is not about to engage** (`VBSM_GPU_KICK_ARMED`, `ui_watchdog.py`): the
  standstill/disengaged kick now also requires cruise main OFF or the car in Park. A kick throws a
  working SoC model away for a ~35 s no-model reload; on 2026-09-13 it fired 10 s after the driver armed
  cruise, refused two engages and starved locationd (see below). A boot-in-Park kick is invisible.
- **Truthful big-model alerts** (`VBSM_GPU_ALERTS`, `selfdrived.py`): "Big Model Ready" only fires when
  `ChestnutActive` is True at the end of loading (it also fired 2 s after a real failure), and "active
  but no modelV2" is ignored inside the 5 s settling window (modeld marks the big model active ~2 s
  before its first published frame, which raised a false "Big Model Failed" banner). `BIG_MODEL_WARMUP_SEC`
  replaces the local `warmup_sec`.
- **Bounded locationd lockout** (`VBSM_LOC_CAP`, `locationd.py`, NEW managed file): the invalid-input
  counters are capped at limit+1. Unbounded, one 2 s burst of rejected gyro samples (the gyro/camera
  yaw-rate cross-check has nothing to compare against during a camera-odometry gap) took ~10 minutes to
  decay, during which every engage was refused with "locationd Temporary Error". Recovery after a fault
  now ends within ~30 s; a persistent fault stays flagged. Upstream-worthy.
- **Buffered message validity** (`VBSM_LOC_VALID`, `locationd.py`): `inputs_valid` also depends on
  `sm.all_valid()`, which had no hysteresis at all, so ONE message flagged invalid out of a 20 Hz stream
  dropped `inputsOK` and put a full-screen "TAKE CONTROL IMMEDIATELY / locationd Temporary Error" on the
  screen for its full 2 s. On 2026-09-15 a single camera frame desync marked one `cameraOdometry` message
  invalid; `inputsOK` was false for 60 ms and the state machine was back in ENABLED before the alert
  finished drawing, but the driver still got the warning mid-drive. Message validity now carries the same
  kind of buffer the sanity counters already had: a bad cycle counts a whole step, a good cycle gives back
  half, and the inputs are called bad at 3. Three consecutive bad messages still fault 100 ms after the
  first, a 50/50 flapping stream faults within 400 ms, and one good message clears it. The counter starts AT the
  limit, so a service that has never arrived still reads as bad from the first cycle and startup is unchanged --
  starting it at zero reported `inputsOK` true for the first two frames of a segment, which openpilot's process
  replay caught on the upstream PR. Replayed against the 2026-09-15 route: the startup episode is cycle-for-cycle
  identical to the old rule (18 cycles both ways) and the one spurious episode disappears.
  Upstream-worthy, and distinct from `VBSM_LOC_CAP`, which only touches the counter branch.
- **Watchdog GPU duties** (`VBSM_GPU_KICK`, `ui_watchdog.py`): restarts a modeld that booted before
  the enclosure enumerated (standstill + disengaged only, gated on the GPU slot holding a bundle and
  `ChestnutActive` false); SIGKILLs a load wedged past 90 s (a GIL-held process ignores everything
  else); vetoes the boot when a GPU-active modeld dies.
- **HUD status**: upstream's chestnut icon (pulsing = loading, green = big model live, orange =
  fallback) is now driven by a proper state machine whose compiled-gate checks the bundle chunk
  manifest — our `VBSM_GPU_HUD` gate patch became obsolete and was retired (see Retired below).
- **Parked power-off** (`VBSM_GPU_IDLE`, `hardwared.py` + `chestnut_power.py`): the enclosure holds
  12 V after some parks and idles at 25–40 W straight off the car battery. After 120 s of offroad the
  GPU rails are cut via the firmware's F3 switch (hardware-validated both directions: 2 A → 1 mA,
  restore retrains the link first try); restored on the offroad→onroad edge. Opt out:
  `/data/vbsm_no_gpu_idle_off`. The privileged CLI runs under `python -B` — a root interpreter
  writing bytecode into the tree silently breaks the updater's git clean.

### 5. Driver HUD — `VBSM_HUD`, `VBSM_EXP_TOGGLE`
- LKAS button (2026-09-08): back to stock sunnypilot behaviour — enables/disables MADS. Experimental/stock switching is upstream's distance-button hold (0.5 s, `cruise_helpers.py`); the fork's LKAS repurpose is off (`VBSM_LKAS_REPURPOSED = False`).
- `hud_renderer.py` set speed (2026-09-08): shown while engaged and whenever cruise is *resumable* — ACC main on with a retained set speed (`carState.cruiseState.available`) — hidden otherwise after the stock 2.5 s fade, so the driver sees what RES will bring back.
- **Persistent set speed** (`hud_renderer.py`): the cruise set-speed no longer fades 2.5 s after a
  change; it stays up whenever engaged.
- **Gap-profile chip** (`hud_renderer.py` + `augmented_road_view.py` + `selfdrived.py`): top-right
  blue-bar indicator — one bar = aggressive, two = standard, three = relaxed — reading the
  personality live from selfdrived's own state. Tap to cycle; selfdrived adopts external personality
  changes on its periodic check (upstream reads the param only at boot and from the wheel button)
  and fires the stock personality-changed alert as feedback.
- **Driving-mode badge** (`hud_renderer.py`): the bottom-left slot always shows the driving mode —
  flask = experimental, the steering wheel itself = stock — whether or not openpilot is engaged.
  Engagement is carried by opacity (dimmed when disengaged) rather than hiding the icon, and the
  wheel keeps tracking the steering angle while the flask sits still. The steer-required critical
  wheel and its exclamation mark are untouched, as are the turn-intent arrows and the blind-spot
  yield. (The original stock badge used `icons/couch.png`, which is black at 40 % alpha — it drew
  every frame and was invisible over the camera feed.)
- **LKAS button = experimental toggle** (`VBSM_EXP_TOGGLE`, `selfdrived.py` + `mads.py`): one press
  of the wheel's LKAS/LDA button flips `ExperimentalMode` (live within ~100 ms via the stack's
  param threads) and fires the stock "Experimental Mode Switched" alert. MADS's stock use of the
  same button (lateral pause) is disabled via `VBSM_LKAS_REPURPOSED` in `mads.py` — set it False
  to restore. TSS2-only signal; the upstream distance-button 0.5 s hold toggle still works too.
  Note the button also still flips the car's own stock LDA setting in the cluster.
- **Driver-monitoring face relocated** (`augmented_road_view.py` + `hud_renderer.py`): the dmoji
  moved from top-left (where the persistent set-speed kept it hidden whenever engaged) to the
  bottom-right eGPU slot; it appears once the eGPU status icon fades. The eGPU icon also lingers
  6 s after a state change (was 2.5 s) so its color is actually readable.
- **Active model names** (`layouts/home.py`): the home screen names both driving-model slots on the
  version line beside the commit hash — the SoC model, plus the big GPU model when the chestnut is
  attached (e.g. `wmiv12 · bmv4`). Short catalog names (`internalName`) only: `displayName` runs to
  56 characters and will not fit the 536 px line. Read from the two `ModelManager_ActiveBundle*`
  params at 1 Hz, never from `modelManagerSP` (which republishes a tick late and would flash the
  wrong model on a chestnut transition).
- **Parked battery voltage** (`layouts/home.py`): "12.2V" readout in the home-screen footer,
  right of the chestnut icon — pandad's peripheralState on a widget-local 2 Hz SubMaster (the
  same source the power monitor uses). Renders only while the screen is already awake; adds no
  screen-on time and no measurable power draw.

### 6. Boot & log hygiene — `VBSM_QUIET`, `VBSM_LOG_LEVEL`
- **Chunk-manifest storm fix** (`models/fetcher.py`): two catalog entries share a fileName with
  different chunk counts, flipping the same manifest twice per second forever (93 % of log volume,
  ~2 h retention). Manifests are now written only when the chunks exist locally. Reported upstream
  (sunnypilot/sunnypilot#1975).
- **Shutdown debounce** (`hardwared.py`): the offroad shutdown decision must hold for 2 consecutive
  iterations before `DoShutdown` fires — kills the race where a shutdown latched in the same
  sampling window as an ignition rise and turned a departure into a double boot.
- **`VBSM_LOG_LEVEL`** (2026-09-20): `SwagLogger.event()` (`common/logging_extra.py`) logs at ERROR
  whenever an `error` kwarg is *present*, whatever its value. `hardwared.py` passed `error=not ok` on
  `chestnut gpu rails` and `error=<bool>` on `chestnut flash done`; `modeld.py` passed `error=False` on
  `chestnut link ready`, `chestnut preflight`, `chestnut ppt limit` and `eGPU load succeeded after lock
  retry` — six success events logged as errors (an AST census of all 21 `.event(` sites carrying
  `error=` in the managed files found exactly these six; the other 15 are genuine failures). The kwarg is now passed only on failure; the failure siblings keep `error=True`.

### 7. Driver-monitoring model pin — `VBSM_DM_PKL_PIN`
- **What**: `dmonitoring_model_tinygrad.pkl.chunk01of01`, `dm_warp_1344x760_tinygrad.pkl` and
  `dm_warp_1928x1208_tinygrad.pkl` are carried as managed *binaries*, pinned to the blobs from the
  pre-`72da24ad` staging tree (`5c969f86`, `06740e07`, `193f9da9`).
- **Why**: sunnypilot staging `72da24ad` (2026-09-17, master `a5f44653`) recompiled all three
  pickles with a newer tinygrad than the `tinygrad_repo` it ships (tree `07174c66`, identical to
  master's submodule pin `f6fc4e3f`, whose `CallInfo` has five fields). The new pickles pass six, so
  `pickle.load` raises `TypeError: CallInfo.__init__() takes from 1 to 6 positional arguments but 7
  were given`, `dmonitoringmodeld` dies 5/5, selfdrived raises `commIssue` on `driverMonitoringState`
  and openpilot cannot engage. Reported upstream as sunnypilot/sunnypilot#2038. `dmonitoringmodeld.py`,
  `helpers.py`, `file_chunker.py` and the camera transforms are unchanged between the two bases, so
  the old blobs plus the shipped tinygrad is exactly the stack that ran clean on `a302344e`.
- **Guard coverage**: the port's drift guard only compares *managed* paths, so a `tinygrad_repo` bump
  alone is invisible to it. Two tripwires cover the realistic upstream fixes: the pickle blobs (any
  recompile changes them) and `dmonitoringmodeld.py`, carried byte-identical as a managed file (the
  open upstream sync sunnypilot/sunnypilot#2036 keeps these exact pickle bytes, bumps `tinygrad_repo`
  to `9cd40014` and rewrites the reader for dict-format pickles — the reader change is what stops the
  port). A silent re-break needs upstream to bump tinygrad *without* touching either, which cannot
  fix #2038, so it is not a working fix path. Belt-and-braces follow-up on master: teach
  `rebase-vbsm.yml` a drift-only `watch` list (`tinygrad_repo` tree id) next to `files`.
- **Retire when**: the port stops on any of these four paths. Before dropping them from the manifest,
  confirm the new pickles unpickle with the *shipped* `tinygrad_repo` (compare the `CallInfo`
  arity in `tinygrad_repo/tinygrad/uop/ops.py` against the pickle, or boot a bench device) and
  that the shipped reader matches the pickle format. Never pin the driving-model pickles the same
  way: those come from the model catalog, not the tree.

## Managed files (27) and markers

| File | Mods | Markers |
|---|---|---|
| `VBSM.md`, `CHESTNUT.md`, `MODS.md` | documentation | — |
| `openpilot/sunnypilot/vision_bsm.py` | §1 (additive file) | — |
| `openpilot/sunnypilot/ui_watchdog.py` | §3, §4 (additive file) | `VBSM_WATCHDOG`, `VBSM_RESTART`, `VBSM_GPU_KICK`, `VBSM_GPU_KICK_ARMED`, `VBSM_GPU_RETRY`, `VBSM_GPU_LATE`, `VBSM_GPU_KICK_REQUEST` |
| `openpilot/sunnypilot/chestnut_power.py` | §4 (additive file) | `VBSM_GPU_IDLE` |
| `openpilot/system/manager/process.py` | §3 | `VBSM_RESTART` |
| `openpilot/system/manager/process_config.py` | §1, §3 process entries, backup_manager sigkill | `VBSM_EXIT` |
| `openpilot/selfdrive/car/card.py` | §1 | — |
| `openpilot/selfdrive/selfdrived/selfdrived.py` | §1 chime, §4 big-model alerts, §5 personality re-read + LKAS toggle | `VBSM_CHIME_HOLD`, `VBSM_CONFIG`, `VBSM_GPU_ALERTS`, `VBSM_HUD`, `VBSM_EXP_TOGGLE`, `VBSM_LKAS_REPURPOSED` |
| `openpilot/sunnypilot/mads/mads.py` | §5 LKAS button freed for the toggle | `VBSM_EXP_TOGGLE` |
| `openpilot/sunnypilot/models/fetcher.py` | §6 manifest storm fix | `VBSM_QUIET` |
| `openpilot/selfdrive/ui/mici/layouts/settings/toggles.py` | §1 settings | `BigConfigControl` |
| `openpilot/selfdrive/ui/mici/onroad/augmented_road_view.py` | §1 preview, §5 tap | `BSM_STATE_PATH`, `VBSM_HUD`, `VBSM_GPU_HUD_TAP` |
| `openpilot/selfdrive/ui/mici/onroad/hud_renderer.py` | §5 | `VBSM_HUD`, `VBSM_GPU_HUD_TAP` |
| `openpilot/selfdrive/ui/mici/layouts/home.py` | §5 parked voltage | `VBSM_HUD` |
| `openpilot/system/athena/athenad.py` | §2 | `VBSM_PRIVACY` |
| `openpilot/sunnypilot/modeld_v2/modeld.py` | §4 cap, fallback ladder, readiness, lock retry, late probe; §3 exit; §6 levels | `VBSM_GPU_PPT`, `VBSM_GPU_FALLBACK`, `VBSM_GPU_READY`, `VBSM_GPU_LOCK_RETRY`, `VBSM_GPU_LATE`, `VBSM_EXIT`, `VBSM_LOG_LEVEL` |
| `openpilot/system/hardware/hardwared.py` | §2b park, §4 idle power, §6 shutdown debounce + rails/flash log level | `VBSM_GPU_IDLE`, `VBSM_PARK`, `VBSM_LOG_LEVEL` |
| `openpilot/selfdrive/pandad/pandad.py` | §2b parkwatch window + park events | `VBSM_PARKWATCH` |
| `openpilot/sunnypilot/parkwatchd.py` | §2b (additive file) | — |
| `openpilot/system/hardware/power_monitoring.py` | §2b parked energy budget | `VBSM_PARK` |
| `openpilot/selfdrive/locationd/locationd.py` | §4 bounded lockout, buffered message validity | `VBSM_LOC_CAP`, `VBSM_LOC_VALID` |
| `openpilot/selfdrive/modeld/models/dmonitoring_model_tinygrad.pkl.chunk01of01`, `dm_warp_1344x760_tinygrad.pkl`, `dm_warp_1928x1208_tinygrad.pkl` | §7 pinned binaries (blobs from the pre-`72da24ad` staging tree) | `VBSM_DM_PKL_PIN` (this file only — binaries carry no marker) |
| `openpilot/selfdrive/modeld/dmonitoringmodeld.py` | §7 tripwire only — carried byte-identical to upstream so the drift guard fires when the pickle reader changes | — |

Retired: `VBSM_COMPAT` (a modeld_v2 unpacking shim, superseded when upstream fixed the API
properly); `VBSM_GPU_HUD` (a ui_state compiled-gate patch for bundle installs, superseded by
upstream's chestnut state machine whose gate checks the bundle chunk manifest — `ui_state.py`
left the managed set with it).

## Tunables and switches

| Control | Effect |
|---|---|
| `/data/vision_bsm.json` | blind spot monitor config — `enabled` (gates the daemon), `camera_view`, `chime`, `chime_always`, calibrated `zones`; all settable from the toggles page |
| `BlindSpot` (param) | on-screen blind spot icons — the one BSM setting that is a param, because upstream owns it |
| `/data/vbsm_gpu_ppt_w` | GPU power cap in watts (default 80, clamp 40–220, 0 disables) |
| `/data/vbsm_no_gpu_idle_off` | opt out of the parked GPU power-off |
| `/dev/shm/vbsm_usbgpu_veto` | per-boot GPU veto (set automatically on failures; clears at reboot) |
| `/dev/shm/vbsm_sync_active` | touched by the Pi's footage sync per batch; fresh (<30 min) = budget/timer shutdown rules suspended, voltage rule kept |

## Operational notes

- Deploys: never kill the updater (the restart policy respawns it and the fresh cycle invalidates
  the consistency marker — a silent skip); instead HUP it and poll for the expected staged head
  *and* the marker in one on-device command, then reboot.
- A root python importing tree modules writes root-owned `__pycache__` that blocks all subsequent
  updates; clean with `find -user root` if updates fail on `git clean`.
- The device RTC resets on offline boots: every boot's first minutes log into the same stale
  window, and crash files written pre-NTP carry stale names *and* mtimes.
- **Committing managed files through the Git Data API**: tree entries carry the file mode and the API does not
  inherit it; hardcoding `100644` stripped the executable bit from `modeld.py` on 2026-09-14 (manager exit
  code 126, two drives with no model). Take each path's mode from `git ls-tree <parent> -- <path>`
  (`pending/reapply_fixes_with_modes.py`) and verify exec bits plus a running modeld after every deploy.
