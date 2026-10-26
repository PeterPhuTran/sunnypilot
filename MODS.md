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
it would raise `UnknownKeyName` on-device. A file that is valid JSON but not an object (a hand edit, a tool bug)
reads as the defaults in the UI, the toggles page and selfdrived instead of crash-looping them (2026-10-19).

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
- 2026-09-21 (`VBSM_PARK` wake spend cap): the panda on a comma four enters stop mode once the SoM is off (power
  save on, SAFETY_SILENT, the parked state) and wakes on any CAN frame or SBU edge by resetting itself
  (`panda/board/main.c`, cuatro branch of the main loop; `board/sys/power_saving.h` arms EXTI on FDCAN1-3 RX
  and SBU1/2); a freshly reset panda starts in `BOOT_BOOTKICK`, so a lock, unlock, door or the car's own ECU
  chatter boots the device with no ignition. `pandad.boot_health` showed `panda_uptime_s` 15-18 with
  `ignition_line=false` on the three instrumented no-ignition boots (09-20 18:48; 09-21 08:59, 11:37); the 09-20
  19:48 boot logged no boot_health and is classified from its shutdown record (`started_seen=false`, 739 s,
  1.0 Wh). The four wakes with records ran 735/739/889/758 s and 1.00/1.00/1.13/1.00 Wh, none followed by an
  ignition inside that window, and a wake that inherits a balance from a voltage-ended park (08:59: 8.4 Wh) is
  not bounded by the boot floor: it runs until that balance is spent (up to the knob; 8.4 Wh is ~1.9 h at 4.5 W)
  or the 11.8 V rule fires. New rule in `should_shutdown()`: a boot that has seen neither ignition nor an onroad
  start may spend `PARK_WAKE_SPEND_WH = 0.4` and no more (reason `wake`, suspended by the sync marker like the
  budget); the inherited balance itself is left for the next drive, and the boot floor (a tenth of the knob) is
  unchanged, so boots that went onroad (`started_seen=true`) and their parks behave exactly as before. "Seen
  ignition" is a tmpfs marker (`/dev/shm/vbsm_boot_ignition`, touched by `calculate()` on the first ignition tick):
  a reboot clears it, a hardwared respawn mid-park does not, and it also covers an ignition boot whose startup
  stayed blocked (Always Offroad, thermal, connectivity or terms gates never set `started_seen`). `MIN_ON_TIME_S` 600 → 300; in practice
  upstream's `DELAY_SHUTDOWN_TIME_S` (300 s after hardwared's first offroad tick, ~20-25 s into boot) is the
  binding gate. A no-ignition boot now ends ~320-345 s after boot (`offroad_s` ≈ 301-320) with `used_wh`
  ≈ 0.40-0.45, reason `wake` (or `wake+voltage` when the LPF is already under 11.8 V; `budget+wake` only if the balance
  ran out while the rules were suspended, a fresh boot cannot reach it because the boot floor, a tenth of the
  knob, is above the cap; reason tokens are emitted in the fixed order force, budget, wake, voltage, timer); the bench's 340 s assumes the 4.5 W floor, the four field wakes averaged 4.7-5.05 W (the
  early-boot BMS reading is higher), so in practice the 300 s delay gate binds. Was 735-889 s and 1.0-1.1 Wh, or
  up to the inherited balance (~1.9 h for 8.4 Wh). The updater's post-boot cycle (observed ~85 s from boot
  to `git reset success`) and `tools/deploy.sh` (waits for `.overlay_consistent` before rebooting) both fit; a
  sync pass that plants the marker before the wake rule fires (~320-345 s into the boot) extends the wake for
  the pull, bounded by the pass ending, the 30-min TTL and the 11.8 V rule, as it does a drive park (09-20: the
  Pi's pass landed inside both hourly at-home wakes). Acceptance: `vbsm_shutdowns.jsonl` records with `started_seen=false`
  carry `reason` `wake` (or `wake+voltage` / `budget+wake`) at `offroad_s` ≈ 301-320 with `used_wh` ≈ 0.40-0.45
  and `boot_ignition=false` — classify by `reason`, never by `monotonic_s`; records with `started_seen=true` (or
  `boot_ignition=true`) are unchanged; the `CarBatteryCapacity` saved after a wake equals max(boot floor,
  inherited balance) minus ~0.4 Wh (0.6 Wh at knob 10 after a budget-ended park, 8.0 Wh after an 8.4 Wh inheritance).
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

### 3. Process reliability — `VBSM_RESTART`, `VBSM_WATCHDOG`, `VBSM_EXIT`, `VBSM_BOOTCAUSE`
- `process.py`: upstream's manager never restarts a process that dies mid-session — one crash means
  the process (and, for the driving model, openpilot engagement) is gone until reboot. The manager
  now reaps a dead child and rebuilds it: 5 restarts per DRIVE, 10 s apart, then it parks with its
  crash files. Field-proven on the driving model, microphone, and sound daemons. The budget resets
  on the offroad→onroad edge in `ensure_running()`: it read "per session" but nothing ever reset it,
  and this device stays up for days across ignition cycles — so it drained silently and then a
  single crash stranded the process for every later drive, the exact failure this exists to prevent.
  A crash *loop* is still bounded within a drive by the cap and `MIN_RESTART_GAP_S`.
- `ui_watchdog.py`: detects a UI that is alive but no longer rendering (frame-beacon based, exact
  proctitle match) and kills it for the manager to rebuild. Since 2026-10-19 it has one GPU duty, the
  load-wedge and freeze backstop (`VBSM_GPU_WEDGE`) — see §4.
- `VBSM_EXIT` (2026-09-20): two processes were SIGKILLed at every ignition edge. `modeld_v2/modeld.py`:
  the manager's SIGINT lands mid-frame and the normal interpreter exit then walks tinygrad's USB/AMD
  teardown (atexit + GC), which outlives the 5 s grace — so every car-off logged `sending signal 2` then
  `sending signal 9 to modeld_tinygrad`; the `KeyboardInterrupt` handler now does `time.sleep(0.2)` +
  `os._exit(0)` like the wedge path. Skipping the teardown is safe: the usbdevfs claim and lock die with
  the process, hardwared cuts the rails 120 s into the park (a power cycle clears everything), and a
  re-open inside those 120 s finds `SCRATCH_REG6` set and takes tinygrad's own full-reset path (mode1
  reset, ~1–2 s slower `open_ms`) — the same path a ui_watchdog wedge SIGKILL already produces. Upstream's
  retry cycle relies on it too: the cycle's SIGINT ends modeld at once instead of after the 5 s SIGKILL wait. Two
  known costs: the GPU is not put at DPM level 0 for those ≤120 s (slightly higher idle draw until
  rails-off), and a SIGINT that lands inside a synchronous libusb transfer is only honoured when the
  transfer returns, so the odd `signal 9` can still appear — expect the common case, not 100 %.
  `process_config.py`: upstream's `backup_manager` blocks in `rk.keep_time()` inside `asyncio.run` and
  never sees SIGINT, so it was SIGKILLed after 5 s at every drive start; it is now `sigkill=True`, the
  same outcome without the stall (upstream fix would be `await asyncio.sleep(1.0)` in its main loop).
  Acceptance: at car-off `modeld_tinygrad is dead with 0` and no `signal 9` for it; at drive start a
  single `sending signal 9 to backup_manager` with no preceding `signal 2`.
- `VBSM_BOOTCAUSE` (2026-09-20): why did the device boot? On 2026-09-20 16:47 PT the SoM came back 76 s
  after a `sudo poweroff` with no ignition; `/sys/bootinfo` said powerup `unknown reboot` after a poweroff
  of `ps_hold, keypad_reset1` — the software poweroff completed, then the panda's bootkick line brought the
  SoM back (on a comma four that line doubles as the reset; there is no separate BOOT_RESET wire as on a
  comma 3X). The journal is volatile, so two
  boot-time events now record the evidence: `hardwared` logs `boot cause` with the three `/sys/bootinfo`
  reasons, and `pandad` logs `pandad.boot_health` from the health it already reads before its first reset
  (panda uptime, `som_reset_triggered`, ignition line/CAN, harness status, heartbeat lost, power save,
  voltage, fault status; the read happens before the wrapper's own reset, the event is emitted on the first
  lap and hardwared's 5 s in, because lines sent before `logmessaged` listens are dropped by the
  non-blocking swaglog socket — the first PR D boot lost both). Reading them: a panda uptime of seconds at SoM boot with ignition off = the
  panda itself rebooted (12 V brownout) and its init kicked the SoM; a large uptime, or
  `som_reset_triggered` true (only possible after a STANDBY→BOOTKICK transition, i.e. no panda reboot
  since the last session), = an ignition/harness edge did. A small uptime with the ignition line on is
  just a harness power-up. `bootkick_tick` runs at 1 Hz: the countdowns are 20 s and 5 s.

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

#### Port note — 2026-10-14 onto sunnypilot `65feb2b6` (v2026.003.000, model selector 20)
Upstream staging moved the model selector to version 20 on 2026-10-08 (catalog
`driving_models_chestnut_v27.json`, a newer tinygrad), and the daily port stopped on nine managed
files. This was a hand-merge; substantive changes to the fork layer:
- `modeld.py`: upstream moved per-model input handling into `model_adapters.py` and renamed
  `ChestnutState` to `ChestnutGpuState`: modeld now publishes only GPU metrics on `chestnutGpuState`,
  and hardwared's new `chestnut_state_thread` reads the bridge and publishes `chestnutState`. Upstream
  also removed the `ChestnutModelError` param from `params_keys.h`, so the fork's writes of it are
  dropped (it was write-only, and an unregistered key raises `UnknownKeyName`). The fallback ladder,
  veto and late-probe logic is unchanged.
- `locationd.py`: upstream renamed `critcal_services` to `critical_services`; the `VBSM_LOC_CAP` and
  `VBSM_LOC_VALID` code follows the new name.
- `toggles.py`: upstream added descriptions to the stock controls; the blind spot controls are
  unchanged.
- `hardwared.py`, `selfdrived.py`, `card.py`, `process.py`, `home.py`, `fetcher.py`: upstream-only
  edits that merged cleanly (sentry removal, `sm.seen` guards on the localizer errors, the bridge
  state thread, the alert-pill icon callback, catalog v27).
- `chestnut_state_thread` opens the bridge over libusb next to modeld's own EP0 reads (`VBSM_GPU_READY`) and
  `chestnut_power.py`. Field check, 2026-10-20 to 10-24: all 33 eGPU starts on the ported builds were `ready` at
  the first read (wait 0.0 s, no unreadable reads, no `no_12v`), so cold-boot loads are unaffected.
- The selector bump invalidates the cached model bundles (as on 2026-09-07), so both model slots
  must be re-selected and re-downloaded after the first boot on this base.

#### Port note — 2026-10-19 onto sunnypilot `b50b82f9` (model selector 21, chestnut firmware 8516b768)
Upstream staging changed 11 managed files (21 conflict blocks), so this was a hand-merge. Substantive changes to the
fork layer:
- GPU retry: upstream's retry-once (`ChestnutRetry` + `OnroadCycleRequested`: one onroad cycle per ignition after a
  failed big-model load) is adopted unchanged. The fork's competing retry paths are retired (kick, armed gate, rail
  retry, driver request, late probe, lock retry, load-fail ladder, veto/strike files, HUD tap). New fork pieces around
  upstream's retry: the session gate, final-at-once `no_12v`/`absent`, the cold-cache budget and heal, the firmware
  gate, the exact-proctitle wedge and freeze backstop and the hardwared cycle guard (§4).
- modeld: upstream's plain-artifact loader (`load_model`, one 798 MB file for MDMV3, a boot cache `<pkl>.cache`),
  frame fusion (`fused_with_warp`), an `error_state` check after every run, `AMDDevice.wait_timeout_ms = 3000` (3 s
  signal waits are real again; the old env var had been dead since before 5fb421d0), `config_realtime_process` after
  the load (the loader now runs unpinned) and stock modeld's `setdefault('AM_POWER_LIMIT', '100')`. The DEV pin and
  the PPT export now sit above upstream's import lines, so an upstream edit there does not conflict with them.
- Experimental mode: upstream made the e2e plan permanent and deleted `ExperimentalMode`, `ExperimentalModeConfirmed`
  and `DynamicExperimentalControl` (any read or write now raises `UnknownKeyName` on the prebuilt libparams),
  `cruise_helpers.py` (the distance-button hold toggle) and the mode icons. The fork's `VBSM_EXP_TOGGLE`,
  `VBSM_LKAS_REPURPOSED` and the driving-mode badge are gone, and `mads.py` left the managed set (24 -> 23 files).
  `home.py` re-imports `time` (upstream dropped it); `toggles.py` lost the confirm page.
- UI: `layouts/settings/device.py` became the package `layouts/settings/device/` (`device_layout.py`);
  hold-to-restart imports `_engaged_confirmation_click` from there (§10). The HUD tap / 2 s hold on the chestnut icon
  is retired; the 3 s hold-to-restart covers a reboot anywhere on the road view, the icon included.
- hardwared: upstream flashes the enclosure before going onroad (onroad waits up to 30 s); the fork holds its rail
  commands while the firmware is out of date or being flashed, and refuses an onroad cycle while engaged. The
  `Offroad_ChestnutUncompiled` filter is retired (upstream no longer raises the alert).
- fetcher: upstream removed model chunking, so VBSM_QUIET's manifest-storm fix is moot; catalogs `driving_models_v25`
  and `driving_models_chestnut_v28`. The selector bump (21) resets both model slots: re-select MDMV3 and the small
  model after the first boot on this base.
- Deploy with the car off: hardwared flashes 8516b768 on the first boot, and only a power cycle without 12 V
  activates it (`chestnut flash done returncode=0`, then product `custom 8516b768-CLEAN`). With 12 V present the old
  firmware stays until the next park shutdown and the firmware gate keeps those drives on the SoC model.
- Review fixes folded in before the push (four adversarial reviews of the port): the cold-cache test computes the
  frame-fusion digest as `load_model()` does (a copied digest read a new warp pkl as warm: no big model ever), the SoC
  model's cold cache is in the watchdog budget (clamp 95 -> 155 s), a 15 s freeze backstop after the first modelV2, the
  firmware gate inside the readiness wait, the cycle guard sees MADS, a per-process retry marker in the watchdog, the
  respawn's write order, the vision-wait excess reported; the settle gate's NO_ENTRY covers `modeldLagging`; a
  profile-chip tap no longer goes home, touches over a pushed prompt or across a draw gap are not holds, the dmoji is
  not hidden by a stale chestnut fade, malformed `/data/vision_bsm.json` reads as defaults; the verify step checks
  code, not comments, for every engagement guard (§4, §5, §9, §10).

## 4. eGPU (chestnut) integration — `VBSM_GPU_*`
The enclosure runs from the 12 V accessory outlet, which shaped everything (measured ~0.45 Ω supply
path in August 2026; a 2026-10-21 fit of car-bus minus dock voltage against dock current, over 41 drives, gives
~0.23 Ω). Full forensic history in [CHESTNUT.md](CHESTNUT.md).
- **Power cap** (`VBSM_GPU_PPT`, `modeld_v2/modeld.py`): SMU package-power limit, default 80 W (the knob on this
  car: 60 W), clamp 40–220 W, from `/data/vbsm_gpu_ppt_w`; bounds the boost transients that browned out the supply.
  Since 2026-10-19 modeld exports it as `AM_POWER_LIMIT` before anything imports tinygrad: tinygrad applies that
  value inside every AMD boot and stock modeld only `setdefault`s 100 W, so the eGPU never boots above the cap.
  `apply_ppt_cap()` still writes the same limit after the bridge is up and before the transfer, and logs the
  read-back (`chestnut ppt limit` requested/applied/open_ms). `0` now means upstream's 100 W, not uncapped. The
  driving model draws 24–34 W; CTV3M pinned 57 W at the 60 W cap (2026-10-18). MDMV3 (from 2026-10-20) draws
  50–52 W at the cap with supply-current peaks of 14–14.8 A; the dock input then dips to ~9.3 V (single samples to
  ~8 V) with no supply fault and no model gap. CTV3M had the same peaks and dips (routes of 2026-10-15 to 10-18).
- **Upstream retry, session gate** (`VBSM_GPU_FALLBACK`, `modeld_v2/modeld.py`, 2026-10-19): upstream's retry-once
  is used unchanged. A failed big-model load sets `ChestnutRetry` (cleared at ignition-on and manager start) and
  `OnroadCycleRequested`; hardwared holds the car offroad for 1 s, the manager restarts every onroad process, and the
  fresh modeld tries once more. A second failure sets `ChestnutActive=False` ("Big Model Failed / Restart the car to
  retry", 20 s, and `Offroad_ChestnutModelError`). Around it the fork keeps four rules:
  (1) session gate: upstream never restarts modeld inside an onroad session, the fork does (`VBSM_RESTART`).
  `ChestnutLoading` is cleared at every offroad edge, at ignition-on and at manager start, and every modeld writes it,
  so a modeld that finds it set is a respawn: it runs the SoC model, never loads the eGPU mid-drive (25–60 s with no
  model) and never requests a cycle at speed; it reports `ChestnutActive=False` when the session had tried the big
  model, queued before its `ChestnutLoading` write (one FIFO writer), so selfdrived never reads "load done, still
  active" in between and flashes "Big Model Ready" before "Big Model Failed". (2) Final at once: the readiness outcomes
  `no_12v` and `absent` (`CHESTNUT_FINAL_REASONS`) mark the retry as used (`ChestnutRetry`, written blocking) and take
  upstream's verdict with no cycle: a 1 s cycle does not switch the outlet or make a missing bridge enumerate
  (2026-10-18: no 12 V at the dock for a whole drive; the dock's 12 V plug was likely not fully seated, and after a
  reseat the next drive, 5 min later, had power from its first sample). So do `fw_mismatch` / `flashing` from the wait
  (`CHESTNUT_FW_REASONS`, firmware gate below), whatever `CHESTNUT_FINAL_REASONS` says. `timeout`, `unreadable`, load
  exceptions, timeouts and wedges take upstream's retry. (3) After a first failure modeld loads
  nothing more: an in-process load behind a loader still inside a C call can hang (route 7f), so it waits up to 5 s
  for the cycle's SIGINT (`VBSM_EXIT`) and otherwise exits for a SoC respawn. (4) On the final attempt a loader that
  is still alive after its budget ends the process (`ChestnutActive=False` written blocking first) and the respawn
  runs SoC, except while a cold cache is still building (`VBSM_GPU_COLDCACHE`). Per ignition: at most two eGPU
  attempts and one cycle, both at the drive start, never while engaged (`ChestnutLoading` is a NO_ENTRY). Replaces
  the boot veto file, the hang-strike counter, the load-fail ladder and the GPU-active death veto (retired below).
- **No device probe at import** (`VBSM_GPU_READY`, `modeld_v2/modeld.py`): upstream's `compile_modeld.py` declares
  `device: str = Device.DEFAULT` as a default argument, so importing modeld_v2 makes tinygrad probe every backend
  (AMD first) before `main()` runs. On a cold boot that probe opens the AMD device before the PCIe link is up, fails
  silently, and leaks the lock fd and the libusb claim -- the 2026-09-13 "lock error 7 ms in, no other holder"
  case. modeld now pins `DEV=QCOM` before tinygrad is imported (verified: importing the module opens nothing);
  every tensor in modeld_v2 names its device explicitly and the eGPU is opened deliberately in `load_big()`. Still
  required at b50b82f9 (`model_adapters.py` imports `compile_modeld`); since 2026-10-19 the pin sits right after the
  module docstring, above upstream's own imports, and the port's verify step checks its position.
- **Link readiness before the first open** (`VBSM_GPU_READY`, `modeld_v2/modeld.py`): the enclosure's custom
  firmware boots with PCIe off, and tinygrad's AMD open powers it on and checks the link once, with no wait. At a cold
  boot the link is often still training, so the first open fails -- and, verified on-device 2026-09-14, that failure
  leaks both tinygrad's lock fd and the libusb claim inside the process, so no later open in that process can ever
  succeed ("Failed to acquire lock", then "Resource busy"). modeld therefore reads the bridge's supply voltage and
  LTSSM straight over usbdevfs before the first open (the same read-only control transfers as `chestnut_power.py
  status`; own fd, no tinygrad, no lock). With 12 V present and the link down it writes the PCIe power bit (0xF3=1)
  once, right after the `ChestnutLoading` param so link training overlaps the vision-stream wait, re-sends it only if
  the LTSSM is still in Detect 10 s later, and polls at 2 Hz until L0 (0x78) or the budget ends: 20 s, and in any
  case 22 s after `main()` entry, so probe + the 60 s loader budget + the SoC load (or the wait for the cycle) stay
  12 s inside ui_watchdog's 100 s deadline (the watchdog matches modeld's exact proctitle, which only appears at
  `setproctitle()` inside `main()`, and polls every 2 s, so its clock starts at or after modeld's; e8b7e968 counted
  from process start and let a slow import shrink the link budget). The reads and the power bit go only to a bridge
  whose product string is the pinned firmware (`CHESTNUT_USB_PRODUCT`): an old, ROM-bootloader or mid-flash bridge
  reads as enumerated but unreadable and gets no traffic from modeld (upstream's cable predicate now lets this code
  run before the product matches).
  Outcomes: `ready` opens; `no_12v` (under 5 V for 10 s, or still under 5 V when the budget ends: a dead outlet,
  `VBSM_GPU_12V_GRACE`) and `absent` (20 reads, ~10 s, with no bridge in sysfs: never less patient than upstream's own
  10 s `wait_for_chestnut()`) are final at once; `timeout` (12 V present, link never trained) and `unreadable`
  (enumerated but silent) are failed loads that take upstream's retry; `fw_mismatch` / `flashing` (an enumerated,
  unreadable bridge on old, ROM or mid-flash firmware, checked every 4th unreadable read and at the deadline) are final
  at once; `probe_error` opens anyway. A skip never starts the loader. A vision-stream wait over 20 s moves the plain
  deadline past main()+22 (`max(.., t0+2)`); that excess goes to ui_watchdog with the COLDWAIT seconds.
  Events: `chestnut preflight` (state at
  start: DEV, opened devices, lock fds, supply, LTSSM, whether the power bit was written and the LTSSM right after),
  `chestnut link ready` / `chestnut link not ready` (reason, reads, F3 writes in the wait and in total, wait, age since
  main and since exec, supply, LTSSM), `eGPU load skipped; link not ready`, and `open_ms` on
  `chestnut ppt limit` (duration of the first open). The first revision (e0c253de: 1 Hz `flash.link_up()` polling)
  wrote 0xF3=1 on every probe, live link included, and its 20 s could overrun the watchdog deadline; it was replaced
  before its first drive. First drive on e8b7e968 (2026-09-14 20:08 and 20:23 PT): both starts warm (hardwared's
  rails-on had trained the link), preflight LTSSM 0x78, ready on the first read, open 4.4 s / 1.9 s, big model from the
  first process in 26 s / 23 s, 100 % big frames, 0 % drops; the cold-boot link-down path is still unobserved.
- **12 V grace** (`VBSM_GPU_12V_GRACE`, `modeld_v2/modeld.py`, 2026-10-15): `no_12v` used to fire after three
  reads under 5 V (1.0 s). That morning the device had booted about a minute before ignition, so modeld started with
  the accessory outlet still off (5-8 mV). It skipped the eGPU after 1.0 s, the late probe saw 13.9 V and L0 2.1 s
  later, and the driver got "Big Model Failed / Restart the car to retry" for 20 s, then a kick reload (CTV3M in
  27 s). The supply now has to stay under 5 V for `CHESTNUT_NO_12V_GRACE_S` = 10 s (three times the observed lag)
  before the wait gives up, inside the unchanged 20/22 s budget, so ui_watchdog's deadline arithmetic holds. A wait
  that runs out of budget while still unpowered reports `no_12v`, which is final at once since 2026-10-19 (no
  onroad cycle). Cost: a truly dead outlet starts the SoC model about 9 s later. Field record before the change: 72 `ready`, 2 `no_12v` (one dead outlet, one
  this late outlet). Scenario tests:
  `pending/port-2026-10-19/notes/integrated/tests/test_wait_ready.py` (old vs new, 9 cases; the `pending/no12v-grace` original lacks the
  firmware-check stub the in-wait gate needs) and `run_wait_dm.py` (13).
- **Cold-boot link wait** (`VBSM_GPU_COLDWAIT`, `modeld_v2/modeld.py` + `ui_watchdog.py`, 2026-09-28): the
  silent-but-enumerated bridge shows up on about half the first starts after the device was off (13 of <=26 from 09-17
  to 09-28) and has never been seen on a later start. Anchored to kernel boot, the silence ended 43.6-48.7 s after boot
  (sd 1.6 s; 09-27: 47.3-50.7), while the plain budget runs from main(), so a modeld that reaches main() early has
  less of it left. 09-27 had the earliest main() of all silent starts (27.1 s after boot vs 29-38): the budget ran
  out at 47.1 s, the eGPU was skipped, the late probe kicked, the SIGKILL restart took 14 s in the boot CPU storm,
  and a reboot (DoReboot, most likely the driver) came 4.4 s into the reload. Now, once a read has come back
  `unreadable`, the wait may run until 70 s after kernel boot (`CHESTNUT_COLD_BOOT_AGE_S`) but never past 40 s after
  main() (`CHESTNUT_COLD_MAX_S`). In practice the extension is max(0, min(20, 50 - boot age at the wait's start)) s:
  a wait that starts 50 s or more into the boot (the observed kick restart started at ~67 s) keeps the plain budget,
  and so do `no_12v`, `absent` and a never-silent `timeout`. Margin over every observed silence: 16-26 s (the old
  budget had 2-4 s on 7 of 12). 09-27 would have read `ready` when the bridge woke and had a big frame ~24 s later
  (~71-75 s after boot) instead of a small model, a kick, a ~40 s no-model reload and the reboot. Costs: an early
  engage on the small model is no longer offered during the wait ("Big Model Loading" blocks engagement, the HUD
  shows LOADING); a bridge that answers and then trains keeps the extension (ends `timeout` at main+40 s if it never
  reaches L0); and in the never-seen case of a bridge that stays silent the small model comes up to ~20 s later
  (12-20 s at the observed main() times). The seconds used past the plain deadline (`extended_s`, rounded up; the
  events also carry `wait_boot_age_s`) go to ui_watchdog through `/dev/shm/vbsm_gpu_preflight_ext` ("<pid>
  <seconds>", cleared by every modeld start, also written if the probe errors after extending); its wedge deadline
  (100 s since 2026-10-19, 90 s before) moves by that much for that pid only (non-finite values ignored, clamped to
  155 s since `VBSM_GPU_COLDCACHE` adds 60 s per cold model cache to the same file), so every other load keeps
  100 s. Ship both
  files together: without the watchdog side an extended start whose load is unusually slow would hit the wedge
  path. Re-derive both constants before anything moves main() earlier (e.g. pinning modeld's imports to core 7).
  Tests: `pending/port-2026-10-19/notes/integrated/tests/test_wait_ready.py` (wait) and `test_wedge_int.py`
  (watchdog deadline).
- **Cold model cache** (`VBSM_GPU_COLDCACHE`, `modeld_v2/modeld.py` + `ui_watchdog.py`, 2026-10-19): upstream's
  plain-artifact loader rebuilds every graph on the first load of a model file on a tinygrad build and writes
  `<pkl>.cache` only at the end. Nothing pre-builds it for the eGPU model (`modeld_warmup` does the SoC model only),
  and its key holds a hash of the tinygrad source, so it is cold on the first drive after every port that bumps
  tinygrad and after every model change. Without help, a cold load over 60 s would fail both attempts and start from
  zero at every ignition: no big model ever. The loader builds upstream's own `_cache_key` the way `load_model()`
  does, frame-fusion digest included (`_extended` with the same `fused_with_warp`; upstream's `cache_is_fresh()` uses
  an empty digest and never matches a fused model), so whatever makes `load_model()` rebuild reads cold: a tinygrad
  or model change, and also a new warp pkl or fused layout on an unchanged tinygrad (the port's first draft copied the
  stored digest into the key, read that case as warm and would have cut the build twice per ignition, forever). The
  digest covers headers and kernels, not the weights: 0.05-0.5 s for 1000-6000 kernels on a Pi 5 bench with
  upstream's real helpers (`check_s` in the event; no mtime shortcut: an update staged before a later cache write would
  read warm). When cold it adds 60 s to the loader budget (`CHESTNUT_COLD_CACHE_EXTRA_S`) and the same to
  ui_watchdog's deadline through `/dev/shm/vbsm_gpu_preflight_ext`. If the final attempt of an ignition still finds
  the build running, modeld leaves it running in the background (upstream's own final behaviour) and loads the SoC
  model, so the cache is written for the next ignition (the freeze backstop below covers a build that then holds the
  GIL). The SoC model loads after the big one with `ChestnutLoading` still True, and `modeld_warmup` never pre-builds a
  native model, so right before that load (on every eGPU path, final-at-once skips included) modeld checks the SoC
  cache the same way and, when cold, adds 60 s more to the watchdog's deadline (not to the join). Events: `chestnut
  cache` and `chestnut small cache` (cold, extended_s, check_s). Measure the first loads (`models loaded in`) and
  resize the constant if a cold load comes close to 120 s or a cold SoC build passes ~66 s (it keeps the 12 s margin
  up to there; the watchdog acts at ~78 s).
- **Firmware gate** (`VBSM_GPU_FW_GATE`, `modeld_v2/modeld.py`, 2026-10-19): upstream pinned firmware 8516b768 and
  hardwared flashes the enclosure before going onroad (onroad waits up to 30 s; the flash budget is 600 s). With 12 V
  present `flash.py` reports that the new firmware "activates on its next power cycle", so the old product string
  can last several ignitions, until a park shutdown. If at `main()` a chestnut is on the bus with the ROM-bootloader
  ids or a product string other than the pinned one (re-read 1 s later; an empty string mid-enumeration does not
  count), or `flash.py` is running, modeld runs the SoC model for that session without touching the bridge,
  upstream's retry or the alerts: `ChestnutLoading` stays False and `ChestnutActive` absent, so there is no false
  "Restart the car to retry" (a restart does not activate firmware) and no onroad cycle (whose 1 s offroad window
  could start a flash retry that holds the car offroad for 30 s). Event: `eGPU skipped: chestnut firmware not
  current` (reason `fw_mismatch` / `flashing`). The HUD shows the orange icon for 6 s. A bridge that enumerates only
  after `main()` (re-enumerating, slow at power-up) reads unreadable in the wait, so the wait runs the same check on
  every 4th unreadable read and once more at its deadline; a mismatch there ends the wait `fw_mismatch` / `flashing`,
  final at once (upstream's "Big Model Failed", no cycle, so no flash can start in a cycle's offroad window). Only the
  gate at `main()` is silent.

- **Truthful big-model alerts** (`VBSM_GPU_ALERTS`, `selfdrived.py`): "Big Model Ready" only fires when
  `ChestnutActive` is True at the end of loading (it also fired 2 s after a real failure), and "active
  but no modelV2" is ignored inside the 5 s settling window (modeld marks the big model active ~2 s
  before its first published frame, which raised a false "Big Model Failed" banner). `BIG_MODEL_WARMUP_SEC`
  replaces the local `warmup_sec`. Under upstream's retry (2026-10-19) `ChestnutActive` is tri-state when loading
  ends: True = big model, False = final failure, absent = first failure with a cycle pending, or a firmware skip; only
  True announces Ready (upstream would announce "Big Model Ready" on every first-failure edge). Latch: upstream now
  re-arms the 5 s settling window on every bigModelFailed, so a failure already raised stays raised
  (`self.big_model_failed or not big_model_settling`); without it the mask cleared the failure and re-raised it every
  5 s while a frozen or not-yet-respawned modeld left `ChestnutActive` True.
- **Load masks only while disengaged** (`VBSM_SETTLE_GATE`, `selfdrived.py`, 2026-09-28): upstream's
  `big_model_settling` (a load, and the 5 s after it) masks commIssue, posenetInvalid,
  locationdTemporaryError and paramsdTemporaryError, and the fork added "active but no modelV2" to it. It
  applied while ENGAGED too, where a load can only mean modeld died under the driver. (a) If modeld dies
  inside the 5 s after a load (the driver can engage the moment "Big Model Ready" shows), the soft
  disable waits for the window to end. (b) When the replacement modeld reaches main() it sets
  ChestnutLoading, the SOFT_DISABLE event vanishes and `softDisabling` goes back to enabled (state.py).
  Together: a death ~1.7 s after the load (the field kick timing) with a 3-5 s launch-to-main left the
  car engaged on a dead modelV2 for 27.6-29.5 s in the time-stepped sim, steering on the last
  desiredCurvature. (b) alone needs a launch-to-main under ~3 s, never seen (restarts measured >=4.3 s).
  Now `_big_model_settling()` is False while engaged (openpilot, or MADS `active`; a paused MADS does not
  steer and keeps the masks), and while the masks apply anything they hide adds `bigModelLoading`
  (NO_ENTRY), since the masks lift on engagement and a soft disable would follow at once. Effect on
  normal starts: every start now refuses an engage for ~1.0-1.7 s after "Big Model Ready" (median 1.3 s
  over 74 field loads, until deviceMotion and all_checks come up), then clears; only 1 of 96 archived
  loads had an engage that early. `bigModelLoading` therefore spans ~1 s past the load end, so tools that
  time loads from that event (tools/cablecmp.py, vbsm-train/brownout_ds/_tools/build_manifest.py) read
  ~1 s long. Benches: `pending/prJ/bench_settle_gate.py` (real source of both versions, 35-case truth
  table) and `bench_settle_gate_dyn.py` (real update_events segments + real state.py and mads/state.py
  at 100 Hz, 13 scenarios: old 27.6-29.5 s engaged on a dead model, new 3.0 s then disengaged).
  Upstream-worthy (the mask is sunnypilot's, not the fork's). Still required at b50b82f9: upstream's own
  `big_model_settling` has no engagement check and now also masks `modeldLagging`; the fork's two gated call sites
  feed both (the port's verify step counts them), and the NO_ENTRY guard covers `modeldLagging` too: inside the window
  a `frameDropPerc` over 1 % refuses the engage (an engage there met an immediate "Driving Model Lagging" soft
  disable once the mask lifted; the 10 s drop filter outlasts the 5 s window).
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
  2026-09-22: counts invalid MESSAGES, not invalid cycles (`validity_step()`). The per-cycle rule re-counted
  whatever message was latest on every 20 Hz cycle, and `extrinsicsCalibration` is 4 Hz and copies
  cameraOdometry's validity, so the same one-frame blip came back as ONE invalid calibration message that
  stayed latest for 250 ms = 5 cycles and still raised the alert (2026-09-16, one occurrence, on this rule). Now
  an invalid message counts once when it arrives (remembered across `sm.update()` timeouts until the next
  cameraOdometry cycle), the counter holds while it stays latest, and decays only when everything is valid. A
  service whose latest message is invalid and that has stopped publishing (not alive, 10x its period) counts
  every cycle, so dying on an invalid message still faults (~2.5 s for calibration). Cost: a sustained fault
  on the 4 Hz service is reported after 3 invalid messages (~0.5 s) instead of ~0.1 s; 20/100 Hz services and
  startup are unchanged. Archive replay (every rlog on the Pi): the only episode that differs is that cascade.
- **Wedge backstop** (`VBSM_GPU_WEDGE`, `ui_watchdog.py`, 2026-10-19): a load can wedge inside a C call that holds the
  GIL (route 81): modeld ignores SIGINT and its own timeout never fires, so neither upstream's retry nor the SoC model
  ever happens. If modeld (exact proctitle `openpilot.selfdrive.modeld.modeld_tinygrad`; upstream's `modeld_warmup`
  also contains `.modeld`) still has `ChestnutLoading` set 100 s (plus the `VBSM_GPU_COLDWAIT`/`VBSM_GPU_COLDCACHE`
  seconds it reported) after the watchdog first saw it, the first time in an ignition the watchdog spends upstream's
  retry (`ChestnutRetry` + `OnroadCycleRequested`) and does not kill: the cycle's own stop (SIGINT, then the manager's
  5 s join and SIGKILL in `start()`) replaces the process with no `VBSM_RESTART` respawn racing the cycle. If the same
  process still loads 15 s later (`GPU_CYCLE_GRACE_S`), or the retry is spent, or openpilot or MADS is engaged, it
  writes `ChestnutActive=False` and `ChestnutLoading=False` and SIGKILLs; the respawn runs SoC (session gate). No
  presence gate: a bridge that vanishes mid-load must not hide a wedge. "Retry spent by this watchdog" is kept per
  process: a quick car restart whose offroad blip fell between two 2 s ticks still gets the new ignition's retry.
  **Freeze**: a modeld that has published modelV2 and then publishes none for 15 s (`GPU_FREEZE_S`) while its pid
  lives is SIGKILLed (no param writes; the respawn runs SoC). It covers the cold-cache heal's background build
  holding the GIL under the SoC model and a big-model `run()` that never returns (both left no model until car-off).
  Armed only by that pid's own output and never on the tick that first sees it, so loads and camerad waits before the
  first frame never count; a camerad stall over 15 s now also costs one modeld respawn. Log: `published no modelV2`.
- **Onroad cycle guard** (`VBSM_CYCLE_GUARD`, `hardwared.py`, 2026-10-19): hardwared consumes `OnroadCycleRequested`
  as upstream does, but refuses (does not defer) the cycle while openpilot is engaged (`selfdriveState.enabled`) or
  MADS steers (`selfdriveStateSP.mads.active`, subscribed for it; MADS-only lateral is not `enabled`): a cycle stops
  selfdrived, card and controlsd with no alert. Every requester today is disengaged by construction (modeld and ui_watchdog only
  while `ChestnutLoading`, which is a NO_ENTRY; upstream's UI behind `_engaged_confirmation_click`), so this is a
  backstop for a future requester. A refused modeld request ends in its 5 s exit and a SoC respawn. Event: `onroad
  cycle refused while engaged`.
- **Retired 2026-10-19** (upstream's retry adopted): the automatic kick (`VBSM_GPU_KICK`) and its cooldown fix,
  `VBSM_GPU_KICK_ARMED`, the rail-gated mid-drive retry (`VBSM_GPU_RETRY`) and the driver request
  (`VBSM_GPU_KICK_REQUEST`, `/dev/shm/vbsm_gpu_kick_request`): competing retry paths, and a failed kick reload with
  `ChestnutRetry` unused would have requested an onroad cycle mid-drive. The late link probe (`VBSM_GPU_LATE`,
  `/dev/shm/vbsm_gpu_link_wait`/`_ready`): it only armed the kick. The lock-contention retry loop and its forensics
  (`VBSM_GPU_LOCK_RETRY`): it never fired after the DEV pin, no external lock holder exists, upstream's
  `_patch_system_flock_acquire` now makes in-process re-opens reentrant, and a fresh process is upstream's retry;
  `chestnut_lock_fds()` stays as the preflight's canary. The load-fail ladder (`/dev/shm/vbsm_gpu_load_fails`), the
  veto file (`/dev/shm/vbsm_usbgpu_veto`), the hang-strike counter (`/dev/shm/vbsm_gpu_hangs`), the GPU-active death
  veto and the drive-end veto clear: `ChestnutRetry` scopes the retry per ignition, upstream swaps to the resident
  SoC model in-process on a mid-run failure, and the session gate keeps respawns off the eGPU. The HUD tap and the
  2 s hold-to-reboot on the chestnut icon (`VBSM_GPU_HUD_TAP`): the tap's target is gone, and the hold rebooted in
  any state, engaged included; `VBSM_HOLD_RESTART` (3 s, upstream's prompt, "Disengage to reboot" while engaged)
  covers it. The orange icon no longer stays on screen while disengaged. The manual force-SoC lever (`touch
  /dev/shm/vbsm_usbgpu_veto`) is gone: clear the chestnut model slot in the model selector instead.
- **HUD status**: upstream's chestnut icon (pulsing = loading, green = big model live, orange =
  fallback) is driven by upstream's state machine (`ui_state._update_chestnut_state`: LOADING wins, FAILED is sticky
  for the session, a tinygrad-runner bundle counts as compiled) — our `VBSM_GPU_HUD` gate patch became obsolete and was
  retired (see Retired below). Since
  2026-10-19 every state, orange included, lingers 6 s after a change and then yields the slot to the dmoji
  (`VBSM_HUD`); the failure itself is carried by upstream's 20 s alert and the offroad alert.
- **Parked power-off** (`VBSM_GPU_IDLE`, `hardwared.py` + `chestnut_power.py`): the enclosure holds
  12 V after some parks and idles at 25–40 W straight off the car battery. After 120 s of offroad the
  GPU rails are cut via the firmware's F3 switch (hardware-validated both directions: 2 A → 1 mA,
  restore retrains the link first try); restored on the offroad→onroad edge. Opt out:
  `/data/vbsm_no_gpu_idle_off`. The privileged CLI runs under `python -B` — a root interpreter
  writing bytecode into the tree silently breaks the updater's git clean. Since 2026-10-19 hardwared sends no
  rail command while the firmware is out of date (unless the flasher gave up after its 3 attempts; the rails then
  resume on the old firmware, where F3 was validated) or while `flash.py` runs, and `chestnut_power.py` refuses
  (exit 3) any bridge that is not on a custom runtime firmware, the ROM bootloader included, re-checked at the moment
  of the command. On 8516b768 the F3=0 rail cut is accepted: 40 of 40 `chestnut gpu rails cmd=off` succeeded from
  2026-10-20 to 10-23. Still unmeasured: the ~1 mA in `chestnut_power.py status` after a park of 120 s or more with
  12 V held.

### 5. Driver HUD — `VBSM_HUD`
- Experimental mode (retired 2026-10-19): upstream b50b82f9 made the e2e plan permanent and deleted the
  `ExperimentalMode`, `ExperimentalModeConfirmed` and `DynamicExperimentalControl` params, the distance-button hold
  toggle (`cruise_helpers.py`) and the mode icons. The fork's `VBSM_EXP_TOGGLE` / `VBSM_LKAS_REPURPOSED` and the
  driving-mode badge are gone, and `mads.py` left the managed set. The LKAS button is stock MADS on/off. A
  distance-button press always cycles following distance.
- `hud_renderer.py` set speed (2026-09-08): shown while engaged and whenever cruise is *resumable* — ACC main on with a retained set speed (`carState.cruiseState.available`) — hidden otherwise after the stock 2.5 s fade, so the driver sees what RES will bring back.
- **Persistent set speed** (`hud_renderer.py`): the cruise set-speed no longer fades 2.5 s after a
  change; it stays up whenever engaged. Known cost since 2026-10-19: upstream's `LongIndicator` (lead car, green when
  the e2e plan leads, distance bars) yields to top icons, so with the set speed always up it never shows while
  engaged. `set_should_draw(True)` in augmented_road_view would bring it back (its slot is free).
- **Gap-profile chip** (`hud_renderer.py` + `augmented_road_view.py` + `selfdrived.py`): top-right
  blue-bar indicator — one bar = aggressive, two = standard, three = relaxed — reading the
  personality live from selfdrived's own state. Tap to cycle; selfdrived picks it up through upstream's
  10 Hz params thread, and the chip's flash is the feedback (the personality alert fires for the wheel
  button; the fork's own 1 Hz re-read rarely gets there first). A chip tap is not also the road view's
  tap-to-home (until 2026-10-19 it hid the road view for ~6 s), and a tap on a dialog pushed over the
  road view never reaches the chip.
- **Dimmed wheel** (`hud_renderer.py`): the bottom-left steering wheel stays visible while disengaged, dimmed to
  45 % instead of sliding away (it used to double as the driving-mode badge; the flask is gone with experimental
  mode). A blind spot still takes the slot. The steer-required critical wheel is untouched.
- **Driver-monitoring face relocated** (`augmented_road_view.py` + `hud_renderer.py`): the dmoji
  moved from top-left (where the persistent set-speed kept it hidden whenever engaged) to the
  bottom-right eGPU slot; it appears once the eGPU status icon fades (with nothing to show the fade resets, so
  a drive that ended with the icon up cannot hide the dmoji for the whole next drive). The eGPU icon also lingers
  6 s after a state change (was 2.5 s) so its color is actually readable.
- **Active model names** (`layouts/home.py`): the home screen names both driving-model slots on the
  version line beside the commit hash — the SoC model, plus the big GPU model when the chestnut is
  attached (e.g. `wmiv12 / bmv4`). Short catalog names (`internalName`) only: `displayName` runs to
  56 characters and will not fit the 536 px line. Read from the two `ModelManager_ActiveBundle*`
  params at 1 Hz, never from `modelManagerSP` (which republishes a tick late and would flash the
  wrong model on a chestnut transition).
- **Parked battery voltage** (`layouts/home.py`): "12.2V" readout in the home-screen footer,
  right of the chestnut icon — pandad's peripheralState on a widget-local 2 Hz SubMaster (the
  same source the power monitor uses). Renders only while the screen is already awake; adds no
  screen-on time and no measurable power draw.

### 6. Boot & log hygiene — `VBSM_QUIET`, `VBSM_LOG_LEVEL`
- **Chunk-manifest storm fix** (`models/fetcher.py`, RETIRED 2026-10-19): two catalog entries shared a fileName
  with different chunk counts, flipping the same manifest twice per second forever (93 % of log volume, ~2 h
  retention); manifests were written only when the chunks existed locally. Reported upstream
  (sunnypilot/sunnypilot#1975). Upstream b50b82f9 removed model chunking and the manifest writer with it.
- **Shutdown debounce** (`hardwared.py`): the offroad shutdown decision must hold for 2 consecutive
  iterations before `DoShutdown` fires — kills the race where a shutdown latched in the same
  sampling window as an ignition rise and turned a departure into a double boot.
- **Offline-boot catalog storm** (`models/fetcher.py`, `VBSM_QUIET`, 2026-09-20): with no network the
  models manager retried every source once a second and logged three warnings each time (~50 lines per
  boot). After any failed fetch (transport, 404/5xx — `HTTPError` is a `RequestException` and was always
  caught, never raised — or a captive-portal page) a source is now held for `FETCH_BACKOFF_S = 15` and
  served from the cache quietly; a successful fetch clears the hold. 15 s keeps a user's "Refresh Model
  List" (20 s spinner) inside one hold; the hold is per source and per process and cannot withhold a
  catalog for longer than that.
- **Oversized sunnylink log** (`athenad.py`, `VBSM_QUIET`, 2026-09-20): a swaglog minute over 32 KB cannot
  be sent to sunnylink in one request; the handler waited ~100 s for a response that never came and
  retried the same file every hour forever. `add_log_to_queue()` now returns False for it and the handler
  marks it done (`LOG_ATTR_VALUE_MAX_UNIX_TIME`) once.
- **Night notice retired** (`augmented_road_view.py`, 2026-09-20): `_refresh_bsm_night` was never called
  (dead since the detector was shelved); it and `_draw_bsm_night_notice` are gone. `BSM_STATE_PATH` stays
  as the documented name of the daemon's state file.
- **`VBSM_LOG_LEVEL`** (2026-09-20): `SwagLogger.event()` (`common/logging_extra.py`) logs at ERROR
  whenever an `error` kwarg is *present*, whatever its value. `hardwared.py` passed `error=not ok` on
  `chestnut gpu rails` and `error=<bool>` on `chestnut flash done`; `modeld.py` passed `error=False` on
  `chestnut link ready`, `chestnut preflight`, `chestnut ppt limit` and `eGPU load succeeded after lock
  retry` — six success events logged as errors (an AST census of all 21 `.event(` sites carrying
  `error=` in the managed files found exactly these six; the other 15 are genuine failures). The kwarg is now passed only on failure; the failure siblings keep `error=True`.

### 7. Driver-monitoring model pin — `VBSM_DM_PKL_PIN` (RETIRED 2026-09-21)
- **Retired**: sunnypilot staging `81d7957c` (2026-09-18) reverted the driver-monitoring pickle to
  the exact pre-`72da24ad` blob (`5c969f86`) and recompiled both warp pickles against the shipped
  tinygrad (`CallInfo` REDUCE arity 5, two constructions each, the same shape as the pinned
  `06740e07`/`193f9da9`); `tinygrad_repo` (tree `07174c66`) and `dmonitoringmodeld.py` (`4010725b`)
  are unchanged. The port's drift guard stopped on the three pickle paths exactly as designed, the
  four paths left the manifest (27 -> 23 files) and the port now carries upstream's pickles. The
  original entry is kept below as the record of what the pin was and how to judge the next one.
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

### 8. Camera view reader-slot pool — `VBSM_VIPC_POOL`
- **What** (`selfdrive/ui/mici/onroad/cameraview.py`): the mici camera view keeps one
  `VisionIpcClient` per stream and reuses it on every road / wide / cabin switch, instead of
  building a new client per switch. Idle pooled clients that are neither on screen nor pending are
  dropped at each offroad transition and when the on-screen client reconnects to a new camerad
  session.
- **Why**: every `VisionIpcClient` takes a msgq reader slot when it is constructed, and msgq never
  gives slots back (`msgq_close_queue` only munmaps). With openpilot longitudinal the view switches to
  the wide camera below 5 m/s (gated on experimental mode before b50b82f9), so a long stop-and-go drive fills the wide queue's `NUM_READERS = 25`
  slots at UI wide switch #23/#24 (26 minus modeld, encoderd and the UI's start stream). The next
  registration evicts every reader. modeld reads its wide client only after the road frame arrives,
  so on a frame where wide N was published before road N (about 41 % of frames) its re-registered
  reader skips wide N and waits about 48 ms for wide N+1: `frames out of sync!`, road N+1 dropped,
  `cameraOdometry.valid = False`, a short invalid cascade (calibration, paramsd, lagd, torqued,
  driver monitoring, planner) and selfdrived's `commIssue` — "TAKE CONTROL IMMEDIATELY", recovering
  after about 0.5 s. Seen on 2026-09-15 and 2026-09-25 (each at the predicted switch number); a scan
  of 53.5 steady hours of logs found the same mechanism behind all 7 `frames out of sync` events
  and no other cause. Upstream report: commaai/openpilot#38505 (msgq `NUM_READERS` 15 -> 25 only
  delayed it).
- **Effect**: the UI holds at most one slot per camera queue per camerad session (plus one per idle
  stream at each started glitch that does not restart camerad), so the wide queue stays near
  3-6 readers and the eviction cannot happen. A lapped idle client is reset by `msgq_reset_reader`
  (no new slot), and `connect()` reuses the constructor's socket, so reconnects do not leak either.
  A switch back to a pooled stream shows its first new frame within one frame period.
- **Not covered**: `vision_bsm.py` still builds a new cabin client after 50 empty receives
  (reconnect path). That is rare onroad and cannot cost the driving model a frame; at worst a
  cabin-queue overflow makes dmonitoringmodeld drop one cabin frame (no validity effect observed).
  UI restarts (`VBSM_RESTART`) build a new pool, 1-3 slots each, bounded by the restart limit.
- **Retire when**: msgq releases reader slots on close, or upstream's camera view stops building a
  client per switch. The port's drift guard stops on any upstream change to `cameraview.py`.

### 9. Reverse view — `VBSM_REVERSE_CAM`
- **What** (`selfdrive/ui/mici/onroad/augmented_road_view.py`): while the car is in reverse, the
  onroad view switches to the cabin camera and shows the whole frame, fitted and mirrored like a
  rear-view mirror, with the driving path hidden. Alerts and the HUD still draw. Reverse takes
  priority over the blind spot window view and the low-speed wide camera. Leaving reverse
  returns to the road camera through the same switch path, so a switch still in flight is cancelled
  rather than landing late (`_cancel_pending_switch`).
- **Switch**: `reverse_view` in `/data/vision_bsm.json`, default on, set from the toggles page
  ("driver camera in reverse"). A missing, unreadable or malformed file (valid JSON that is not an object) reads
  as on. With `NavigationEnabled` on (default off) upstream's navigation panel narrows the mirror, as it does the
  blind spot window view (not changed).
- **Reverse alert timeout** (2026-10-15; 5 s -> 2 s on 2026-10-21): the stock `reverseGear/permanent` alert is full
  size and stays up for the whole time in reverse, so it covered the cabin view. `ReverseAlertRenderer` draws it for
  `REVERSE_ALERT_S` = 2 s, then lets it fade out. Only the drawing changes: selfdrived still raises `reverseGear`
  (engagement stays blocked), the `reverseGear/noEntry` alert shown on an engage attempt is untouched, and every other
  alert draws as before. The alert also kept the screen awake (sunnypilot's onroad brightness timer resets while an
  alert shows), so the view resets that timer itself while in reverse. With `reverse_view` off the alert behaves as
  stock. Test: `pending/reverse-alert/test_reverse_alert.py`; `pending/reverse-alert-2s/test_reverse_alert_2s.py` runs
  the same cases with the timeout read from the file.
- **Trust rule**: as for the window view, the gear only counts while `carState` is alive and valid,
  so a frozen `carState` cannot hold the cabin view on screen.
- **Why**: the comma four has no rear camera; the cabin camera is the only one facing backward.
- **Tested**: `pending/reverse-cam/test_reverse_view.py` (out of tree) drives the extracted methods
  through 13 cases: switch, priority, stale or invalid `carState`, toggle off, shift out before and
  after the cabin stream connects, no cabin stream, and the frame matrix;
  `pending/port-2026-10-19/notes/integrated/tests/test_reverse_view_new.py` adds upstream's camera-view switch path (16 cases).

### 10. Hold to restart — `VBSM_HOLD_RESTART`
- **What** (`selfdrive/ui/mici/onroad/augmented_road_view.py`): onroad, hold a finger still anywhere on the road
  view for 3 s (`HOLD_RESTART_S`) and the restart prompt of Settings > Device > reboot opens, unchanged: "slide to
  reboot" when disengaged, "Disengage to reboot" when engaged, the same `DoReboot` action, and the same exits
  (swipe down from the top to go back; the nav stack also closes it when the car leaves standstill).
- **Rules**: a touch that stays within 20 px (`HOLD_RESTART_SLOP_PX`) for 0.5 s (`HOLD_RESTART_LOCK_S`) locks as a
  hold. From then on the main scroller ignores it (`AugmentedRoadView.is_swiping_left()` also answers True while a
  locked hold is down: the hook `main.py` already uses to stop scrolling during a bookmark swipe), it is no bookmark
  swipe and no tap, it may drift up to 60 px (`HOLD_RESTART_LOCKED_SLOP_PX`), and a ring on the other half of the view
  fills up to the prompt (it vanishes the moment the hold cancels). Before the lock existed (2026-10-20), a drift of more
  than 12 px to the right became the main scroller's drag and a drift of more than 20 px any way ended the hold, with
  nothing on screen: a finger resting for 3 s in a moving car drifts that far, so the gesture felt unreliable (the
  headless sim missed the prompt in 5 of 6 drift paths), and the release of a failed hold went to home. A hold let go
  early or moved past 60 px now does nothing on release and logs `VBSM_HOLD_RESTART: hold ended without the prompt
  after <s> s, moved <px> px: <why>` (released / moved / covered or scrolled / view not drawn). A hold on the chestnut
  icon opens the prompt like anywhere else (its own tap/hold gestures were retired 2026-10-19); the release of a hold
  that opened the prompt does nothing else (no personality cycle, no bookmark, no tap-to-home: the flags stay set until
  the next press, because the parent view handles a frame's events after this icon); the prompt opens once per touch.
  Only a touch the road view itself owns counts: one made over a pushed widget (the prompt itself) or one the main
  scroller took before the lock (a drag of more than 12 px to the right in the first 0.5 s) is never a hold, and a touch
  whose release was missed while the view was not drawn (car off, view scrolled away; a gap of more than 0.5 s) is
  dropped instead of opening the prompt at the next drive.
- **Why**: Settings > Device > reboot takes several taps and a scroll while the car is on.
- **Coupling**: the prompt is upstream's `_engaged_confirmation_click` in `layouts/settings/device/device_layout.py`
  (upstream turned `layouts/settings/device.py` into a package on 2026-10-18), imported when the gesture fires. A
  rename or move upstream disables the gesture (logged) and never stops the UI from starting; the port's verify step
  checks the function's signature there and the import line in `augmented_road_view.py`.
- **Tested**: `pending/port-2026-10-19/notes/integrated/tests/test_hold_restart_order.py` replays touches through the real handlers
  in the real event order (the child icon before the parent view; prompt pushed over the view, offroad gap), 10 cases, and
  `test_hold_restart_dm.py` (13 cases: upstream's new bookmark state, 150 px swipe threshold). The 2026-10-20 lock:
  `pending/hold-drift-2026-10-20/tests/` (the two replays updated for the lock, 10 + 18 cases, and `test_hold_drift.py`,
  14 cases on the real main scroller: six drift paths, early release, a big move, tap, swipes, bookmark). End to end:
  `pending/port-2026-10-19/notes/review-ui.d/` (`setup_sim.sh`, then `test_ui_sim.py` / `test_ui_findings.py`) boots
  the real mici UI headless. The old `pending/hold-restart/test_hold_restart.py` fails on upstream's new bookmark
  (`_active_alpha`).

## Managed files (23) and markers

| File | Mods | Markers |
|---|---|---|
| `VBSM.md`, `CHESTNUT.md`, `MODS.md` | documentation | — |
| `openpilot/sunnypilot/vision_bsm.py` | §1 (additive file) | — |
| `openpilot/sunnypilot/ui_watchdog.py` | §3, §4 wedge backstop (additive file) | `VBSM_WATCHDOG`, `VBSM_RESTART`, `VBSM_GPU_WEDGE`, `VBSM_GPU_COLDWAIT`, `VBSM_GPU_COLDCACHE` |
| `openpilot/sunnypilot/chestnut_power.py` | §4 (additive file) | `VBSM_GPU_IDLE` |
| `openpilot/system/manager/process.py` | §3 | `VBSM_RESTART` |
| `openpilot/system/manager/process_config.py` | §1, §2b, §3 process entries, backup_manager sigkill | `VBSM_EXIT`, `VBSM_PARKWATCH` |
| `openpilot/selfdrive/car/card.py` | §1 | — |
| `openpilot/selfdrive/selfdrived/selfdrived.py` | §1 chime, §4 big-model alerts + latch + settle gate, §5 personality re-read | `VBSM_CHIME_HOLD`, `VBSM_CONFIG`, `VBSM_GPU_ALERTS`, `VBSM_SETTLE_GATE`, `VBSM_HUD` |
| `openpilot/sunnypilot/models/fetcher.py` | §6 offline fetch backoff | `VBSM_QUIET` |
| `openpilot/selfdrive/ui/mici/layouts/settings/toggles.py` | §1, §9 settings | `BigConfigControl`, `VBSM_REVERSE_CAM` |
| `openpilot/selfdrive/ui/mici/onroad/augmented_road_view.py` | §1 preview, §5 profile-chip tap, §9 reverse view, §10 hold to restart | `BSM_STATE_PATH`, `VBSM_HUD`, `VBSM_REVERSE_CAM`, `VBSM_HOLD_RESTART` |
| `openpilot/selfdrive/ui/mici/onroad/hud_renderer.py` | §5 | `VBSM_HUD` |
| `openpilot/selfdrive/ui/mici/layouts/home.py` | §5 parked voltage | `VBSM_HUD` |
| `openpilot/selfdrive/ui/mici/onroad/cameraview.py` | §8 reader-slot pool | `VBSM_VIPC_POOL` |
| `openpilot/system/athena/athenad.py` | §2; §6 oversized-log skip | `VBSM_PRIVACY`, `VBSM_QUIET` |
| `openpilot/sunnypilot/modeld_v2/modeld.py` | §4 cap, upstream retry + session gate, readiness, 12 V grace, cold-boot wait, cold cache, firmware gate; §3 exit; §6 levels | `VBSM_GPU_PPT`, `VBSM_GPU_FALLBACK`, `VBSM_GPU_READY`, `VBSM_GPU_12V_GRACE`, `VBSM_GPU_COLDWAIT`, `VBSM_GPU_COLDCACHE`, `VBSM_GPU_FW_GATE`, `VBSM_EXIT`, `VBSM_LOG_LEVEL` |
| `openpilot/system/hardware/hardwared.py` | §2b park, §3 boot cause, §4 idle power + flash hold + cycle guard, §6 shutdown debounce + rails/flash log level | `VBSM_GPU_IDLE`, `VBSM_CYCLE_GUARD`, `VBSM_PARK`, `VBSM_LOG_LEVEL`, `VBSM_BOOTCAUSE` |
| `openpilot/selfdrive/pandad/pandad.py` | §2b parkwatch window + park events; §3 boot health | `VBSM_PARKWATCH`, `VBSM_BOOTCAUSE` |
| `openpilot/sunnypilot/parkwatchd.py` | §2b (additive file) | — |
| `openpilot/system/hardware/power_monitoring.py` | §2b parked energy budget | `VBSM_PARK` |
| `openpilot/selfdrive/locationd/locationd.py` | §4 bounded lockout, buffered message validity | `VBSM_LOC_CAP`, `VBSM_LOC_VALID` |

Retired: the eGPU retry family (§4, 2026-10-19, upstream's retry adopted): `VBSM_GPU_KICK`, `VBSM_GPU_KICK_ARMED`,
`VBSM_GPU_RETRY`, `VBSM_GPU_KICK_REQUEST`, `VBSM_GPU_LATE`, `VBSM_GPU_LOCK_RETRY`, `VBSM_GPU_HUD_TAP`, the load-fail
ladder, the veto and strike files, and hardwared's `Offroad_ChestnutUncompiled` filter (upstream removed the alert);
`VBSM_EXP_TOGGLE` / `VBSM_LKAS_REPURPOSED` and the driving-mode badge (§5, 2026-10-19, when upstream b50b82f9 made
experimental mode permanent; `mads.py` left the managed set); `VBSM_DM_PKL_PIN` (§7, the three driver-monitoring pickles and the `dmonitoringmodeld.py`
tripwire, dropped 2026-09-21 once upstream staging `81d7957c` shipped consistent pickles); `VBSM_COMPAT` (a modeld_v2 unpacking shim, superseded when upstream fixed the API
properly); `VBSM_GPU_HUD` (a ui_state compiled-gate patch for bundle installs, superseded by
upstream's chestnut state machine whose gate checks the bundle chunk manifest — `ui_state.py`
left the managed set with it).

## Tunables and switches

| Control | Effect |
|---|---|
| `/data/vision_bsm.json` | blind spot monitor config — `enabled` (gates the daemon), `camera_view`, `chime`, `chime_always`, `reverse_view` (§9), calibrated `zones`; all settable from the toggles page |
| `BlindSpot` (param) | on-screen blind spot icons — the one BSM setting that is a param, because upstream owns it |
| `/data/vbsm_gpu_ppt_w` | GPU power cap in watts (default 80, clamp 40–220; 0 = upstream's 100 W). Exported as `AM_POWER_LIMIT` at modeld import and written again after the bridge opens |
| `/data/vbsm_no_gpu_idle_off` | opt out of the parked GPU power-off |
| `/data/vbsm_park_budget_wh` | parked energy allowance in whole Wh (clamped 5–30; absent = stock 30), §2b |
| `/data/parkwatch.json` | its presence enables `parkwatchd` (process_config); ntfy server and topic, parked watch window `window_s` (default 1200 s), §2b |
| `/dev/shm/vbsm_sync_active` | touched by the Pi's footage sync per batch; fresh (<30 min) = budget/timer shutdown rules suspended, voltage rule kept |

## Operational notes

- Deploys: never kill the updater (the restart policy respawns it and the fresh cycle invalidates
  the consistency marker — a silent skip); instead HUP it and poll for the expected staged head
  *and* the marker in one on-device command, then reboot.
- A root python importing tree modules writes root-owned `__pycache__` that blocks all subsequent
  updates; clean with `find -user root` if updates fail on `git clean`.
- After a port that bumps the model selector (21 since 2026-10-19), re-select both model slots before driving. The
  first eGPU load after any tinygrad or model change builds the model cache (`chestnut cache cold=True`) and can take
  up to 120 s, and a cold SoC cache (`chestnut small cache cold=True`) adds its own build; the next ignition starts
  warm.
- The device RTC resets on offline boots: every boot's first minutes log into the same stale
  window, and crash files written pre-NTP carry stale names *and* mtimes.
- **Committing managed files through the Git Data API**: tree entries carry the file mode and the API does not
  inherit it; hardcoding `100644` stripped the executable bit from `modeld.py` on 2026-09-14 (manager exit
  code 126, two drives with no model). Take each path's mode from `git ls-tree <parent> -- <path>`
  (`pending/reapply_fixes_with_modes.py`) and verify exec bits plus a running modeld after every deploy.
