# Chestnut eGPU on the Accessory Outlet: A Field Debugging Log

This branch runs a comma four with a chestnut USB-GPU enclosure (AMD Radeon, 16 GB) powered from
the car's 12 V accessory outlet — the plug-in install, no rewiring. Getting big models stable in
that configuration surfaced a chain of distinct failure modes, several of them masking each other.
This document records each problem, the evidence that identified it, and the fix, so the next
person doesn't have to rediscover the chain. Companion doc: [VBSM.md](VBSM.md).

All fixes live in `openpilot/sunnypilot/modeld_v2/modeld.py` (markers `VBSM_GPU_FALLBACK`,
`VBSM_GPU_PPT`), `openpilot/sunnypilot/ui_watchdog.py` (`VBSM_GPU_KICK`), and
`openpilot/selfdrive/ui/ui_state.py` (`VBSM_GPU_HUD`), maintained across upstream rebases by the
content-port workflow described in VBSM.md.

## The short version

| # | Symptom | Actual cause | Fix |
|---|---------|--------------|-----|
| 1 | Model load dies mid-transfer; USB resets storm in dmesg even at idle | Bad USB-C cable (protocol-layer: `error -71`, "Device not responding to setup address") | Replace the cable. A reseat changes nothing — swap it |
| 2 | Load succeeds, then "Device hang detected" ~46 s into engaged driving | Supply-path resistance: ~0.33–0.45 Ω from outlet to enclosure. Fine at idle amps, collapses at inference current | Cap GPU package power via the SMU (80 W) |
| 3 | Every start burns a 60 s dead window; drive has no model at all | eGPU loader wedges inside a C call **holding the GIL** — the in-process timeout thread is frozen with it | Fall back by process replacement, never in-process |
| 4 | Load fails in 33 ms, GPU healthy | flock contention on tinygrad's `am_usb` device lock | One free retry; forensics hook logs the lock holder |
| 5 | Healthy GPU session killed 1 s after load completes | Our own watchdog: gated on `UsbGpuLoading==False`, which is also true on *success* | Gate on `UsbGpuActive` |
| 6 | GPU present but unused all drive (switched-power cold start) | modeld checks `usbgpu_present()` exactly once, at startup, and the enclosure boots in parallel | Watchdog "kick": restart modeld once, only at standstill + disengaged |
| 7 | No way to tell loading / ready / failed from the driver's seat | The stock eGPU HUD icon is gated on the stock big-model file, which bundle installs never have | Accept a cached bundle pkl as equivalent |
| 8 | One-off mid-transfer stall right after physically routing the cable | Connector/bend settling under first sustained 5 Gb/s load | The free retry absorbed it; recurrence would mean re-route |

## The power story (issues 2 and 3, and why they were hard)

The accessory outlet is the designed power source for the enclosure, and at idle it looks perfect:
the enclosure's own telemetry (the ASMedia bridge reports `supplyVoltage`/`supplyCurrent` over USB)
showed a healthy 13.4 V at 2 A. The failure only appears under load, and the decisive measurement
took days to capture because **the bridge is single-owner** — only modeld can hold the USB device,
so nothing can sample voltage concurrently while a load attempt is wedging it.

When the numbers finally landed, they were unambiguous:

```
t+52s:  13.41 V @ 1.94 A   (GPU idle)
t+53s:  12.98 V @ 3.26 A   (GPU starting work)   → ΔV/ΔI ≈ 0.33 Ω
later, full series:  13.59 V no-load → 11.76 V at 3–4 A  → ~0.45 Ω real path
```

That resistance is *normal* for an accessory circuit — it's built for 1–2 A dashcams. A stock
~150 W GPU pulls 11 A+ at 12 V; across 0.45 Ω that's a ~5 V sag, which collapses the input below
the enclosure's cutoff. Hence the exact signature: load survives (moderate current), then
inference boost transients hit and tinygrad raises `Device hang detected` within the first minute.

**The fix is one SMU message.** After opening the device — and critically, *before* the ~1.7 GB
model transfer, which also runs uncapped boost transients — send `PPSMC_MSG_SetPptLimit` (80 W
default here, tunable via `/data/vbsm_gpu_ppt_w`, clamped 40–220, 0 disables) and read it back:

```
chestnut ppt limit: requested=80 applied=80
```

Measured result: the driving model actually draws **24–34 W at 2–7 % GPU utilization**. The cap
costs nothing in throughput — its entire job is bounding boost transients the model never needed.
First capped drive: 27.5 minutes of continuous GPU inference, zero gaps, 56 °C steady.

## The wedge (issue 3, the one that defeats naive fixes)

When the link or power disturbs the transfer, the loader does not always *fail* — it can block
inside a C-level USB call **while holding the Python GIL**. Every thread in the process freezes,
including the timeout thread you wrote to guard against exactly this. Field evidence: a modeld
that ignored SIGINT at shutdown and needed the manager's SIGKILL escalation, with its own
60-second timeout never having fired.

Consequences worth internalizing:

- **In-process fallbacks are structurally unreachable** in this failure mode. The working design
  is fallback by *process replacement*: write a per-boot veto marker (`/dev/shm`, so a reboot
  grants a fresh chance), `os._exit(1)` past the wedged thread, and let the manager's restart
  policy respawn. The respawn sees the veto, skips the GPU, and loads the SoC model in ~2 s.
- The first load failure of a boot exits **without** the veto — cold-start rail transients
  deserve one retry on a settled rail (this retry has already absorbed two real one-off failures).
  The second failure vetoes the boot.
- An external watchdog owns the case where even the exit is unreachable: `UsbGpuLoading`
  persisting past 90 s (every legitimate path resolves by 60) means a wedge; SIGKILL from outside
  the process is the only mechanism a GIL-held process cannot ignore.
- `HCQDEV_WAIT_TIMEOUT_MS=3000` matters: it converts many would-be wedges into clean 3-second
  failures with a real traceback.

## Diagnostic playbook (what actually discriminated the causes)

- **dmesg is the cable's confession.** A bad cable throws `error -71` (EPROTO) and reset storms
  *at enumeration and at idle*, before any real load. A good cable enumerates in one line. If
  resets only appear under load, think power, not cable.
- **Check the negotiated speed, not just presence.** `cat /sys/bus/usb/devices/*/speed` — a
  marginal long cable silently negotiates 480 (USB 2), which "works" but makes a 1.7 GB load blow
  any sane timeout. 5000 or it fails validation.
- **The enclosure logs its own supply.** `chestnutState.supplyVoltage/Current` from the bridge is
  the only voltage measurement at the actual load point; wiring-side theories are guesses without it.
- **Time the load.** Healthy band here: 39–52 s for a ~1.7 GB bundle at 5 Gb/s. A "successful"
  load well outside the band is a degraded link telling you about your future.
- **Watch the exit codes.** `dead with 0/-2` = clean stop; `exit 1` = the Python raise path ran;
  `-9` = something external had to kill it — either a designed kill or a wedge that ignored SIGINT.
- **Distrust every clock.** The RTC resets on offline boots; the first minutes of every boot log
  into the same stale window, and crash files written pre-NTP carry stale names *and* stale
  mtimes. Cross-reference by boot, not by timestamp.

## Driver-facing state (issue 7)

The mici HUD's eGPU icon, once un-gated for bundle installs: **pulsing** = loading (wait),
**green** = big model live on the GPU, **orange** = fell back to the SoC model (fully drivable),
**crossed** = engaged while the GPU was failed. Rule of thumb: pulsing = wait, green or orange = go.

## What stayed unsolved on purpose

Rewiring (a fused battery tap) would remove the power ceiling entirely and is the "correct"
hardware fix. The 80 W cap made it unnecessary for this use case: the model needs a fraction of
that budget, so the outlet install — the whole point of a plug-in eGPU — stands.

## Cold-boot lock contention (issue 8, 2026-09-13)

**Symptom.** "Big Model Failed / Restart the car to retry" seconds into the first drive after a cold
boot; the drive ran on the SoC model for ~2.5 min until the watchdog's standstill kick reloaded the
eGPU, which then ran the rest of the drive clean. The driver's unplug/replug of the enclosure came
after that drive had ended and changed nothing; the next start loaded first try like every warm start.

**Cause.** The first attempt failed 7 ms after "loading model": tinygrad could not take its exclusive
flock on the USB GPU (`Failed to acquire lock file am_usb:4-2.lock`), i.e. another process had the
device open at that instant. The enclosure was healthy the whole time (13.7 V, 1.5 A idle). The error
arrives wrapped in tinygrad's `ExceptionGroup: No interface for AMD:0 is available` — the first two
sub-errors (no `/dev/kfd`, no PCI bus) are always true on a comma four; the lock error is the real one.

**What was wrong in the fork.** The lock-holder capture keyed on `str(e)` of the outer group, which
never contains the lock text, so `fuser` never ran and the holder went unrecorded. Fixed: detection
reads the full traceback text, the capture runs on every contention, and the loader retries lock
contention up to 5 times 2 s apart before falling back (`VBSM_GPU_LOCK_RETRY`).

**Reading the logs.** The first minutes of a boot log with the stale Jul-28 clock until NTP syncs, so
filter boot records by their `created` window, not wall time; the rlog's `logMessage` keeps the full
traceback when the swaglog files are noisy. `chestnutState.pcieLtssm/powerLimitW/powerDrawW` are only
populated once modeld owns the AMD device — zeros before that are an artifact, not a dead link.

**Follow-up (same day).** Why the driver stayed locked out after the fallback: the watchdog kick fired
10 s after cruise was armed and threw the working SoC model away for a 35 s reload ("Big Model Loading /
openpilot Unavailable"); two false alerts fired at the swap; and locationd, starved of camera odometry
during the gap, rejected a 2 s burst of gyro samples whose unbounded counter then took ~10 minutes to
decay — every engage refused with "locationd Temporary Error". Fixes: the kick now waits for cruise
main off or Park; the alerts are gated on a real success and the settling window; the locationd counter
is capped so recovery is bounded to ~30 s. A truly driveable reload (publish the SoC model while the big
one loads) was evaluated and rejected for now: both models run their warp on the SoC GPU through
tinygrad, which is not thread-safe, and a swap while engaged would trip the model-lagging soft disable.

## Executable bit lost in an API commit (issue 9, 2026-09-14)

**Symptom.** Two drives with no driving model at all: pulsing chestnut icon, "Posenet Speed Invalid / Speed
Error: nan m/s", "Process Not Running: modeld_tinygrad". Nothing to do with the GPU.

**Cause.** The lock-retry and fallback commits were written through the GitHub Git Data API with mode
100644 for every file. `modeld.py` is 100755 in the tree and the `modeld_tinygrad` launcher execs it
directly, so the manager logged "died with exitcode 126" (cannot execute) five times per drive and parked
the process under the VBSM_RESTART per-drive cap. selfdrived.py and locationd.py lost the bit too but are
imported, not exec'd.

**Fix.** `chmod +x` on the running tree over comma prime's SSH proxy (`ssh.comma.ai`) during the drive; the
manager's restart budget resets on the next ignition edge. On the branch: a rollback to the known-good tree,
then the same fixes re-committed with every path's mode read from the parent tree.

**Rules that came out of it.** Read modes with `git ls-tree <parent> -- <path>` for every API commit; after a
deploy verify the exec bits with `ls -l` and that modeld_tinygrad is actually running, not just that the file
compiles. Manager exit code 126 means "not executable" and 127 "not found"; neither is a Python error.

## Cold boot: PCIe link not up, and why retries could not help (issue 10, 2026-09-14)

**Symptom.** After a cold boot the first modeld process ran the small model and the driver restarted the car to get the
big one (the watchdog kick had also fired, 70 s after the failure). Warm starts were fine.

**Cause, proven on the device.** tinygrad's AMD open takes its lock, powers PCIe on (0xF3=1, the custom firmware boots
with it off) and reads the LTSSM once: "PCIe link not up (LTSSM=0x00), custom firmware not ready" if training is not
finished. That failure leaves the lock fd open in the process (dropping the exception and `gc.collect()` do not free it)
and the libusb interface claimed, so every later open in the same process fails: first at the lock, and after closing the
leaked fd, with `libusb_set_configuration: Resource busy`. The 09-13 retry ladder therefore burned five attempts against
the process's own descriptor; `fuser` listed only modeld itself. A fresh process (kick or restart) finds the link up.

**Second cause, same day.** Importing modeld_v2 already touches the AMD device: `compile_modeld.py` evaluates
`Device.DEFAULT` in a default argument, and tinygrad's default-device probe tries AMD first. At a cold boot that import-time
open loses the same link race, so by the time `load_big()` runs the process is already poisoned -- which is why the
2026-09-13 failure hit the lock 7 ms in with no other holder. Pinning `DEV=QCOM` before tinygrad is imported removes the
probe (verified: importing the module opens nothing).

**Fix (revision 2, same day; the first revision e0c253de never drove).** Pin `DEV=QCOM` before tinygrad is
imported. Before the first open, read the bridge's supply voltage and LTSSM over usbdevfs (the read-only control
transfers `chestnut_power.py status` uses; own fd, no tinygrad, no lock). With 12 V present and the link down, write
0xF3=1 once, right after the `ChestnutLoading` param so training overlaps the vision-stream wait, re-send only if the
LTSSM is still in Detect (0x00/0x01) 10 s later, and poll at 2 Hz until L0 (0x78) or the budget ends: 20 s, and by
pid age 22 s at the latest, so probe + the 60 s loader budget + the SoC load stay inside ui_watchdog's 90 s deadline
(the pid age comes from `/proc/self/stat`, never the wall clock, which jumps at the boot-time NTP sync). `no_12v`
(dead outlet) and `absent` skip the open without a strike, so the kick can retry once the outlet is live; `timeout`
skips the open and counts a strike; a probe bug (`probe_error`) opens anyway. Retry an open only when the failure was
raised inside tinygrad's `flock_acquire` while the process held no lock fd beforehand (an external holder), after
closing the unlocked fd tinygrad leaves behind; never after any other failure. Every failed open logs the real
sub-exceptions with file:line, the flock classification and the fd counts; the first open's duration is logged as
`open_ms`. The first revision polled `flash.link_up()`, which writes 0xF3=1 on every probe (live link included) and
whose 20 s could overrun the watchdog deadline; its retry rule ("no leaked fd") could never fire because a refused
flock still leaves an unlocked fd behind.

**Bench (car off, rails off, offroad checked before every step).** Importing the module opens nothing and leaves
`DEV=QCOM` and no lock fd (pid age 7 s after imports). The raw read shows 72 mV / LTSSM 0x00 (unpowered links read
0x00 or 0x01), and the wait returns `no_12v` after three reads in 1.0 s with zero F3 writes and no lock fd. A forced
open in a throwaway process fails at usb.py:143 with one leaked fd and classifies as not retryable; a second open in
that process fails at the flock on its own fd and is refused; with an external holder of `/tmp/am_usb:4-2.lock` the
first open fails at the flock with no prior fd, the retry passes the flock once the holder is gone, and `sudo fuser`
names the holder. The 12 V paths (`ready` with one F3 write, the first open's `open_ms`, `timeout`) can only be
observed at the next ignition: expect, from the first modeld pid, `chestnut preflight ... ltssm=0x00`, `chestnut
link ready reason=ready f3_writes=1 wait_s=X`, `chestnut ppt limit ... open_ms`, `models loaded in ~21s`.
