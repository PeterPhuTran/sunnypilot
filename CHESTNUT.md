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
