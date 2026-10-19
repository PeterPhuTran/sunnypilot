#!/usr/bin/env python3
"""VBSM_WATCHDOG: recover a mici UI that is alive but has stopped drawing,
and (VBSM_GPU_WEDGE) replace a modeld whose eGPU load wedged.

Division of labour with the manager: VBSM_RESTART in system/manager/process.py
reaps a dead child so start() can rebuild it, which is what restores the
process AND clears the processNotRunning soft-disable. A *frozen* process is
still alive, so no restart policy can see it — that is this watchdog's UI job.
It proves the UI has stopped presenting frames, kills it, and lets the
manager bring it back properly.

It deliberately spawns nothing. The v2 watchdog spawned replacements itself,
which was wrong twice over: the replacement was not manager-tracked (so
managerState still reported ui down and openpilot stayed soft-disabled), and a
spawn racing a live UI produced a second 'uiDebug' publisher, whose
MultiplePublishersError took down the real UI mid-drive.

Liveness is the 'uiDebug' message that ui.py publishes once per rendered
frame. gui_app.render() skips the loop body — and so the publish — whenever
the screen is off, but onroad the screen is always awake (ui_state:
awake = ignition or not interaction_timeout), so onroad "no uiDebug" means
"not drawing frames". That is a direct signal. It replaces a CPU-tick
heuristic that could not tell a busy UI from an idle sibling: that heuristic
matched the substring "selfdrive.ui", which also matches
'openpilot.selfdrive.ui.soundd', and killed soundd for idling as designed.
Process identity is therefore matched on the exact proctitle, never a prefix.

VBSM_GPU_WEDGE: a big-model load can wedge inside a C call that HOLDS THE GIL,
which freezes every thread in modeld, its own 60 s timeout and SIGINT handler
included (route 81: only the manager's car-off SIGKILL ended it). Upstream has
no exit from that state. ChestnutLoading still set past the load deadline is
definitive, so this watchdog ends it the way upstream ends any failed load:
the first time in an ignition it spends upstream's one retry (ChestnutRetry +
OnroadCycleRequested, the onroad cycle modeld itself requests) and does NOT
kill: the cycle's own stop (SIGINT, then the manager's 5 s join and SIGKILL in
start()) replaces the frozen process, with no VBSM_RESTART respawn racing the
cycle. If the same process is still loading GPU_CYCLE_GRACE_S later, or the
retry is spent, or openpilot/MADS is engaged, it reports the failure
(ChestnutActive False) and SIGKILLs; the VBSM_RESTART respawn runs the SoC
model (modeld skips the eGPU when it is not the first of its onroad session).
modeld is matched on its exact proctitle (modeld_warmup's contains ".modeld").
The fork's own retry paths (the late-enumeration kick, the rail-gated
mid-drive retry, the driver request, the veto/strike files) were retired on
2026-10-19 in favour of upstream's retry.

The same backstop covers a modeld that freezes AFTER its load: one that has
published modelV2 and then publishes nothing for GPU_FREEZE_S while its pid
lives (the cold-cache heal's background eGPU build holding the GIL under the
SoC model, or a big-model run() that never returns) is SIGKILLed, and the
respawn runs the SoC model. Armed only by that pid's own output, so a load or
a camerad wait before the first frame never counts.
"""
import math
import os
import signal
import time

from openpilot.cereal import messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

UI_PROCTITLE = "openpilot.selfdrive.ui.ui"
MODELD_PROCTITLE = "openpilot.selfdrive.modeld.modeld_tinygrad"   # VBSM_GPU_WEDGE: modeld_v2's setproctitle(PROCESS_NAME)
EVIDENCE_DIR = "/data/vbsm_eval"

CHECK_INTERVAL = 2.0
STALL_TIMEOUT = 30.0     # onroad seconds with no rendered frame = frozen
MIN_ONROAD_S = 90.0      # let the camera stack and first model runs settle
MAX_KILLS_PER_BOOT = 3
MIN_KILL_GAP_S = 60.0    # manager restart + UI startup needs room to finish

# VBSM_GPU_WEDGE: every legitimate load path resolves 12 s inside this: the link wait (<= 22 s after
# main(), more only with VBSM_GPU_COLDWAIT, below) + the 60 s loader budget (more only with
# VBSM_GPU_COLDCACHE, below) + <= 6 s for the small model (more only for a cold SoC cache, also
# VBSM_GPU_COLDCACHE) or for the onroad cycle after a failed load
GPU_LOAD_DEADLINE_S = 100.0
# after spending upstream's retry: how long the onroad cycle gets to replace the frozen process (its SIGINT,
# then the manager's 5 s join + SIGKILL in start(), ~7 s in all) before this watchdog kills it itself
GPU_CYCLE_GRACE_S = 15.0
# VBSM_GPU_COLDWAIT / VBSM_GPU_COLDCACHE: modeld writes "<pid> <seconds>" when its link wait ran past the plain
# budget, its loader found a cold big-model cache and/or the SoC model's cache is cold; the load deadline moves by
# that much for that pid only, bounded
GPU_PREFLIGHT_EXT_FILE = "/dev/shm/vbsm_gpu_preflight_ext"
GPU_PREFLIGHT_EXT_MAX_S = 155.0  # modeld's worst case: 35 s link wait (20 s extension + poll + two 2 s reads + a 10 s F3 write) + 60 s cold big-model cache + 60 s cold small-model cache
# VBSM_GPU_WEDGE (freeze): a modeld that published modelV2 and then stays silent this long while alive is frozen
# with the GIL held: the cold-cache heal's background eGPU build under the SoC model, or a big-model run() that
# never returns. The respawn runs SoC. Cost: a camerad stall longer than this also costs one modeld respawn
# (selfdrived has soft-disabled on the missing modelV2 long before)
GPU_FREEZE_S = 15.0


def preflight_ext_s(pid):
  """VBSM_GPU_COLDWAIT / VBSM_GPU_COLDCACHE: the load-deadline extension modeld `pid` reported: the link-wait
  seconds past its plain budget plus 60 s for each cold model cache (0 if none or unreadable), clamped."""
  try:
    with open(GPU_PREFLIGHT_EXT_FILE) as f:
      p, s = f.read().split()
    v = float(s)
    return min(max(v, 0.0), GPU_PREFLIGHT_EXT_MAX_S) if math.isfinite(v) and int(p) == pid else 0.0   # nan would disarm the kill
  except (OSError, ValueError):
    return 0.0


def find_exact(title):
  """PID whose proctitle is exactly `title`. Exact, never a substring: 'openpilot.selfdrive.ui.soundd' shares
  the UI's prefix, and 'openpilot.sunnypilot.modeld_v2.warmup' (upstream's offroad modeld_warmup, which can
  still be alive for a moment at an onroad edge) contains '.modeld'."""
  for entry in os.listdir("/proc"):
    if not entry.isdigit():
      continue
    try:
      with open(f"/proc/{entry}/cmdline", "rb") as f:
        cmd = f.read().rstrip(b"\0").replace(b"\0", b" ").decode(errors="replace")
    except OSError:
      continue
    if cmd == title:
      return int(entry)
  return None


def find_ui():
  """PID of the UI, matched on its exact proctitle."""
  return find_exact(UI_PROCTITLE)


def dump_evidence(pid):
  """wchan + per-thread kernel stacks name the blocked syscall.

  This is also the evidence comma asks for on the splash-hang issue class
  (commaai/openpilot#37845), so keep it even when recovery works.
  """
  try:
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    path = os.path.join(EVIDENCE_DIR, f"ui_stall_{int(time.time())}.log")
    with open(path, "w") as out:
      for name in ("status", "wchan", "syscall", "stack"):
        try:
          with open(f"/proc/{pid}/{name}") as f:
            out.write(f"### {name}\n{f.read()}\n")
        except OSError as e:
          out.write(f"### {name} unreadable: {e}\n")
      try:
        for tid in sorted(os.listdir(f"/proc/{pid}/task")):
          for name in ("wchan", "stack"):
            try:
              with open(f"/proc/{pid}/task/{tid}/{name}") as f:
                out.write(f"### task {tid} {name}\n{f.read()}\n")
            except OSError:
              pass
      except OSError:
        pass
    return path
  except OSError:
    return None


class GpuWedge:
  """VBSM_GPU_WEDGE state; one tick per watchdog loop pass."""

  def __init__(self, params):
    self.params = params
    self.modeld_seen = None  # (pid, monotonic first seen)
    self.cycle_asked = None  # (pid, monotonic) when this watchdog spent upstream's retry for that process
    self.model_seen = None   # (pid, monotonic) of the last modelV2 seen while that pid ran

  def tick(self, sm, started, now):
    if not started:
      self.modeld_seen = None
      self.cycle_asked = None
      self.model_seen = None
      return

    pid = find_exact(MODELD_PROCTITLE)
    if pid is None:
      self.modeld_seen = None
      self.model_seen = None
      return
    first = self.modeld_seen is None or self.modeld_seen[0] != pid
    if first:
      self.modeld_seen = (pid, now)
    # VBSM_GPU_WEDGE (freeze): armed only by this pid's own modelV2, never on the tick that first sees it (that
    # update can still hold the previous pid's last message), so a load or a camerad wait before the first frame
    # never counts; then GPU_FREEZE_S without one while the pid lives = frozen
    if sm.updated['modelV2'] and not first:
      self.model_seen = (pid, now)
    elif self.model_seen is not None and self.model_seen[0] == pid and now - self.model_seen[1] > GPU_FREEZE_S:
      cloudlog.error(f"ui_watchdog: modeld pid {pid} published no modelV2 for {now - self.model_seen[1]:.0f}s while alive, "
                     f"killing for a SoC respawn")
      try:
        os.kill(pid, signal.SIGKILL)
      except OSError as e:
        cloudlog.error(f"ui_watchdog: freeze kill failed: {e}")
      self.model_seen = None
      return

    load_age = now - self.modeld_seen[1]
    if not (self.params.get_bool("ChestnutLoading") and load_age > GPU_LOAD_DEADLINE_S + preflight_ext_s(pid)):   # VBSM_GPU_COLDWAIT
      return
    if self.cycle_asked is not None and self.cycle_asked[0] == pid and now - self.cycle_asked[1] < GPU_CYCLE_GRACE_S:
      return   # the onroad cycle replaces the frozen process: SIGINT, then the manager's join + SIGKILL
    # ChestnutLoading is only ever True in the FIRST modeld of an onroad session, before it has published
    # a single modelV2 (bigModelLoading NO_ENTRY the whole time), so nobody can be engaged on it here; checked
    # anyway, because an onroad cycle under an engaged driver would stop selfdrived, card and controlsd
    engaged = bool((sm.alive['selfdriveState'] and sm['selfdriveState'].enabled) or
                   (sm.alive['selfdriveStateSP'] and sm['selfdriveStateSP'].mads.active))
    # one cycle per process; ChestnutRetry makes it one per ignition. A cycle_asked left from an earlier pid means an
    # offroad blip (the cycle, or a 1-2 s car restart) fell between two ticks, not that this ignition's retry is spent
    if not self.params.get_bool("ChestnutRetry") and not engaged and (self.cycle_asked is None or self.cycle_asked[0] != pid):
      self.params.put_bool("ChestnutRetry", True, block=True)
      self.params.put_bool("OnroadCycleRequested", True, block=True)
      self.cycle_asked = (pid, now)
      cloudlog.error(f"ui_watchdog: modeld pid {pid} wedged in eGPU load ({load_age:.0f}s, preflight ext "
                     f"{preflight_ext_s(pid):.1f}s), spending upstream's retry (onroad cycle), not killing")
      return
    # the same verdict as modeld's own final failure; ChestnutLoading False also marks the session for the respawn
    self.params.put_bool("ChestnutActive", False, block=True)
    self.params.put_bool("ChestnutLoading", False, block=True)
    cloudlog.error(f"ui_watchdog: modeld pid {pid} wedged in eGPU load ({load_age:.0f}s, preflight ext "
                   f"{preflight_ext_s(pid):.1f}s, engaged {engaged}, cycle asked {self.cycle_asked is not None}), "
                   f"final, killing for a SoC respawn")
    try:
      os.kill(pid, signal.SIGKILL)
    except OSError as e:
      cloudlog.error(f"ui_watchdog: wedge kill failed: {e}")
    self.modeld_seen = None


def main():
  params = Params()
  sm = messaging.SubMaster(['uiDebug', 'deviceState', 'selfdriveState', 'selfdriveStateSP', 'modelV2'])
  gpu = GpuWedge(params)
  now = time.monotonic()
  last_frame = now
  onroad_since = None
  kills = 0
  last_kill = 0.0

  while True:
    time.sleep(CHECK_INTERVAL)
    now = time.monotonic()
    sm.update(0)

    if sm.updated['uiDebug']:
      last_frame = now

    started = bool(sm['deviceState'].started) if sm.seen['deviceState'] else False

    gpu.tick(sm, started, now)

    if not started:
      # offroad the screen is allowed to sleep, which legitimately stops
      # rendering: there is no liveness signal to judge here.
      onroad_since = None
      last_frame = now
      continue

    if onroad_since is None:
      onroad_since = now
      last_frame = now
      continue

    if now - onroad_since < MIN_ONROAD_S or now - last_frame < STALL_TIMEOUT:
      continue
    if kills >= MAX_KILLS_PER_BOOT or now - last_kill < MIN_KILL_GAP_S:
      continue

    if not sm.seen['uiDebug']:
      # never received a single frame message, so silence proves nothing about
      # the UI — it could equally be this subscription that is broken, and
      # killing on that would be the same class of false positive as v2.
      # A UI that never starts is a death, which the manager already owns.
      continue

    pid = find_ui()
    if pid is None:
      # already gone; the manager's restart policy owns this case
      last_frame = now
      continue

    evidence = dump_evidence(pid)
    kills += 1
    last_kill = now
    last_frame = now
    cloudlog.error(f"ui_watchdog: ui pid {pid} drew no frame for {STALL_TIMEOUT:.0f}s onroad, "
                   f"killing for manager restart ({kills}/{MAX_KILLS_PER_BOOT}), evidence={evidence}")
    try:
      os.kill(pid, signal.SIGKILL)
    except OSError as e:
      cloudlog.error(f"ui_watchdog: kill failed: {e}")


if __name__ == "__main__":
  main()
