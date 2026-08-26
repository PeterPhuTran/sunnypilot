#!/usr/bin/env python3
"""VBSM_WATCHDOG: recover a mici UI that is alive but has stopped drawing,
and (VBSM_GPU_KICK) cycle modeld onto the chestnut GPU when it boots late.

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

VBSM_GPU_KICK: the chestnut enclosure is powered from the car's SWITCHED 12V
outlet, so it boots at the same moment openpilot does and loses the race —
modeld evaluates usbgpu_present() exactly once at startup, so a drive that
starts before the enclosure enumerates runs the whole way on the SoC with the
GPU idle. When the GPU is provably usable (same usbgpu_present() check modeld
itself runs), the models manager has finished switching the active bundle to
the AMD catalog, and the running modeld reports it booted without the GPU
(UsbGpuLoading False), this watchdog kills modeld ONCE at a safe moment —
standing still, openpilot not engaged — and the manager's restart policy
brings it back; the fresh modeld re-runs its GPU check and loads the AMD
bundle from the pre-cached chunks. Bundle ping-pong across power cycles is
the manager's own per-catalog stash/restore and needs no help from here.

NOTE: crash files under /data/community/crashes carry a stale Jul-28 date when
the boot predates NTP sync; search by mtime, not by name.
"""
import os
import signal
import time

from openpilot.cereal import messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.modeld.helpers import usbgpu_present

UI_PROCTITLE = "openpilot.selfdrive.ui.ui"
EVIDENCE_DIR = "/data/vbsm_eval"
EVIDENCE_CRASH_DIR = "/data/community/crashes"

CHECK_INTERVAL = 2.0
STALL_TIMEOUT = 30.0     # onroad seconds with no rendered frame = frozen
MIN_ONROAD_S = 90.0      # let the camera stack and first model runs settle
MAX_KILLS_PER_BOOT = 3
MIN_KILL_GAP_S = 60.0    # manager restart + UI startup needs room to finish

GPU_VETO_FILE = "/dev/shm/vbsm_usbgpu_veto"  # tmpfs: a reboot grants a fresh chance
GPU_STABLE_S = 10.0      # enclosure must stay enumerated this long
MODELD_SETTLE_S = 15.0   # give a fresh modeld time to write UsbGpuLoading
GPU_KICK_COOLDOWN_S = 120.0
MAX_GPU_KICKS = 2        # per boot; a modeld that then dies on AMD is the
                         # manager restart policy's problem, not a kick loop
GPU_VEGO_MAX = 0.5       # m/s — only ever kick at a standstill


def find_proc(match, exclude=()):
  """PID whose proctitle contains match and none of exclude."""
  for entry in os.listdir("/proc"):
    if not entry.isdigit():
      continue
    try:
      with open(f"/proc/{entry}/cmdline", "rb") as f:
        cmd = f.read().rstrip(b"\0").replace(b"\0", b" ").decode(errors="replace")
    except OSError:
      continue
    if match in cmd and not any(x in cmd for x in exclude):
      return int(entry)
  return None


def find_ui():
  """PID of the UI, matched on its exact proctitle.

  Exact, not substring: 'openpilot.selfdrive.ui.soundd' shares the prefix.
  """
  for entry in os.listdir("/proc"):
    if not entry.isdigit():
      continue
    try:
      with open(f"/proc/{entry}/cmdline", "rb") as f:
        cmd = f.read().rstrip(b"\0").replace(b"\0", b" ").decode(errors="replace")
    except OSError:
      continue
    if cmd == UI_PROCTITLE:
      return int(entry)
  return None


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


class GpuKick:
  """VBSM_GPU_KICK state; one tick per watchdog loop pass."""

  def __init__(self, params):
    self.params = params
    self.gpu_since = None
    self.modeld_seen = None  # (pid, monotonic first seen)
    self.gpu_was_active = False
    self.kicks = 0
    self.last_kick = 0.0

  def _veto(self):
    """Sustained inference hangs the accessory-outlet enclosure (tinygrad
    "Device hang detected"); without a veto every restart after a hang burned
    another 60s no-model window re-attempting the GPU mid-drive. One inference
    death = SoC for the rest of this boot; modeld checks the marker at start."""
    try:
      with open(GPU_VETO_FILE, "w") as f:
        f.write(str(int(time.time())))
      cloudlog.error("ui_watchdog: eGPU-active modeld died; vetoing the GPU until reboot")
    except OSError as e:
      cloudlog.error(f"ui_watchdog: veto write failed: {e}")

  def tick(self, sm, started, now):
    if not started:
      self.gpu_since = None
      self.modeld_seen = None
      self.gpu_was_active = False
      return

    if not usbgpu_present():
      self.gpu_since = None
      return
    if self.gpu_since is None:
      self.gpu_since = now

    pid = find_proc(".modeld", exclude=("dmonitoring", "watchdog"))
    if pid is None:
      # a GPU-active modeld that vanishes while still onroad, with a crash
      # file seconds old, died of the device hang -- a clean offroad stop
      # leaves no fresh crash file and never trips this
      if self.modeld_seen is not None and self.gpu_was_active and not os.path.exists(GPU_VETO_FILE):
        try:
          newest = max((os.path.getmtime(os.path.join(EVIDENCE_CRASH_DIR, f))
                        for f in os.listdir(EVIDENCE_CRASH_DIR)), default=0)
        except OSError:
          newest = 0
        if time.time() - newest < 90:
          self._veto()
      self.modeld_seen = None
      self.gpu_was_active = False
      return
    self.gpu_was_active = self.params.get_bool("UsbGpuActive")
    if self.modeld_seen is None or self.modeld_seen[0] != pid:
      self.modeld_seen = (pid, now)

    if now - self.gpu_since < GPU_STABLE_S or now - self.modeld_seen[1] < MODELD_SETTLE_S:
      return
    if self.kicks >= MAX_GPU_KICKS or now - self.last_kick < GPU_KICK_COOLDOWN_S:
      return
    if os.path.exists(GPU_VETO_FILE):
      return

    # the running modeld must itself say it booted without the GPU; a missing
    # param proves nothing and must never trigger a kill
    loading = self.params.get("UsbGpuLoading")
    if loading is None or self.params.get_bool("UsbGpuLoading"):
      return

    # the manager must have finished moving selection to the AMD catalog,
    # otherwise a restarted modeld would marry the GPU to a Qualcomm bundle
    active_json = self.params.get("ModelManager_ActiveJson") or ""
    if "usbgpu" not in str(active_json) or self.params.get("ModelManager_ActiveBundle") is None:
      return

    # only at a standstill with openpilot disengaged — never take the model
    # away from a moving car
    if not (sm.seen['carState'] and sm.seen['selfdriveState']):
      return
    if sm['carState'].vEgo > GPU_VEGO_MAX or sm['selfdriveState'].enabled:
      return

    self.kicks += 1
    self.last_kick = now
    cloudlog.error(f"ui_watchdog: gpu present but modeld pid {pid} booted without it; "
                   f"kicking for AMD reload ({self.kicks}/{MAX_GPU_KICKS})")
    try:
      os.kill(pid, signal.SIGKILL)
    except OSError as e:
      cloudlog.error(f"ui_watchdog: gpu kick failed: {e}")
    self.modeld_seen = None


def main():
  params = Params()
  sm = messaging.SubMaster(['uiDebug', 'deviceState', 'carState', 'selfdriveState'])
  gpu = GpuKick(params)
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
