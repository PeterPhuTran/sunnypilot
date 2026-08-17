#!/usr/bin/env python3
"""VBSM_WATCHDOG: recover a mici UI that is alive but has stopped drawing.

Division of labour with the manager: VBSM_RESTART in system/manager/process.py
reaps a dead child so start() can rebuild it, which is what restores the
process AND clears the processNotRunning soft-disable. A *frozen* process is
still alive, so no restart policy can see it — that is this watchdog's only
job. It proves the UI has stopped presenting frames, kills it, and lets the
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

NOTE: crash files under /data/community/crashes carry a stale Jul-28 date when
the boot predates NTP sync; search by mtime, not by name.
"""
import os
import signal
import time

from openpilot.cereal import messaging
from openpilot.common.swaglog import cloudlog

UI_PROCTITLE = "openpilot.selfdrive.ui.ui"
EVIDENCE_DIR = "/data/vbsm_eval"

CHECK_INTERVAL = 2.0
STALL_TIMEOUT = 30.0     # onroad seconds with no rendered frame = frozen
MIN_ONROAD_S = 90.0      # let the camera stack and first model runs settle
MAX_KILLS_PER_BOOT = 3
MIN_KILL_GAP_S = 60.0    # manager restart + UI startup needs room to finish


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


def main():
  sm = messaging.SubMaster(['uiDebug', 'deviceState'])
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
