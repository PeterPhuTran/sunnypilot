#!/usr/bin/env python3
"""VBSM_WATCHDOG: recover a UI that is alive but never paints.

Failure class seen on-road: the UI process starts, initializes raylib, then
deadlocks before presenting its first frame (suspected EGL setup contending
with the driving model's first-boot kernel compilation). The process is alive,
so the ordinary process watchdog sees nothing, and the boot splash stays on
screen for the whole drive.

Detection: while onroad (camerad present), a healthy UI renders continuously
and burns CPU; a deadlocked one accrues essentially none. If the UI process
gains less than ~1s of CPU time over a 30s window after the system has been
onroad for 90s, it is stalled: capture /proc evidence for the post-mortem,
kill it, and let the manager respawn it. By then the boot storm has usually
passed and the respawn paints. At most 3 kills per boot, so a genuinely
broken UI ends up parked rather than thrashing.
"""
import os
import signal
import time

from openpilot.common.swaglog import cloudlog

EVIDENCE_DIR = "/data/vbsm_eval"
CHECK_INTERVAL = 5.0
WINDOW = 30.0
MIN_ONROAD_S = 90.0
MIN_ACTIVE_TICKS = 100   # < ~1s CPU over the window while onroad = stalled
MAX_KILLS_PER_BOOT = 3


def scan_procs():
  """One /proc pass: (ui_pid, camerad_running)."""
  ui, camerad = None, False
  for p in os.listdir("/proc"):
    if not p.isdigit():
      continue
    try:
      with open(f"/proc/{p}/cmdline", "rb") as f:
        cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
      continue
    if "camerad" in cmd and "watchdog" not in cmd:
      camerad = True
    if "selfdrive.ui" in cmd and "watchdog" not in cmd:
      ui = int(p)
  return ui, camerad


def cpu_ticks(pid):
  try:
    with open(f"/proc/{pid}/stat") as f:
      parts = f.read().rsplit(")", 1)[1].split()
    return int(parts[11]) + int(parts[12])  # utime + stime
  except (OSError, IndexError, ValueError):
    return None


def dump_evidence(pid):
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
  onroad_since = None
  prev = None  # (pid, ticks, monotonic)
  kills = 0
  while True:
    time.sleep(CHECK_INTERVAL)
    now = time.monotonic()
    ui, camerad = scan_procs()
    if not camerad:
      onroad_since = None
      prev = None
      continue
    if onroad_since is None:
      onroad_since = now
    if ui is None:
      prev = None  # manager owns respawn
      continue
    ticks = cpu_ticks(ui)
    if ticks is None:
      prev = None
      continue
    if prev is None or prev[0] != ui:
      prev = (ui, ticks, now)
      continue
    if now - prev[2] < WINDOW:
      continue
    delta = ticks - prev[1]
    if now - onroad_since >= MIN_ONROAD_S and delta < MIN_ACTIVE_TICKS and kills < MAX_KILLS_PER_BOOT:
      evidence = dump_evidence(ui)
      kills += 1
      cloudlog.error(f"ui_watchdog: ui pid {ui} stalled ({delta} ticks/{WINDOW:.0f}s onroad), "
                     f"kill {kills}/{MAX_KILLS_PER_BOOT}, evidence={evidence}")
      try:
        os.kill(ui, signal.SIGKILL)
      except OSError:
        pass
      prev = None
    else:
      prev = (ui, ticks, now)


if __name__ == "__main__":
  main()
