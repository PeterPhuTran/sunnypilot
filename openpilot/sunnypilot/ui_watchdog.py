#!/usr/bin/env python3
"""VBSM_WATCHDOG: keep the mici UI alive — respawn on death, recover stalls.

Route-46 post-mortem facts this design rests on:
- This tree's manager NEVER respawns a process that dies mid-session
  (ManagerProcess.start() no-ops while self.proc is set), so a UI crash
  leaves the boot splash up for the rest of the drive.
- The observed failures were (a) a crash (NameError) with no respawn and
  (b) an offroad boot where the UI completed construction then never
  presented its first frame (upstream issue class: commaai/openpilot#37845).

So this watchdog owns recovery itself:
- DEAD:  if no UI process exists (onroad or offroad), spawn a replacement
  directly with the same interpreter and environment.
- STALL: while onroad, a healthy UI renders continuously and burns CPU; if
  the UI process gains < ~1s CPU over a 30s window after 90s onroad, dump
  /proc evidence (wchan + kernel stacks name the blocked syscall — also the
  evidence comma asks for on #37845), kill it, and respawn.

Rate limits: max 6 respawns per boot, min 20s apart — a deterministically
crashing UI ends up parked with its crash files rather than thrashing.
NOTE: crash files under /data/community/crashes carry a stale Jul-28 date
when the boot predates NTP sync; search by mtime, not name.
"""
import os
import signal
import subprocess
import sys
import time

from openpilot.common.swaglog import cloudlog

EVIDENCE_DIR = "/data/vbsm_eval"
CHECK_INTERVAL = 5.0
WINDOW = 30.0
MIN_ONROAD_S = 90.0
MIN_ACTIVE_TICKS = 100   # < ~1s CPU over the window while onroad = stalled
MAX_RESPAWNS_PER_BOOT = 6
MIN_RESPAWN_GAP_S = 20.0
STARTUP_GRACE_S = 30.0   # leave a freshly spawned UI alone this long


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


def spawn_ui():
  # Same interpreter and environment as this managed process; own session so
  # a watchdog restart never takes the UI down with it.
  return subprocess.Popen(
    [sys.executable, "-c", "from openpilot.selfdrive.ui.ui import main; main()"],
    cwd="/data/openpilot", env=os.environ.copy(),
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    start_new_session=True,
  )


def main():
  onroad_since = None
  prev = None  # (pid, ticks, monotonic)
  respawns = 0
  last_respawn = 0.0
  boot_mono = time.monotonic()
  while True:
    time.sleep(CHECK_INTERVAL)
    now = time.monotonic()
    ui, camerad = scan_procs()

    if not camerad:
      onroad_since = None
    elif onroad_since is None:
      onroad_since = now

    def may_respawn():
      return (respawns < MAX_RESPAWNS_PER_BOOT
              and now - last_respawn >= MIN_RESPAWN_GAP_S
              and now - boot_mono >= STARTUP_GRACE_S)

    # DEAD: manager will not bring it back; we do.
    if ui is None:
      prev = None
      if may_respawn():
        respawns += 1
        last_respawn = now
        cloudlog.error(f"ui_watchdog: ui dead, respawning ({respawns}/{MAX_RESPAWNS_PER_BOOT})")
        try:
          spawn_ui()
        except OSError as e:
          cloudlog.error(f"ui_watchdog: respawn failed: {e}")
      continue

    # STALL (onroad only — an idle offroad UI legitimately burns ~no CPU)
    if onroad_since is None:
      prev = None
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
    if now - onroad_since >= MIN_ONROAD_S and delta < MIN_ACTIVE_TICKS and may_respawn():
      evidence = dump_evidence(ui)
      respawns += 1
      last_respawn = now
      cloudlog.error(f"ui_watchdog: ui pid {ui} stalled ({delta} ticks/{WINDOW:.0f}s onroad), "
                     f"kill+respawn {respawns}/{MAX_RESPAWNS_PER_BOOT}, evidence={evidence}")
      try:
        os.kill(ui, signal.SIGKILL)
      except OSError:
        pass
      try:
        spawn_ui()
      except OSError as e:
        cloudlog.error(f"ui_watchdog: respawn failed: {e}")
      prev = None
    else:
      prev = (ui, ticks, now)


if __name__ == "__main__":
  main()
