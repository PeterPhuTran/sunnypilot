#!/usr/bin/env python3
"""
parkwatchd -- report parked/unlocked state of the car to the parkwatch notifier.

Why this exists
---------------
A Toyota auto-unlocks every door when you shift to Park, so the last lock state
openpilot sees before it goes offroad is ALWAYS "unlocked" -- it carries no
information about whether you then locked the car. The only evidence of a lock
is a ~5 second body-CAN wake burst the car emits when the fob (or a door-handle
touch sensor) is used while parked. Measured on this car 2026-08-30: the bus is
completely silent between events and wakes for 5.0-5.8s on each lock, unlock, or
door open, and every burst re-announces DOOR_LOCKS (0x638) at 1Hz.

The panda kills its CAN transceivers the moment ignition drops, so seeing those
bursts requires the pandad parkwatch patch (see PARKWATCH_SECONDS_FILE) to hold
them alive for a bounded window. Without that patch this daemon still runs, but
it will never see a burst and will report the park as unlocked -- which is why
the notifier treats a missing "locked" as an alert, not as silence.

This daemon only reports state. The 15-minute timer and the actual notification
live on the Pi, which cannot be powered down by the car's battery logic.
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import deque

from openpilot.cereal import messaging
from openpilot.common.swaglog import cloudlog

CONFIG_FILE = "/data/parkwatch.json"
PARKWATCH_SECONDS_FILE = "/data/parkwatch_seconds"  # read by the pandad patch

DOOR_LOCKS_ADDR = 1592  # 0x638, Toyota body CAN, present on bus 0
CAR_BUS = 0

DEFAULTS = {
  "ntfy_server": "https://ntfy.sh",
  "control_topic": "",
  "window_s": 1200,        # how long to watch after ignition-off
  "publish_timeout_s": 10,
  "retry_interval_s": 15,
}


def load_config():
  cfg = dict(DEFAULTS)
  try:
    with open(CONFIG_FILE) as f:
      cfg.update(json.load(f))
  except FileNotFoundError:
    return None
  except Exception:
    cloudlog.exception("parkwatchd: bad config, using defaults")
  if not cfg.get("control_topic"):
    return None
  return cfg


def decode_door_locks(dat):
  """DOOR_LOCKS (1592) per opendbc toyota _toyota_2017.dbc.

  DBC big-endian ("Motorola") start bits: LOCK_STATUS_CHANGED 15|1,
  LOCK_STATUS 20|1 (0 = locked, 1 = unlocked), LOCKED_VIA_KEYFOB 23|1.
  """
  if len(dat) < 8:
    return None

  def sig(start, length):
    val, pos = 0, start
    for _ in range(length):
      byte, bit = pos // 8, pos % 8
      val = (val << 1) | ((dat[byte] >> bit) & 1)
      pos = pos + 15 if bit == 0 else pos - 1
    return val

  return {
    "changed": sig(15, 1),
    "unlocked": bool(sig(20, 1)),
    "via_keyfob": sig(23, 1),
  }


class Publisher:
  """Fire-and-forget ntfy publishing with a retry queue.

  Parking spots lose signal; an event that cannot go out now is retried until
  the watch window ends. Runs on its own thread so a slow POST never stalls the
  CAN read loop (a missed burst is a missed lock).
  """

  def __init__(self, cfg):
    self.cfg = cfg
    self.url = f"{cfg['ntfy_server'].rstrip('/')}/{cfg['control_topic']}"
    self.q = deque(maxlen=64)
    self.lock = threading.Lock()
    self.thread = threading.Thread(target=self._run, daemon=True)
    self.thread.start()

  def send(self, event):
    with self.lock:
      self.q.append(event)

  def _post(self, event):
    body = json.dumps(event).encode()
    req = urllib.request.Request(
      self.url, data=body, method="POST",
      headers={"Content-Type": "application/json", "X-Title": event.get("type", "parkwatch")},
    )
    with urllib.request.urlopen(req, timeout=self.cfg["publish_timeout_s"]) as r:
      return 200 <= r.status < 300

  def _run(self):
    while True:
      event = None
      with self.lock:
        if self.q:
          event = self.q[0]
      if event is None:
        time.sleep(1.0)
        continue
      try:
        ok = self._post(event)
      except (urllib.error.URLError, OSError, TimeoutError) as e:
        cloudlog.debug(f"parkwatchd: publish deferred ({e})")
        ok = False
      except Exception:
        cloudlog.exception("parkwatchd: publish failed hard")
        ok = False
      if ok:
        with self.lock:
          if self.q and self.q[0] is event:
            self.q.popleft()
        cloudlog.event("parkwatchd published", event_type=event.get("type"), session=event.get("session"))
      else:
        time.sleep(self.cfg["retry_interval_s"])


def main():
  cfg = load_config()
  if cfg is None:
    # Not configured: stay alive but idle, so the manager does not respawn-loop.
    cloudlog.info("parkwatchd: no config, idling")
    while True:
      time.sleep(60)

  # single source of truth for the window: hand it to the pandad patch too
  try:
    with open(PARKWATCH_SECONDS_FILE, "w") as f:
      f.write(str(int(cfg["window_s"])))
  except OSError:
    cloudlog.exception("parkwatchd: could not write parkwatch_seconds")

  pub = Publisher(cfg)
  sm = messaging.SubMaster(["pandaStates"])

  can_sock = None          # created only while parked -- onroad CAN is high rate
  ignition = None
  session = None
  window_ends = 0.0
  unlocked = True          # Toyota auto-unlocks on shift-to-Park
  seq = 0

  def publish(event_type, **extra):
    nonlocal seq
    seq += 1
    pub.send({
      "type": event_type, "session": session, "seq": seq,
      "ts": time.time(), "window_s": int(cfg["window_s"]), **extra,
    })

  cloudlog.info(f"parkwatchd: started, window={cfg['window_s']}s")

  while True:
    sm.update(100)

    # Read the last known ignition rather than requiring an update this cycle:
    # draining CAN must never be gated on pandaStates ticking, because a missed
    # wake burst is a missed lock.
    if len(sm["pandaStates"]) > 0:
      ps = sm["pandaStates"][0]
      ign = bool(ps.ignitionLine) or bool(ps.ignitionCan)

      if ign != ignition:
        if ignition is not None and not ign:
          # ---- ignition just dropped: a park begins ----
          session = f"{int(time.time())}"
          window_ends = time.time() + cfg["window_s"]
          unlocked = True   # auto-unlock on Park; corrected by any wake burst
          if can_sock is None:
            can_sock = messaging.sub_sock("can", conflate=False, timeout=20)
          publish("park", unlocked=True)
          cloudlog.event("parkwatchd park", session=session)
        elif ign:
          # ---- ignition returned: the drive owns the car now ----
          if session is not None:
            publish("ignition_on")
          session = None
          can_sock = None
        ignition = ign

    if session is None or can_sock is None:
      continue

    # ---- parked: drain CAN looking for wake bursts ----
    for msg in messaging.drain_sock(can_sock):
      for c in msg.can:
        if c.address != DOOR_LOCKS_ADDR or c.src != CAR_BUS:
          continue
        d = decode_door_locks(bytes(c.dat))
        if d is not None and d["unlocked"] != unlocked:
          unlocked = d["unlocked"]
          publish("unlocked" if unlocked else "locked", via_keyfob=d["via_keyfob"])
          cloudlog.event("parkwatchd lock change", unlocked=unlocked, session=session)

    if time.time() > window_ends:
      publish("window_end", unlocked=unlocked)
      cloudlog.event("parkwatchd window end", unlocked=unlocked, session=session)
      session = None
      can_sock = None


if __name__ == "__main__":
  try:
    main()
  except Exception:
    cloudlog.exception("parkwatchd: fatal")
    # never take the manager down with us
    while True:
      time.sleep(60)
