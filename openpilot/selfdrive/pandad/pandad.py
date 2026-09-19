#!/usr/bin/env python3
# simple pandad wrapper that updates the panda first
import os
import usb1
import time
import signal
import subprocess

from panda import Panda, PandaDFU, PandaProtocolMismatch, McuType, FW_PATH
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.hardware import HARDWARE
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.selfdrive.pandad.rivian_long_flasher import flash_rivian_long


def get_expected_signature() -> bytes:
  fn = os.path.join(FW_PATH, McuType.H7.config.app_fn)
  return Panda.get_signature_from_firmware(fn)

def flash_panda(panda_serial: str):
  panda = Panda(panda_serial)

  # skip flashing if the detected panda is not supported
  if panda.get_type() not in Panda.SUPPORTED_DEVICES:
    cloudlog.warning(f"Panda {panda_serial} is not supported (hw_type: {panda.get_type()}), skipping flash...")
    panda.close()
    return

  fw_signature = get_expected_signature()
  internal_panda = panda.is_internal()

  panda_version = "bootstub" if panda.bootstub else panda.get_version()
  panda_signature = b"" if panda.bootstub else panda.get_signature()
  cloudlog.warning(f"Panda {panda_serial} connected, version: {panda_version}, signature {panda_signature.hex()[:16]}, expected {fw_signature.hex()[:16]}")

  if panda.bootstub or panda_signature != fw_signature:
    cloudlog.info("Panda firmware out of date, update required")
    panda.flash()
    cloudlog.info("Done flashing")

  if panda.bootstub:
    bootstub_version = panda.get_version()
    cloudlog.info(f"Flashed firmware not booting, flashing development bootloader. {bootstub_version=}, {internal_panda=}")
    if internal_panda:
      HARDWARE.recover_internal_panda()
    panda.recover(reset=(not internal_panda))
    cloudlog.info("Done flashing bootstub")

  if panda.bootstub:
    cloudlog.info("Panda still not booting, exiting")
    raise AssertionError

  panda_signature = panda.get_signature()
  if panda_signature != fw_signature:
    cloudlog.info("Version mismatch after flashing, exiting")
    raise AssertionError

  panda.close()


def check_panda_support(panda_serials: list[str]) -> list[str]:
  spi_serials = set(Panda.spi_list())
  for serial in panda_serials:
    if serial in spi_serials:
      return [serial]

  for serial in panda_serials:
    panda = Panda(serial)
    is_internal = panda.is_internal()
    panda.close()
    if is_internal:
      return [serial]

  return []


# ---------------------------------------------------------------- parkwatch
# VBSM_PARKWATCH: the C++ pandad puts the panda into power save the instant
# ignition goes away, and power save calls enable_can_transceivers(false). That
# leaves the device physically deaf to the ~5s body-CAN wake bursts a parked
# Toyota emits when it is locked or unlocked -- measured on this car 2026-08-30:
# the bus is silent between events, wakes for 5.0-5.8s on each lock/unlock/door,
# and every burst re-announces DOOR_LOCKS (0x638).
#
# pandad ships as a committed binary and this device has no capnp headers, so
# the source cannot be patched into effect. Instead we stand in for it: on the
# ignition-off edge we stop the C++ child and run a small Python loop that holds
# the transceivers on and republishes `can`, handing the panda straight back the
# moment ignition returns.
#
# Deliberately publishes NO pandaStates. hardwared forces ignition False when
# pandaStates goes stale (see DISCONNECT_TIMEOUT in hardwared.py), so a silent
# window is the *safe* failure mode: the worst case is that the device stays
# offroad until the C++ pandad is back, a couple of seconds after ignition.
# Publishing a pandaStates ourselves would risk asserting a wrong ignition for
# the whole window, which is the one outcome that actually matters.

PARKWATCH_CONF = "/data/parkwatch_seconds"
PARKWATCH_MAX_S = 2400

# set by pandad's signal handler so a manager shutdown cuts the window short
_PW = {"exit": False}
# reset_internal_panda() returns 10 ms after releasing reset and the app needs ~1 s to
# come up; poll this long for it before the relaunch (VBSM_PARKWATCH handback)
PARKWATCH_ENUM_S = 6.0
# a window that died mid-park is retried once after a hard reset if this much is left
PARKWATCH_RETRY_MIN_S = 60


def _parkwatch_window_s() -> int:
  try:
    with open(PARKWATCH_CONF) as f:
      return max(0, min(int(f.read().strip() or 0), PARKWATCH_MAX_S))
  except (OSError, ValueError):
    return 0


def _parkwatch_run(window_s: int, serial: str) -> str:
  """Own the panda while parked, republishing `can` so parkwatchd sees bursts.

  Returns why the window ended: ignition / timeout / exit / noclaim / failed.
  """
  p = None
  failed = False
  safety_silent = None
  t_start = time.monotonic()
  t_end = t_start + window_s
  try:
    # inside the try so a setup failure hands back like any other, instead of
    # escaping to main()'s except and the recovery ladder
    from openpilot.cereal import messaging
    from opendbc.car.structs import CarParams
    safety_silent = CarParams.SafetyModel.silent
    pm = messaging.PubMaster(["can"])
    for _ in range(10):
      try:
        # disable_checks=True (the default) already clears power save and
        # disables the panda's heartbeat timeout; the known serial keeps the
        # bare constructor from probing the DFU bootloader on the same bus
        p = Panda(serial)
        break
      except Exception:
        time.sleep(1)
    if p is None:
      cloudlog.error("parkwatch: could not claim panda, handing back")
      return "noclaim"

    p.set_heartbeat_disabled()
    p.set_power_save(0)
    # noOutput keeps the harness relay closed -- the same electrical state the
    # car sits in while parked normally, stock camera connected
    p.set_safety_mode(CarParams.SafetyModel.noOutput)
    cloudlog.event("parkwatch window start", window_s=window_s)

    last_health = 0.0
    while time.monotonic() < t_end and not _PW["exit"]:
      now = time.monotonic()
      if now - last_health > 0.2:
        last_health = now
        h = p.health()
        if h["ignition_line"] or h["ignition_can"]:
          cloudlog.event("parkwatch window end", reason="ignition")
          return "ignition"
        if h["power_save_enabled"]:
          p.set_power_save(0)

      msgs = p.can_recv()
      if msgs:
        evt = messaging.new_message("can", len(msgs))
        evt.valid = True
        for i, m in enumerate(msgs):
          addr, dat, src = m[0], m[-2], m[-1]
          evt.can[i].address = addr
          evt.can[i].dat = bytes(dat)
          evt.can[i].src = src
        pm.send("can", evt)
      else:
        time.sleep(0.02)
    reason = "exit" if _PW["exit"] else "timeout"
    cloudlog.event("parkwatch window end", reason=reason)
    return reason
  except Exception:
    failed = True
    cloudlog.exception(f"parkwatch: window failed after {time.monotonic() - t_start:.1f} s, handing panda back")
    return "failed"
  finally:
    if p is not None:
      # a link that just NACKed will NACK these writes too (15 s of retries for
      # a second traceback); the relaunch resets the panda anyway. Leave it in
      # SILENT + power save = stock's parked state: on the manager-exit path no
      # relaunch follows, and a panda still in noOutput when the SoM powers off
      # trips the firmware's low-power assert and idles at full clock all park.
      if not failed:
        try:
          if safety_silent is not None:
            p.set_safety_mode(safety_silent)
          p.set_power_save(1)
        except Exception:
          cloudlog.warning("parkwatch: could not park the panda; the relaunch reset will")
      try:
        p.close()
      except Exception:
        cloudlog.warning("parkwatch: panda close failed")


def _wait_for_panda(timeout_s: float) -> bool:
  """After a reset, wait for the app to enumerate (Panda.list() is empty while it boots)."""
  t_end = time.monotonic() + timeout_s
  while time.monotonic() < t_end and not _PW["exit"]:
    try:
      if Panda.list():
        return True
    except Exception:
      pass
    time.sleep(0.25)
  return False


def _supervise(process, serial: str) -> bool:
  """Wait for the C++ pandad, but take the panda over on the ignition-off edge.

  Requires an observed True->False transition, so relaunching while already
  parked (which is exactly what happens when a window ends) cannot re-trigger:
  one window per drive. Returns True when it returns because a window ran (the
  panda is healthy and only needs a plain reset), False when pandad exited.
  """
  from openpilot.cereal import messaging

  window_s = _parkwatch_window_s()
  if window_s <= 0:
    process.wait()
    return False

  sm = messaging.SubMaster(["pandaStates"])
  seen_ignition = False
  while True:
    if process.poll() is not None:
      return False
    sm.update(500)
    if not sm.updated["pandaStates"] or len(sm["pandaStates"]) == 0:
      continue

    ps = sm["pandaStates"][0]
    if bool(ps.ignitionLine) or bool(ps.ignitionCan):
      seen_ignition = True
    elif seen_ignition:
      # re-read at the edge: the window can be changed (or the feature turned
      # off) without waiting for a pandad restart
      window_s = _parkwatch_window_s()
      if window_s <= 0:
        process.wait()
        return False
      cloudlog.event("parkwatch handoff", window_s=window_s)
      process.send_signal(signal.SIGINT)
      try:
        process.wait(timeout=10)
      except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
      t_end = time.monotonic() + window_s
      if _parkwatch_run(window_s, serial) == "failed" and not _PW["exit"]:
        # one retry after a hard reset so a mid-park SPI fault does not leave
        # the rest of the park unmonitored; the claim loop covers the app boot
        remaining = int(t_end - time.monotonic())
        if remaining >= PARKWATCH_RETRY_MIN_S:
          cloudlog.event("parkwatch retry", remaining_s=remaining)
          try:
            HARDWARE.reset_internal_panda()
          except Exception:
            cloudlog.exception("parkwatch: retry reset failed")
          else:
            _parkwatch_run(remaining, serial)
      return True  # outer loop relaunches the real pandad


def main() -> None:
  # signal pandad to close the relay and exit
  def signal_handler(signum, frame):
    cloudlog.info(f"Caught signal {signum}, exiting")
    nonlocal do_exit
    do_exit = True
    _PW["exit"] = True
    if process is not None:
      process.send_signal(signal.SIGINT)

  process = None
  do_exit = False
  signal.signal(signal.SIGINT, signal_handler)

  # check health for lost heartbeat
  try:
    for s in Panda.list():
      with Panda(s) as p:
        health = p.health()
        if p.is_internal() and health["heartbeat_lost"]:
          Params().put_bool("PandaHeartbeatLost", True, block=True)
          cloudlog.event("heartbeat lost", deviceState=health)
  except Exception:
    cloudlog.exception("pandad.uncaught_exception")

  count = 0
  handback = False  # VBSM_PARKWATCH: _supervise() returned because a window ran
  while not do_exit:
    try:
      cloudlog.event("pandad.flash_and_connect", count=count, handback=handback)
      if handback:
        # The window hands back a healthy panda. Reset it the way a fresh boot
        # does (re-inits the SPI slave, wipes the window's can-speed/power-save
        # writes; the C++ pandad's first heartbeat re-arms the heartbeat check)
        # and wait for the app. Stock's even/odd ladder below is for a crashed
        # pandad: taken here it DFU-erased and reflashed the panda after every
        # park (count was already odd, and on the even laps Panda.list() came
        # back empty 10 ms after the reset pulse, so the next lap recovered).
        handback = False
        HARDWARE.reset_internal_panda()
        if not _wait_for_panda(PARKWATCH_ENUM_S):
          cloudlog.warning("parkwatch: panda did not enumerate after the handback reset, falling back to the recovery ladder")
      elif (count % 2) == 0:
        HARDWARE.reset_internal_panda()
        count += 1
      else:
        HARDWARE.recover_internal_panda()
        count += 1

      # Flash all Pandas in DFU mode
      for serial in PandaDFU.list():
        cloudlog.info(f"Panda in DFU mode found, flashing recovery {serial}")
        PandaDFU(serial).recover()
        time.sleep(1)

      panda_serials = Panda.list()
      if len(panda_serials):
        # custom flasher for xnor's Rivian Longitudinal Upgrade Kit
        flash_rivian_long(panda_serials)
        # find the internal supported panda (e.g. skip external Black Panda)
        panda_serials = check_panda_support(panda_serials)

        assert len(panda_serials) == 1
        cloudlog.info(f"{len(panda_serials)} panda found, connecting - {panda_serials}")
        flash_panda(panda_serials[0])

        if do_exit:
          break  # a manager exit landed in this lap: do not launch a child nobody will signal

        # run real pandad
        os.environ['MANAGER_DAEMON'] = 'pandad'
        process = subprocess.Popen(["./pandad"], cwd=os.path.join(BASEDIR, "openpilot/selfdrive/pandad"))
        handback = _supervise(process, panda_serials[0])
    # TODO: wrap all panda exceptions in a base panda exception
    except (usb1.USBErrorNoDevice, usb1.USBErrorPipe):
      # a panda was disconnected while setting everything up. let's try again
      cloudlog.exception("Panda USB exception while setting up")
    except PandaProtocolMismatch:
      cloudlog.exception("pandad.protocol_mismatch")
    except Exception:
      cloudlog.exception("pandad.uncaught_exception")


if __name__ == "__main__":
  main()
