#!/usr/bin/env python3
"""Chestnut GPU rail control (VBSM_GPU_IDLE).

The enclosure's F3 vendor request gates the GPU power rails; hardware-validated
on firmware ed4e39b7 with the 12V input live: F3=0 collapses draw from ~2A to
~1mA (parked battery-drain killer), F3=1 restores power and the PCIe link
retrains to L0 first try. The bridge itself stays enumerated on USB-C power
throughout, so detection and this CLI keep working with the rails down.
Upstream pinned firmware 8516b768 on 2026-10-18: F3=1 is the same request as
upstream's own flash.link_up(), but F3=0 (the cut) is unverified on it.

Only a bridge on a custom runtime firmware is ever touched: never the ROM
bootloader (hardwared's flasher may be driving it) and never an unknown
firmware. hardwared already holds its rail commands while the firmware is out
of date or being flashed; this re-checks at the moment of the command.

CAUTION from field experience: with the 12V input dead (accessory outlet off),
every command "fails" in confusing ways — check `status` first; supply readings
in the double-digit mV range mean there is no input power to switch.

Usage (root, matches the flash.py sudo pattern):  chestnut_power.py on|off|status
Exit: 0 ok, 1 failed, 2 not on the bus (status), 3 refused (not on a runtime firmware).
"""
import ctypes
import fcntl
import struct
import sys
import time

sys.path.insert(0, "/data/openpilot")
from openpilot.system.hardware.chestnut.flash import (find_chestnut, open_device, Ctrl,  # noqa: E402
                                                      USBDEVFS_CONTROL, link_up, in_rom_bootloader)


def _runtime_bridge():
  """VBSM_GPU_IDLE: (path, product) of the bridge while it runs a custom runtime firmware, else (None, product)."""
  path, vid_pid, product = find_chestnut()
  if not path or in_rom_bootloader(vid_pid, product) or not (product or "").startswith("custom "):
    return None, product
  return path, product


def _reads():
  path, _ = _runtime_bridge()
  if not path:
    return None
  fd = open_device(path)
  try:
    buf = (ctypes.c_ubyte * 5)()
    fcntl.ioctl(fd, USBDEVFS_CONTROL, Ctrl(0xC0, 0xC0, 0, 0, 5, 2000, ctypes.cast(buf, ctypes.c_void_p)))
    lt = (ctypes.c_ubyte * 1)()
    fcntl.ioctl(fd, USBDEVFS_CONTROL, Ctrl(0xC0, 0xE4, 0xB450, 0, 1, 2000, ctypes.cast(lt, ctypes.c_void_p)))
    v, i = struct.unpack("<Hh", bytes(buf)[:4])
    return v, i, lt[0]
  finally:
    import os
    os.close(fd)


def rails_off() -> bool:
  path, _ = _runtime_bridge()
  if not path:
    return False
  fd = open_device(path)
  try:
    fcntl.ioctl(fd, USBDEVFS_CONTROL, Ctrl(0x40, 0xF3, 0, 0, 0, 2000, None))
    return True
  except OSError:
    return False
  finally:
    import os
    os.close(fd)


def rails_on(retries: int = 3) -> bool:
  for n in range(retries):
    if _runtime_bridge()[0] is None:   # link_up() itself would talk to any chestnut id, ROM included
      return False
    if link_up():  # F3=1 + LTSSM must read L0
      return True
    time.sleep(2)
  return False


def main() -> int:
  cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
  if cmd in ("on", "off"):
    path, product = _runtime_bridge()
    if path is None:
      print(f"refused: chestnut not on a custom runtime firmware (product {product!r})")
      return 3
  if cmd == "off":
    ok = rails_off()
    print(f"rails_off: {ok}")
    return 0 if ok else 1
  if cmd == "on":
    ok = rails_on()
    print(f"rails_on: {ok}")
    return 0 if ok else 1
  r = _reads()
  if r is None:
    print(f"chestnut not on the bus on a custom runtime firmware (product {_runtime_bridge()[1]!r})")
    return 2
  print(f"supply={r[0]}mV/{r[1]}mA ltssm=0x{r[2]:02x}" + (" (L0)" if r[2] == 0x78 else ""))
  return 0


if __name__ == "__main__":
  sys.exit(main())
