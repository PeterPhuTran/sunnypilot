#!/usr/bin/env python3
"""Chestnut GPU rail control (VBSM_GPU_IDLE).

The enclosure's F3 vendor request gates the GPU power rails; hardware-validated
on firmware ed4e39b7 with the 12V input live: F3=0 collapses draw from ~2A to
~1mA (parked battery-drain killer), F3=1 restores power and the PCIe link
retrains to L0 first try. The bridge itself stays enumerated on USB-C power
throughout, so detection and this CLI keep working with the rails down.

CAUTION from field experience: with the 12V input dead (accessory outlet off),
every command "fails" in confusing ways — check `status` first; supply readings
in the double-digit mV range mean there is no input power to switch.

Usage (root, matches the flash.py sudo pattern):  chestnut_power.py on|off|status
"""
import ctypes
import fcntl
import struct
import sys
import time

sys.path.insert(0, "/data/openpilot")
from openpilot.system.hardware.chestnut.flash import (find_chestnut, open_device, Ctrl,  # noqa: E402
                                                      USBDEVFS_CONTROL, link_up)


def _reads():
  path, _, _ = find_chestnut()
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
  path, _, _ = find_chestnut()
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
    if link_up():  # F3=1 + LTSSM must read L0
      return True
    time.sleep(2)
  return False


def main() -> int:
  cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
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
    print("chestnut not on bus")
    return 2
  print(f"supply={r[0]}mV/{r[1]}mA ltssm=0x{r[2]:02x}" + (" (L0)" if r[2] == 0x78 else ""))
  return 0


if __name__ == "__main__":
  sys.exit(main())
