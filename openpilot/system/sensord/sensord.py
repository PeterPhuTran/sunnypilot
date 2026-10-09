#!/usr/bin/env python3
import os
import time
import ctypes
import select
import threading

import openpilot.cereal.messaging as messaging
from openpilot.cereal.services import SERVICE_LIST
from openpilot.common.utils import sudo_write
from openpilot.common.realtime import config_realtime_process, Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.common.gpio import gpiochip_get_ro_value_fd, gpioevent_data

from openpilot.system.sensord.sensors.i2c_sensor import Sensor
from openpilot.system.sensord.sensors.lsm6ds3_accel import LSM6DS3_Accel
from openpilot.system.sensord.sensors.lsm6ds3_gyro import LSM6DS3_Gyro
from openpilot.system.sensord.sensors.lsm6ds3_temp import LSM6DS3_Temp

I2C_BUS_IMU = 1

# VBSM_SENSORD: at a cold boot the manager can start sensord before udev has applied
# 99-gpio.rules (root:gpio 0660) to /dev/gpiochip0, and the open fails with EACCES
# (2026-10-08: sensord started at 15:05:02, the chown landed at 15:05:04.03). Wait for
# it; when udev is only seconds late this stays inside selfdrived's 10 s sensorDataInvalid.
GPIO_OPEN_WAIT_S = 30.
GPIO_OPEN_RETRY_S = 0.25

def open_irq_fd(event: threading.Event) -> int | None:
  t0 = time.monotonic()
  logged = False
  while True:
    try:
      fd = gpiochip_get_ro_value_fd("sensord", 0, 84)
    except (PermissionError, FileNotFoundError) as e:
      # both fail at os.open(), before any fd exists, so a retry leaks nothing
      if time.monotonic() - t0 >= GPIO_OPEN_WAIT_S:
        raise
      if not logged:
        cloudlog.error(f"sensord: gpiochip0 not ready ({e}), waiting up to {GPIO_OPEN_WAIT_S:.0f} s")
        logged = True
      if event.wait(GPIO_OPEN_RETRY_S):
        return None
      continue
    if logged:
      cloudlog.warning(f"sensord: gpiochip0 ready after {time.monotonic() - t0:.2f} s")
    return fd

def interrupt_loop(sensors: list[tuple[Sensor, str, bool]], event) -> None:
  pm = messaging.PubMaster([service for sensor, service, interrupt in sensors if interrupt])

  # NOTE: the gyro and accelerometer share an IRQ due to the comma three
  # routing only one GPIO from the LSM to the SOC, but comma 3X and four
  # have two. if we want better timestamps in the future, we can use both.

  # Requesting both edges as the data ready pulse from the lsm6ds sensor is
  # very short (75us) and is mostly detected as falling edge instead of rising.
  # So if it is detected as rising the following falling edge is skipped.
  fd = open_irq_fd(event)
  if fd is None:
    return

  # Configure IRQ affinity
  irq_path = "/proc/irq/336/smp_affinity_list"
  if not os.path.exists(irq_path):
    irq_path = "/proc/irq/335/smp_affinity_list"
  if os.path.exists(irq_path):
    sudo_write('1\n', irq_path)

  offset = time.time_ns() - time.monotonic_ns()

  poller = select.poll()
  poller.register(fd, select.POLLIN | select.POLLPRI)
  while not event.is_set():
    events = poller.poll(100)
    if not events:
      cloudlog.error("poll timed out")
      continue
    if not (events[0][1] & (select.POLLIN | select.POLLPRI)):
      cloudlog.error("no poll events set")
      continue

    dat = os.read(fd, ctypes.sizeof(gpioevent_data)*16)
    evd = gpioevent_data.from_buffer_copy(dat)

    cur_offset = time.time_ns() - time.monotonic_ns()
    if abs(cur_offset - offset) > 10 * 1e6:  # ms
      cloudlog.warning(f"time jumped: {cur_offset} {offset}")
      offset = cur_offset
      continue

    ts = evd.timestamp - cur_offset
    for sensor, service, interrupt in sensors:
      if interrupt:
        try:
          evt = sensor.get_event(ts)
          if not sensor.is_data_valid():
            continue
          msg = messaging.new_message(service, valid=True)
          setattr(msg, service, evt)
          pm.send(service, msg)
        except Sensor.DataNotReady:
          pass
        except Exception:
          cloudlog.exception(f"Error processing {service}")


def run_logged(target, *args) -> None:
  # VBSM_SENSORD: an uncaught thread exception only reaches stderr (the tmux pane), never swaglog
  try:
    target(*args)
  except Exception:
    cloudlog.exception(f"sensord: {target.__name__} died")
    raise


def polling_loop(sensor: Sensor, service: str, event: threading.Event) -> None:
  pm = messaging.PubMaster([service])
  rk = Ratekeeper(SERVICE_LIST[service].frequency, print_delay_threshold=None)
  while not event.is_set():
    try:
      evt = sensor.get_event()
      if not sensor.is_data_valid():
        continue
      msg = messaging.new_message(service, valid=True)
      setattr(msg, service, evt)
      pm.send(service, msg)
    except Exception:
      cloudlog.exception(f"Error in {service} polling loop")
    rk.keep_time()

def main() -> None:
  config_realtime_process([1, ], 1)

  sensors_cfg = [
    (LSM6DS3_Accel(I2C_BUS_IMU), "accelerometer", True),
    (LSM6DS3_Gyro(I2C_BUS_IMU), "gyroscope", True),
    (LSM6DS3_Temp(I2C_BUS_IMU), "temperatureSensor", False),
  ]

  # Reset sensors
  for sensor, _, _ in sensors_cfg:
    try:
      sensor.reset()
    except Exception:
      cloudlog.exception(f"Error initializing {sensor} sensor")

  # Initialize sensors
  exit_event = threading.Event()
  threads = [
    threading.Thread(target=run_logged, args=(interrupt_loop, sensors_cfg, exit_event), daemon=True)
  ]
  for sensor, service, interrupt in sensors_cfg:
    try:
      sensor.init()
      if not interrupt:
        # Start polling thread for sensors without interrupts
        threads.append(threading.Thread(
          target=run_logged,
          args=(polling_loop, sensor, service, exit_event),
          daemon=True
        ))
    except Exception:
      cloudlog.exception(f"Error initializing {service} sensor")

  died = False
  try:
    for t in threads:
      t.start()
    # VBSM_SENSORD: all(), not any(). With any(), a dead interrupt thread left the temperature
    # poller holding the process up: the manager saw sensord running while accelerometer and
    # gyroscope were silent for the whole drive. Exit instead and let VBSM_RESTART rebuild it.
    while all(t.is_alive() for t in threads):
      time.sleep(1)
    died = True
  except KeyboardInterrupt:
    pass
  finally:
    exit_event.set()
    for t in threads:
      if t.is_alive():
        t.join()

    for sensor, _, _ in sensors_cfg:
      try:
        sensor.shutdown()
      except Exception:
        cloudlog.exception("Error shutting down sensor")

  if died:
    cloudlog.error("sensord: a sensor thread died, exiting for a restart")
    raise SystemExit(1)

if __name__ == "__main__":
  main()
