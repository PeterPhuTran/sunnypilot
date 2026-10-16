import os
import time
import threading

from openpilot.common.params import Params
from openpilot.common.hardware import HARDWARE
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.system.statsd import statlog

CAR_VOLTAGE_LOW_PASS_K = 0.011 # LPF gain for 45s tau (dt/tau / (dt/tau + 1))

# While driving, a battery charges completely in about 30-60 minutes
CAR_BATTERY_CAPACITY_uWh = 30e6
CAR_CHARGING_RATE_W = 45

# VBSM_PARK: the parked energy allowance is the only lever that changes how
# much the device takes from the 12V battery per park (a weak hybrid AGM that
# rests at ~55% SoC; simulated ~56 Wh/day at the stock 30 Wh). Override in
# whole Wh via /data/vbsm_park_budget_wh (clamped 5..30; absent = stock 30).
PARK_BUDGET_FILE = "/data/vbsm_park_budget_wh"
PARK_BUDGET_MIN_WH = 5
PARK_BUDGET_MAX_WH = 30
PARK_SHUTDOWN_LOG = "/data/vbsm_shutdowns.jsonl"
# VBSM_PARK: the comma four has no /sys/class/hwmon/hwmon1/power1_input, so
# HARDWARE.get_current_power_draw() reads 0 W there and the budget never
# integrated (field, 2026-09-08: 9.1 h parked, used_wh 0.0, shutdown on the
# voltage rule). Fall back to the SoM battery-management reading, then to a
# fixed floor. The BMS sees the SoM only: measured at the panda the whole
# device draws 4.1-4.8 W parked (5-6 W with diagnostics) while the BMS reads
# 2.6-3.2 W, so integrating the bare reading spent ~17 Wh per "10 Wh" park
# (2026-09-19 review: four 3.7 h parks ran the balance to the boot floor).
# Scale the BMS reading by the measured ratio and never go below the measured
# whole-device idle.
PARK_SOM_TO_DEVICE = 1.6
PARK_DRAW_FLOOR_W = 4.5
# VBSM_PARK: how much a boot that never sees ignition may spend before it
# powers off. On a comma four the panda enters stop mode once the SoM is off
# (power save on, SAFETY_SILENT) and any CAN frame or SBU edge wakes it by
# resetting it; a freshly reset panda boot-kicks the SoM (panda/board/main.c,
# cuatro branch of the main loop; board/sys/power_saving.h). So a parked car
# that is locked, unlocked or polled by its own ECUs boots the device with no
# ignition. Four such wakes with records on 2026-09-20/21 ran 12-15 min and
# 1.0-1.1 Wh each, none followed by an ignition inside that window, and a wake
# that inherits a saved balance from a voltage-ended park is not bounded by the
# boot floor: it runs until that balance is spent or the 11.8 V rule fires. This
# caps the spend, not the balance: the saved balance stays for the next drive. 0.4 Wh at the 4.5 W floor is ~320 s of parked
# time, enough for the updater's post-boot cycle (observed ~85 s from boot to
# 'git reset success') and for deploy.sh, which waits for the finalized overlay.
PARK_WAKE_SPEND_WH = 0.4
# VBSM_PARK: while the home Pi is actively pulling footage it touches this
# tmpfs marker (once per batch). A fresh marker suspends the budget and timer
# rules so a park never ends mid-sync; the 11.8 V rule and ForcePowerDown are
# untouched -- the battery floor is not negotiable. tmpfs clears on reboot and
# the TTL bounds a Pi that vanished mid-sync (wall-clock mtime: a forward NTP
# jump can only expire it early, which fails safe).
SYNC_ACTIVE_FILE = "/dev/shm/vbsm_sync_active"
# VBSM_PARK: touched the first time this boot sees ignition; tmpfs, so a reboot
# clears it but a hardwared respawn does not (started_seen is process-local).
BOOT_IGNITION_FILE = "/dev/shm/vbsm_boot_ignition"
SYNC_ACTIVE_TTL_S = 30 * 60


def sync_active() -> bool:
  try:
    return (time.time() - os.path.getmtime(SYNC_ACTIVE_FILE)) < SYNC_ACTIVE_TTL_S
  except OSError:
    return False


def park_budget_uWh() -> float:
  try:
    with open(PARK_BUDGET_FILE) as f:
      wh = int(f.read().strip())
  except (OSError, ValueError):
    return CAR_BATTERY_CAPACITY_uWh
  return float(max(PARK_BUDGET_MIN_WH, min(PARK_BUDGET_MAX_WH, wh))) * 1e6


def park_power_draw() -> tuple[float, str, float]:
  """VBSM_PARK: (watts integrated, source, raw sensor watts) -- the platform
  sensor when it reads, else the SoM BMS reading scaled to the whole device,
  else PARK_DRAW_FLOOR_W. The raw reading is kept for the shutdown record so
  the calibration stays checkable against the sensor. Never below zero."""
  try:
    p = float(HARDWARE.get_current_power_draw() or 0.0)
  except Exception:
    p = 0.0
  if p > 0:
    return p, "hwmon", p
  try:
    som = float(HARDWARE.get_som_power_draw() or 0.0)
  except Exception:
    som = 0.0
  if som > 0:
    return max(som * PARK_SOM_TO_DEVICE, PARK_DRAW_FLOOR_W), "bms", som
  return PARK_DRAW_FLOOR_W, "floor", 0.0

VBATT_PAUSE_CHARGING = 11.8           # Lower limit on the LPF car battery voltage
MAX_TIME_OFFROAD_S = 30*3600
# VBSM_PARK: upstream keeps a fresh device awake for an hour after boot so a new
# install is not powered off before it is registered. Here the only offroad
# boots are updater/deploy reboots while parked, and the hour blocked every
# park rule each time (2026-09-16: 82 min awake at ~4.5 W on a battery that
# then rested at 11.89 V). 600 s covered the manager and the updater's
# post-boot cycle; 300 s still does. In practice DELAY_SHUTDOWN_TIME_S below
# (300 s after hardwared's first offroad tick, ~20-25 s into boot) is the gate
# that binds, so a no-ignition boot ends ~320-345 s after boot (2026-09-21).
MIN_ON_TIME_S = 300
DELAY_SHUTDOWN_TIME_S = 300 # Wait at least DELAY_SHUTDOWN_TIME_S seconds after offroad_time to shutdown.
VOLTAGE_SHUTDOWN_MIN_OFFROAD_TIME_S = 60

class PowerMonitoring:
  def __init__(self):
    self.params = Params()
    self.last_measurement_time = None           # Used for integration delta
    self.last_save_time = 0                     # Used for saving current value in a param
    self.power_used_uWh = 0                     # Integrated power usage in uWh since going into offroad
    self.next_pulsed_measurement_time = None
    self.car_voltage_mV = 12e3                  # Low-passed version of peripheralState voltage
    self.car_voltage_instant_mV = 12e3          # Last value of peripheralState voltage
    self.integration_lock = threading.Lock()

    car_battery_capacity_uWh = self.params.get("CarBatteryCapacity") or 0

    # VBSM_PARK: the allowance and the boot floor scale with the knob; the
    # cap in calculate() brings a saved balance above the new budget down.
    self.budget_uWh = park_budget_uWh()
    self.last_eval: dict = {}
    self.draw_w = 0.0
    self.draw_raw_w = 0.0
    self.draw_source = "none"
    self._draw_source_logged: str | None = None
    self.boot_ignition = os.path.exists(BOOT_IGNITION_FILE)  # VBSM_PARK: survives a respawn, not a reboot
    if self.budget_uWh != CAR_BATTERY_CAPACITY_uWh:
      cloudlog.event("vbsm park budget", wh=self.budget_uWh / 1e6, stock_wh=CAR_BATTERY_CAPACITY_uWh / 1e6)

    # Reset capacity if it's low
    self.car_battery_capacity_uWh = max((self.budget_uWh / 10), car_battery_capacity_uWh)

  # Calculation tick
  def calculate(self, voltage: float | None, ignition: bool):
    try:
      now = time.monotonic()
      if ignition and not self.boot_ignition:  # VBSM_PARK: this boot has seen ignition, the wake cap no longer applies
        self.boot_ignition = True
        try:
          open(BOOT_IGNITION_FILE, "a").close()
        except OSError:
          pass

      # If peripheralState is None, we're probably not in a car, so we don't care
      if voltage is None:
        with self.integration_lock:
          self.last_measurement_time = None
          self.next_pulsed_measurement_time = None
          self.power_used_uWh = 0
        return

      # Low-pass battery voltage
      self.car_voltage_instant_mV = voltage
      self.car_voltage_mV = ((voltage * CAR_VOLTAGE_LOW_PASS_K) + (self.car_voltage_mV * (1 - CAR_VOLTAGE_LOW_PASS_K)))
      statlog.gauge("car_voltage", self.car_voltage_mV / 1e3)

      # Cap the car battery power and save it in a param every 10-ish seconds
      self.car_battery_capacity_uWh = max(self.car_battery_capacity_uWh, 0)
      self.car_battery_capacity_uWh = min(self.car_battery_capacity_uWh, self.budget_uWh)
      if now - self.last_save_time >= 10:
        self.params.put("CarBatteryCapacity", int(self.car_battery_capacity_uWh))
        self.last_save_time = now

      # First measurement, set integration time
      with self.integration_lock:
        if self.last_measurement_time is None:
          self.last_measurement_time = now
          return

      if ignition:
        # If there is ignition, we integrate the charging rate of the car
        with self.integration_lock:
          self.power_used_uWh = 0
          integration_time_h = (now - self.last_measurement_time) / 3600
          if integration_time_h < 0:
            raise ValueError(f"Negative integration time: {integration_time_h}h")
          self.car_battery_capacity_uWh += (CAR_CHARGING_RATE_W * 1e6 * integration_time_h)
          self.last_measurement_time = now
      else:
        # Get current power draw somehow
        current_power, self.draw_source, self.draw_raw_w = park_power_draw()
        self.draw_w = current_power
        if self.draw_source != self._draw_source_logged:
          self._draw_source_logged = self.draw_source
          cloudlog.event("vbsm park draw source", source=self.draw_source, watts=round(current_power, 2))

        # Do the integration
        self._perform_integration(now, current_power)
    except Exception:
      cloudlog.exception("Power monitoring calculation failed")

  def _perform_integration(self, t: float, current_power: float) -> None:
    with self.integration_lock:
      try:
        if self.last_measurement_time:
          integration_time_h = (t - self.last_measurement_time) / 3600
          power_used = (current_power * 1000000) * integration_time_h
          if power_used < 0:
            raise ValueError(f"Negative power used! Integration time: {integration_time_h} h Current Power: {power_used} uWh")
          self.power_used_uWh += power_used
          self.car_battery_capacity_uWh -= power_used
          self.last_measurement_time = t
      except Exception:
        cloudlog.exception("Integration failed")

  # Get the power usage
  def get_power_used(self) -> int:
    return int(self.power_used_uWh)

  def get_car_battery_capacity(self) -> int:
    return int(self.car_battery_capacity_uWh)

  # Max Time Offroad
  def max_time_offroad_exceeded(self, offroad_time):
    param = self.params.get("MaxTimeOffroad")  # minutes, 0 = no limit
    if param is not None and param >= 0:
      return 0 < param * 60 <= offroad_time
    return offroad_time > MAX_TIME_OFFROAD_S

  # See if we need to shutdown
  def should_shutdown(self, ignition: bool, in_car: bool, offroad_timestamp: float | None, started_seen: bool):
    if offroad_timestamp is None:
      return False

    now = time.monotonic()
    should_shutdown = False
    offroad_time = (now - offroad_timestamp)
    low_voltage_shutdown = (self.car_voltage_mV < (VBATT_PAUSE_CHARGING * 1e3) and
                            offroad_time > VOLTAGE_SHUTDOWN_MIN_OFFROAD_TIME_S)
    # VBSM_PARK: the same decision as upstream, evaluated term by term so the
    # reason survives to the shutdown record -- until now a park that ended
    # early could not be told apart as timer, voltage or budget.
    sync = sync_active()  # VBSM_PARK: footage pull in progress -> budget/timer suspended, voltage kept
    timer = self.max_time_offroad_exceeded(offroad_time) and not sync
    budget = self.car_battery_capacity_uWh <= 0 and not sync
    # VBSM_PARK: a boot that has seen neither ignition (tmpfs marker, survives a
    # hardwared respawn) nor an onroad start (started_seen) -- a CAN/SBU wake, an
    # updater or deploy reboot -- may spend PARK_WAKE_SPEND_WH and no more,
    # whatever balance it inherited; the sync marker suspends it like the budget.
    wake = (not started_seen) and (not self.boot_ignition) and self.power_used_uWh >= PARK_WAKE_SPEND_WH * 1e6 and not sync
    disable_power_down = self.params.get_bool("DisablePowerDown")
    force = self.params.get_bool("ForcePowerDown")
    min_on_ok = started_seen or (now > MIN_ON_TIME_S)
    should_shutdown |= timer
    should_shutdown |= low_voltage_shutdown
    should_shutdown |= budget
    should_shutdown |= wake
    should_shutdown &= not ignition
    should_shutdown &= (not disable_power_down)
    should_shutdown &= in_car
    should_shutdown &= offroad_time > DELAY_SHUTDOWN_TIME_S
    should_shutdown |= force
    should_shutdown &= min_on_ok
    reasons = [name for name, hit in (("force", force), ("budget", budget), ("wake", wake), ("voltage", low_voltage_shutdown), ("timer", timer)) if hit]
    self.last_eval = {
      "decision": bool(should_shutdown), "reason": "+".join(reasons) if reasons else "none",
      "timer": bool(timer), "voltage": bool(low_voltage_shutdown), "budget": bool(budget), "wake": bool(wake), "boot_ignition": bool(self.boot_ignition), "force": bool(force), "sync_active": bool(sync),
      "ignition": bool(ignition), "in_car": bool(in_car), "disable_power_down": bool(disable_power_down),
      "delay_ok": bool(offroad_time > DELAY_SHUTDOWN_TIME_S), "min_on_ok": bool(min_on_ok), "started_seen": bool(started_seen),
      "offroad_s": round(offroad_time, 1), "monotonic_s": round(now, 1),
      "capacity_wh": round(self.car_battery_capacity_uWh / 1e6, 3), "used_wh": round(self.power_used_uWh / 1e6, 3),
      "budget_wh": round(self.budget_uWh / 1e6, 1), "draw_w": round(self.draw_w, 2), "draw_source": self.draw_source, "raw_w": round(self.draw_raw_w, 2),
      "lpf_mV": int(self.car_voltage_mV), "instant_mV": int(self.car_voltage_instant_mV),
    }
    return should_shutdown

  def shutdown_record(self) -> dict:
    """VBSM_PARK: the last should_shutdown() evaluation, for the persistent record."""
    return dict(self.last_eval)
