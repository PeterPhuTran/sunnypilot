#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

# VBSM_GPU_READY / VBSM_GPU_PPT: fork block, kept above upstream's own import lines so an upstream edit there
# merges without touching it. Nothing in it may import tinygrad, numpy or openpilot.
import math   # VBSM_GPU_COLDWAIT
import os
# VBSM_GPU_READY: tinygrad resolves Device.DEFAULT by probing every backend in order
# (AMD before QCOM) the first time anything asks for it, and modeld_v2's import chain
# asks at import time (compile_modeld.py declares `device: str = Device.DEFAULT` as a
# default argument). On a cold boot that probe opens the AMD device before the PCIe
# link is up, fails, and leaks tinygrad's lock fd and the libusb claim, after which
# this process can never open the eGPU (verified on-device 2026-09-14). Pinning DEV
# makes DEFAULT resolve without probing; every tensor in modeld_v2 names its device
# explicitly, and the eGPU is opened deliberately in load_big().
os.environ.setdefault("DEV", "QCOM")


def chestnut_ppt_limit_w() -> int:
  """VBSM_GPU_PPT: the eGPU package power cap in watts from /data/vbsm_gpu_ppt_w (default 80, clamped 40..220),
  or 0 when the file says 0 or less: then upstream's AM_POWER_LIMIT applies (stock modeld's setdefault, 100 W)."""
  try:
    with open("/data/vbsm_gpu_ppt_w") as f:
      w = int(f.read().strip())
  except (OSError, ValueError):
    w = 80
  return 0 if w <= 0 else max(40, min(220, w))


# VBSM_GPU_PPT: tinygrad applies AM_POWER_LIMIT inside every AMD boot (amdev.py, full and partial boot) and reads
# it once (getenv is cached). Stock modeld, imported below for ChestnutGpuState, only setdefault()s 100 W, so the
# value exported here wins and the eGPU never boots above the fork's cap. apply_ppt_cap() still writes the same
# limit and reads it back once the bridge is open (the "chestnut ppt limit" event).
if chestnut_ppt_limit_w():
  os.environ["AM_POWER_LIMIT"] = str(chestnut_ppt_limit_w())

from collections.abc import Callable
import gc
import os
os.environ['GMMU'] = '0'
from tinygrad.device import Device
import numpy as np
import threading
import time
from setproctitle import setproctitle

import openpilot.cereal.messaging as messaging
from openpilot.common.hardware import COMMA_HARDWARE
from openpilot.selfdrive.modeld.helpers import chestnut_present, wait_for_chestnut
from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.cereal.services import SERVICE_LIST
from openpilot.cereal.messaging import PubMaster, SubMaster
from openpilot.cereal.visionipc import VisionStreamType
from msgq.visionipc import VisionIpcClient, VisionBuf
from opendbc.car.car_helpers import get_demo_car_params
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.hardware.usb import cable_connected
from openpilot.common.hardware.usb import CHESTNUT_USB_PRODUCT, get_usb_state, is_chestnut_usb_id  # VBSM_GPU_READY
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, smooth_value
from openpilot.selfdrive.modeld.modeld import ChestnutGpuState

from openpilot.sunnypilot.modeld_v2.fill_model_msg import fill_model_msg, fill_pose_msg, PublishState, get_curvature_from_output
from openpilot.sunnypilot.modeld_v2.parse_model_outputs import Parser
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants, Plan
from openpilot.sunnypilot.modeld_v2.meta_helper import load_meta_constants
from openpilot.sunnypilot.modeld_v2.camera_offset_helper import CameraOffsetHelper
from openpilot.sunnypilot.livedelay.helpers import get_lat_delay
from openpilot.sunnypilot.modeld_v2.modeld_base import ModelStateBase
from openpilot.sunnypilot.modeld_v2.helpers import load_model
from openpilot.sunnypilot.modeld_v2.frame_fusion import fused_with_warp
from openpilot.sunnypilot.modeld_v2.model_adapters import get_model_adapter
from openpilot.sunnypilot.models.helpers import get_active_bundle
from openpilot.sunnypilot.selfdrive.controls.lib.relc import RoadEdgeLaneChangeController

PROCESS_NAME = "openpilot.selfdrive.modeld.modeld_tinygrad"
BIG_MODEL_TIMEOUT = 60


def _pkl_exists(path):
  return os.path.exists(path)


def _find_driving_pkl(bundle):
  if (override := os.environ.get('COMBINED_MODEL_PKL')) and _pkl_exists(override):
    return override
  if bundle is None or not bundle.models:
    return None
  from openpilot.common.hardware.hw import Paths
  model_root = Paths.model_root()

  pkl_name = bundle.models[0].artifact.fileName
  pkl_path = os.path.join(model_root, pkl_name)
  if _pkl_exists(pkl_path):
    return pkl_path
  return None


class FrameMeta:
  frame_id: int = 0
  timestamp_sof: int = 0
  timestamp_eof: int = 0

  def __init__(self, vipc=None):
    if vipc is not None:
      self.frame_id, self.timestamp_sof, self.timestamp_eof = vipc.frame_id, vipc.timestamp_sof, vipc.timestamp_eof


class ModelState(ModelStateBase):
  inputs: dict[str, np.ndarray]
  prev_desire: np.ndarray

  def __init__(self, cam_w: int, cam_h: int, chestnut: bool = False):
    ModelStateBase.__init__(self)

    env_pkl = os.environ.get('COMBINED_MODEL_PKL')
    if env_pkl and os.path.exists(env_pkl):
      model_bundle = None
    else:
      model_bundle = get_active_bundle(chestnut=chestnut)
    self.generation = model_bundle.generation if model_bundle is not None else None
    overrides = {override.key: override.value for override in model_bundle.overrides} if model_bundle else {}

    self.LAT_SMOOTH_SECONDS = float(overrides.get('lat', ".0"))
    self.LONG_SMOOTH_SECONDS = float(overrides.get('long', ".0"))
    self.MIN_LAT_CONTROL_SPEED = 0.3
    self.PLANPLUS_CONTROL: float = 1.0
    self.chestnut = chestnut

    pkl_path = _find_driving_pkl(model_bundle)
    assert pkl_path is not None, f"No driving pkl found for {'chestnut' if chestnut else 'small model'} — all models must be compiled with compile_modeld.py"
    self._init_combined(pkl_path, cam_w, cam_h, model_bundle)

  def _init_combined(self, pkl_path, cam_w, cam_h, bundle):
    cloudlog.warning(f"loading combined pkl: {pkl_path}")
    jits = load_model(pkl_path, extend=fused_with_warp(cam_w, cam_h, self.chestnut))

    metadata = jits['metadata']
    self.WARP_DEV = metadata.get('warp_dev', 'QCOM') if COMMA_HARDWARE else 'CPU'
    self.DEV = ('AMD' if self.chestnut else 'QCOM') if COMMA_HARDWARE else 'CPU'
    self.QUEUE_DEV = self.DEV
    self.adapter = get_model_adapter(jits, cam_w, cam_h, self.DEV, self.QUEUE_DEV, self.WARP_DEV, self.chestnut)
    self.vision_output_slices = self.adapter.vision_output_slices
    self.policy_output_slices = self.adapter.policy_output_slices
    self._policy_slices_list = self.adapter._policy_slices_list
    self._combined_model_type = self.adapter._combined_model_type
    self._vision_input_names = self.adapter._vision_input_names
    self.numpy_inputs = self.adapter.numpy_inputs
    self._policy_keys = self.adapter._policy_keys
    self._has_on_policy = self.adapter._has_on_policy
    self._desire_key = self.adapter._desire_key
    self._road_key = self.adapter._road_key
    self._wide_key = self.adapter._wide_key
    self.frame_buf_params = self.adapter.frame_buf_params

    is_20hz = bundle.is20hz if bundle else self._combined_model_type in ('split', 'multi_policy')
    if is_20hz:
      from openpilot.sunnypilot.models.split_model_constants import SplitModelConstants
      self.constants = SplitModelConstants()
    else:
      self.constants = ModelConstants()

    self.parser = Parser()
    self.prev_desire = np.zeros(self.constants.DESIRE_LEN, dtype=np.float32)

  def warmup(self) -> None:
    dummy_frames, transforms, dummy_inputs = self.adapter.get_dummy_inputs()
    self.run(dummy_frames, transforms, dummy_inputs)
    self.adapter.reset_warmup_buffers()
    self.prev_desire[:] = 0

  @property
  def mlsim(self) -> bool:
    return bool(self.generation is not None and self.generation >= 11)

  @property
  def vision_input_names(self) -> list[str]:
    return self._vision_input_names

  @property
  def desire_key(self) -> str:
    return self._desire_key

  def run(self, bufs: dict[str, VisionBuf], transforms: dict[str, np.ndarray],
          inputs: dict[str, np.ndarray],
          after_enqueue: Callable[[], None] | None = None) -> dict[str, np.ndarray] | None:
    self.adapter.copy_frames(bufs)

    desire_key = self.desire_key
    inputs[desire_key][0] = 0
    (self.numpy_inputs[desire_key].flat if self.adapter.is_native else self.numpy_inputs[desire_key])[:] = \
      np.where(inputs[desire_key] - self.prev_desire > .99, inputs[desire_key], 0)
    self.prev_desire[:] = inputs[desire_key]

    for key in ('traffic_convention', 'lateral_control_params', 'action_t'):
      if key in self.numpy_inputs and key in inputs:
        self.numpy_inputs[key][:] = inputs[key]

    if self.adapter.is_native:
      for i, key in enumerate(self._vision_input_names):
        if key in transforms:
          self.numpy_inputs['tfm'][i] = transforms[key].reshape(3, 3)
    else:
      self.numpy_inputs['tfm'][:, :] = transforms[self._road_key].reshape(3, 3)
      self.numpy_inputs['big_tfm'][:, :] = transforms[self._wide_key].reshape(3, 3)

    raw_outputs = self.adapter.run()
    if self.chestnut and (err := Device[self.DEV].error_state.host.view(fmt='q')[0]):
      raise RuntimeError(f"chestnut failed with {err}")

    if after_enqueue is not None:
      after_enqueue()

    if self._combined_model_type == 'supercombo':
      model_output = raw_outputs.numpy().flatten()
      if self.chestnut and not np.all(np.isfinite(model_output)):
        raise RuntimeError("model output not finite")
      sliced = {k: model_output[np.newaxis, v] for k, v in self.vision_output_slices.items()}
      outputs = self.parser.parse_outputs(sliced)
      if 'prev_feat' in self.numpy_inputs and 'hidden_state' in self.vision_output_slices:
        (self.numpy_inputs['prev_feat'].flat if self.adapter.is_native else self.numpy_inputs['prev_feat'])[:] = \
          model_output[self.vision_output_slices['hidden_state']]
    else:
      vision_output = raw_outputs[0].numpy().flatten()
      vision_sliced = {k: vision_output[np.newaxis, v] for k, v in self.vision_output_slices.items()}
      outputs = self.parser.parse_vision_outputs(vision_sliced)

      if 'prev_feat' in self.numpy_inputs and 'hidden_state' in self.vision_output_slices:
        (self.numpy_inputs['prev_feat'].flat if self.adapter.is_native else self.numpy_inputs['prev_feat'])[:] = \
          vision_output[self.vision_output_slices['hidden_state']]

      for i, policy_slices in enumerate(self._policy_slices_list):
        policy_output = raw_outputs[i + 1].numpy().flatten()
        policy_sliced = {k: policy_output[np.newaxis, v] for k, v in policy_slices.items()}
        parsed = self.parser.parse_policy_outputs(policy_sliced)
        if 'off' in self._policy_keys[i] and self._has_on_policy:
          for key in ('plan', 'planplus', 'action'):
            if any(key in self._policy_slices_list[j] for j, k in enumerate(self._policy_keys) if 'on' in k.lower()):
              parsed.pop(key, None)

        outputs.update(parsed)

      if 'planplus' in outputs and 'plan' in outputs:
        outputs['plan'] = outputs['plan'] + outputs['planplus']

    if 'desired_curvature' in outputs and 'prev_desired_curv' in self.numpy_inputs:
      buf = self.numpy_inputs['prev_desired_curv']
      buf[0, :-1] = buf[0, 1:]
      buf[0, -1, :] = outputs['desired_curvature'][0, :] if not self.mlsim else 0

    return outputs

  def get_action_from_model(self, model_output: dict[str, np.ndarray], prev_action: log.ModelDataV2.Action,
                            lat_action_t: float, long_action_t: float, v_ego: float) -> log.ModelDataV2.Action:
    if 'action' not in model_output:
      plan = model_output['plan'][0]
      desired_accel = get_accel_from_plan(plan[:, Plan.VELOCITY][:, 0], plan[:, Plan.ACCELERATION][:, 0], self.constants.T_IDXS,
                                          action_t=long_action_t)

      curvature_plan = (plan + (self.PLANPLUS_CONTROL - 1.0) * model_output['planplus'][0]
                        if 'planplus' in model_output and self.PLANPLUS_CONTROL != 1.0 else plan)
      desired_curvature = get_curvature_from_output(model_output, curvature_plan, v_ego, lat_action_t, self.mlsim)
    else:
      desired_accel = model_output['action'][0, 1]
      desired_curvature = model_output['action'][0, 0] / (max(1.0, v_ego))**2

    stop = v_ego < 0.3 and desired_accel < 0.1
    desired_accel = smooth_value(desired_accel, prev_action.desiredAcceleration, self.LONG_SMOOTH_SECONDS)

    if self.generation is not None and self.generation >= 10: # smooth curvature for post FOF models
      if v_ego > self.MIN_LAT_CONTROL_SPEED:
        desired_curvature = smooth_value(desired_curvature, prev_action.desiredCurvature, self.LAT_SMOOTH_SECONDS)
      else:
        desired_curvature = prev_action.desiredCurvature

    return log.ModelDataV2.Action(desiredCurvature=float(desired_curvature), desiredAcceleration=float(desired_accel), shouldStop=bool(stop))


# VBSM_GPU_READY: the enclosure's custom firmware boots with PCIe off, and tinygrad's
# AMD open powers it on and checks the link ONCE (tinygrad usb.py: set_pcie_power then
# a single LTSSM read). At a cold boot the link is often still training, the open
# fails, and -- verified on-device 2026-09-14 -- that failure leaks both the tinygrad
# lock fd and the libusb claim inside this process, so no later open in the same
# process can ever succeed ("Failed to acquire lock", then "Resource busy"). So the
# FIRST open has to succeed. Before opening, modeld reads the bridge's supply and
# LTSSM registers straight over usbdevfs (read-only, own fd, no tinygrad, no lock),
# writes the PCIe power bit at most once and only with 12 V present, and waits,
# bounded, for L0. Since the 2026-10-19 port tinygrad's open is retried in a FRESH process by upstream's
# one onroad cycle (ChestnutRetry), so the wait no longer protects against the leak: it makes the FIRST
# attempt succeed, so a training or silent bridge does not spend the one retry. 'timeout', 'unreadable'
# and a failed load take upstream's retry (first: cycle, second: final); 'no_12v' and 'absent' are final at
# once (CHESTNUT_FINAL_REASONS), and so are 'fw_mismatch' / 'flashing' (CHESTNUT_FW_REASONS: the bridge came
# up during the wait on old, ROM or mid-flash firmware). The wait ends 22 s after main() starts, so 22 + 60
# (loader budget) + up to 6 (small model, or the wait for the cycle) stays 12 s under ui_watchdog's 100 s load
# deadline (VBSM_GPU_COLDWAIT may extend the wait at a cold boot, VBSM_GPU_COLDCACHE the loader budget and the
# SoC model's cold build; each moves that deadline by the same amount): the watchdog matches the exact
# proctitle, which only appears at setproctitle() a few ms into main(), and polls every 2 s, so its clock starts
# at or after ours. Imports (4-11 s warm, longer cold) come before main() and must not shrink the link budget
# (they did in e8b7e968, which counted from process start).
CHESTNUT_READY_TIMEOUT_S = 20.0
CHESTNUT_READY_END_BY_S = 22.0   # after main() entry, see above
CHESTNUT_READY_POLL_S = 0.5
# VBSM_GPU_12V_GRACE: the accessory outlet can power up a few seconds after modeld starts. On 2026-10-15 the device
# had booted ~1 min before ignition, modeld read 5-8 mV, called no_12v after three reads (1.0 s), and its late probe
# saw 13.9 V and L0 2.1 s later: the eGPU was skipped, "Big Model Failed / Restart the car" showed for 20 s and the
# kick reloaded modeld. The supply now has to read unpowered for this long before the wait gives up on it; a wait
# that runs out of budget while still unpowered also ends as no_12v (final at once: a 1 s cycle does not switch
# the outlet).
CHESTNUT_NO_12V_GRACE_S = 10.0
# VBSM_GPU_COLDWAIT: at a fresh power-up the bridge can be enumerated but silent ('unreadable'). In the field
# (2026-09-17..28, anchored to kernel boot) the silence ended 43.6-48.7 s after kernel boot (09-27: 47.3-50.7),
# only on the first start after the device had been off. The plain budget runs from main(), so a modeld that
# reaches main() early has less of it left: 09-27 had the earliest main() of all silent starts and missed it,
# the eGPU was skipped (then: a late-probe kick, ~40 s with no model). So once a read has come back
# unreadable, the wait may run on until CHESTNUT_COLD_BOOT_AGE_S after kernel boot, but never past
# CHESTNUT_COLD_MAX_S after main(). In practice the extension is max(0, min(20, 50 - boot age at the wait's
# start)) s: a wait that starts 50 s or more into the boot (e.g. the retry cycle) keeps the plain budget. The
# seconds used past the plain deadline go to ui_watchdog through CHESTNUT_PREFLIGHT_EXT_FILE, which moves its
# load deadline for this pid by as much. Re-derive both constants before anything makes main() earlier.
CHESTNUT_COLD_BOOT_AGE_S = 70.0
CHESTNUT_COLD_MAX_S = 40.0
CHESTNUT_PREFLIGHT_EXT_FILE = "/dev/shm/vbsm_gpu_preflight_ext"
CHESTNUT_F3_TIMEOUT_MS = 10000   # tinygrad's own timeout for the PCIe power write
CHESTNUT_F3_RESEND_S = 10.0      # re-send F3=1 only if the LTSSM is still in Detect this long after the last write
CHESTNUT_POWERED_MV = 5000       # helpers.CHESTNUT_POWERED_VOLTAGE
CHESTNUT_PCIE_L0 = 0x78
CHESTNUT_LOCK_DIR = "/tmp"
# VBSM_GPU_READY: 'absent' = this many reads (0.5 s apart) with no bridge in sysfs, ~10 s: upstream's own
# wait_for_chestnut() gives enumeration 10 s once a cable is seen, so the fork is never less patient
CHESTNUT_ABSENT_READS = 20
# VBSM_GPU_FALLBACK: readiness outcomes that are final for this ignition at once. A 1 s onroad cycle does not
# switch the accessory outlet or make a missing bridge enumerate (2026-10-18: the outlet came back only after a
# car restart), so these mark upstream's retry as used (ChestnutRetry) instead of spending it on a cycle, and
# upstream's own else-branch then reports the failure. Set to () to let them take upstream's retry as well.
CHESTNUT_FINAL_REASONS = ("no_12v", "absent")
# VBSM_GPU_FW_GATE: readiness outcomes for a bridge that came up during the wait (after main()'s gate) on old, ROM
# or mid-flash firmware (chestnut_firmware_busy). Always final at once, whatever CHESTNUT_FINAL_REASONS says: the
# firmware does not change this session, and an onroad cycle could start a flash in its 1 s offroad window
CHESTNUT_FW_REASONS = ("fw_mismatch", "flashing")
# VBSM_GPU_FALLBACK: after requesting upstream's onroad cycle, wait this long to be stopped (SIGINT ->
# VBSM_EXIT) before exiting on our own; 22 + 60 + 5 s stays 13 s under ui_watchdog's 100 s load deadline
CHESTNUT_CYCLE_WAIT_S = 5
# VBSM_GPU_COLDCACHE: the first load of a model file on a tinygrad build rebuilds every graph and only then
# writes <pkl>.cache (modeld_v2/helpers.py load_model); nothing pre-builds it for the eGPU model (modeld_warmup
# does the SoC model only), and the key includes a hash of the tinygrad source, so it is cold after every port
# that bumps tinygrad and after every model change. A cold load gets this much more loader budget, reported to
# ui_watchdog with the COLDWAIT seconds; and on the final attempt a cold build that is still running is left to
# finish in the background (upstream's own final behaviour) so the next ignition starts warm. The SoC model loads
# after it with ChestnutLoading still True and modeld_warmup never pre-builds a native model, so a cold SoC cache
# adds the same seconds to ui_watchdog's deadline (not to the loader's join, which is the big model's).
CHESTNUT_COLD_CACHE_EXTRA_S = 60.0
_last_f3 = float("-inf")
_f3_writes = 0


def proc_start_monotonic() -> float:
  """time.monotonic() at which this process started (from /proc/self/stat), for the `pid_age_s` field of the link
  events (imports included). The budget itself runs from `t_main`, as does ui_watchdog's 100 s load deadline (from
  its first sighting of the exact proctitle). Never the wall clock: it jumps at the boot-time NTP sync."""
  try:
    with open("/proc/self/stat") as f:
      st = f.read()
    ticks = int(st[st.rindex(")") + 2:].split()[19])   # field 22, starttime in clock ticks since boot
    age_s = time.clock_gettime(time.CLOCK_BOOTTIME) - ticks / os.sysconf("SC_CLK_TCK")
    return time.monotonic() - age_s
  except Exception:
    return time.monotonic()


PROC_START = proc_start_monotonic()


def chestnut_lock_fds() -> list[int]:
  """fds of this process open on tinygrad's am_usb:*.lock files -- a leaked one means an in-process open failed"""
  out = []
  for f in os.listdir("/proc/self/fd"):
    try:
      path = os.readlink(f"/proc/self/fd/{f}")
    except OSError:
      continue
    if os.path.dirname(path) == CHESTNUT_LOCK_DIR and os.path.basename(path).startswith("am_usb:") and path.endswith(".lock"):
      out.append(int(f))
  return out


def chestnut_raw() -> tuple[int, int, int] | None:
  """READ-ONLY (supply_mv, supply_ma, ltssm) from the bridge over usbdevfs EP0 (same reads as chestnut_power.py);
  own fd, closed in finally; None on any exception or when the bridge is not on the runtime firmware (hardwared
  flashes it before onroad, which upstream's cable predicate no longer waits for). No F3 write, no tinygrad, no flock."""
  import ctypes
  import fcntl
  import struct
  from openpilot.system.hardware.chestnut.flash import Ctrl, USBDEVFS_CONTROL, find_chestnut, open_device
  try:
    path, _, product = find_chestnut()
    if not path or product != CHESTNUT_USB_PRODUCT:   # VBSM_GPU_READY: never talk to old, ROM or mid-flash firmware
      return None
    fd = open_device(path)
  except Exception:
    return None
  try:
    buf = (ctypes.c_ubyte * 5)()
    fcntl.ioctl(fd, USBDEVFS_CONTROL, Ctrl(0xC0, 0xC0, 0, 0, 5, 2000, ctypes.cast(buf, ctypes.c_void_p)))
    lt = (ctypes.c_ubyte * 1)()
    fcntl.ioctl(fd, USBDEVFS_CONTROL, Ctrl(0xC0, 0xE4, 0xB450, 0, 1, 2000, ctypes.cast(lt, ctypes.c_void_p)))
    v, i = struct.unpack("<Hh", bytes(buf)[:4])
    return int(v), int(i), int(lt[0])
  except Exception:
    return None
  finally:
    os.close(fd)


def chestnut_enumerated() -> bool:
  """sysfs lists the bridge (runtime or ROM VID:PID, any product string). On 2026-09-16 the bridge was enumerated
  at SuperSpeed from boot but answered no control transfer for ~30 s. An old-firmware, ROM or mid-flash bridge
  reads as enumerated but unreadable (chestnut_raw() refuses it), so the wait keeps its budget and sends nothing."""
  try:
    from openpilot.system.hardware.chestnut.flash import find_chestnut
    return find_chestnut()[0] is not None
  except Exception:
    return False


def chestnut_f3_on() -> bool:
  """ONE PCIe power-on write (0xF3=1) with tinygrad's timeout; own fd; False on any exception."""
  global _last_f3, _f3_writes
  import fcntl
  from openpilot.system.hardware.chestnut.flash import Ctrl, USBDEVFS_CONTROL, find_chestnut, open_device
  try:
    path, _, product = find_chestnut()
    if not path or product != CHESTNUT_USB_PRODUCT:   # VBSM_GPU_READY: as chestnut_raw()
      return False
  except Exception:
    return False
  _last_f3 = time.monotonic()   # an attempt, successful or not: a failed write must not re-arm an immediate resend
  _f3_writes += 1
  try:
    fd = open_device(path)
  except Exception:
    return False
  try:
    fcntl.ioctl(fd, USBDEVFS_CONTROL, Ctrl(0x40, 0xF3, 1, 0, 0, CHESTNUT_F3_TIMEOUT_MS, None))
    return True
  except Exception:
    return False
  finally:
    os.close(fd)


def chestnut_fields(raw) -> dict:
  return {"supply_mv": raw[0], "supply_ma": raw[1], "ltssm": f"0x{raw[2]:02x}"} if raw else {"supply_mv": None, "supply_ma": None, "ltssm": None}


def chestnut_firmware_busy() -> str | None:
  """VBSM_GPU_FW_GATE: 'flashing' while hardwared's flasher (chestnut/flash.py, run through sudo) is alive, or
  'fw_mismatch' while a chestnut is on the bus with the ROM-bootloader ids or with a product string other than
  the pinned firmware (re-read 1 s later; an empty product, seen mid-enumeration, does not count); else None.
  sysfs and /proc only, no USB traffic. With 12 V present flash.py itself reports that new firmware "activates
  on its next power cycle", so neither state turns into a usable bridge during this onroad session, and an
  onroad cycle would not change that (it could only start a flash retry inside its 1 s offroad window)."""
  def mismatch() -> bool:
    for d in get_usb_state():
      if not is_chestnut_usb_id(d["vendorId"], d["productId"], include_bootloader=True):
        continue
      if not is_chestnut_usb_id(d["vendorId"], d["productId"]):
        return True   # ROM bootloader ids
      if d["product"] and d["product"] != CHESTNUT_USB_PRODUCT:
        return True   # old or unknown runtime firmware
    return False
  try:
    for pid in os.listdir("/proc"):
      if pid.isdigit():
        try:
          with open(f"/proc/{pid}/cmdline", "rb") as f:
            if b"chestnut/flash.py" in f.read():
              return "flashing"
        except OSError:
          pass
    if mismatch():
      time.sleep(1.0)
      if mismatch():
        return "fw_mismatch"
  except Exception:
    cloudlog.exception("chestnut firmware check failed")
  return None


def chestnut_cache_cold(pkl_path, extend=None) -> bool:
  """VBSM_GPU_COLDCACHE: True unless load_model(pkl_path, extend=extend) will hit <pkl>.cache. Builds upstream's
  own key (modeld_v2/helpers.py _cache_key: format, tinygrad source hash, size, mtime, frame-fusion digest, device
  map, runtime env) the way load_model() does, digest included (_extended with the same extend), so a new warp pkl
  or fused layout on an unchanged tinygrad reads cold too. upstream's cache_is_fresh() cannot answer this: it builds
  the key with an empty 'extra', which never matches a native (fused) model. tinygrad_identity() is cached, so
  load_model() does not hash the source again. The digest covers the fused part's header and program code only,
  not the weights: ~0.05-0.5 s for 1000-6000 kernels (Pi 5, upstream's real helpers on a synthetic bundle), the
  same prefix load_model() runs next over the same pages. Any doubt reads cold: a longer budget and, on the final
  attempt, the background build instead of the wedge exit (ui_watchdog's freeze check backs that one up)."""
  try:
    import json
    import struct
    from openpilot.sunnypilot.modeld_v2.helpers import _cache_key, _extended
    extra = _extended(pkl_path, extend)[2]
    with open(f"{pkl_path}.cache", "rb") as f:
      stored = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    return stored != json.loads(_cache_key(pkl_path, None, extra))
  except Exception:
    return True


def chestnut_watchdog_ext_add(extra_s: float) -> bool:
  """VBSM_GPU_COLDCACHE: add extra_s to the load-deadline extension this process reports to ui_watchdog through
  CHESTNUT_PREFLIGHT_EXT_FILE ("<pid> <seconds>"), on top of any VBSM_GPU_COLDWAIT seconds already in it.
  False if the file could not be written: the caller must then not use the extra budget."""
  ext = 0.0
  try:
    with open(CHESTNUT_PREFLIGHT_EXT_FILE) as f:
      p, s = f.read().split()
    if int(p) == os.getpid() and math.isfinite(float(s)):
      ext = max(0.0, float(s))
  except (OSError, ValueError):
    pass
  try:
    with open(CHESTNUT_PREFLIGHT_EXT_FILE, "w") as f:
      f.write(f"{os.getpid()} {math.ceil((ext + extra_s) * 10) / 10}")
    return True
  except OSError:
    cloudlog.exception("chestnut preflight extension marker write failed")
    return False


def wait_chestnut_ready(t_main: float) -> str:
  """'ready' | 'timeout' | 'no_12v' | 'absent' | 'unreadable' | 'fw_mismatch' | 'flashing' | 'probe_error'. Never raises. Read-first: F3=1 only with
  12 V and the link down; the preflight's write counts (f3_total), so a cold boot normally shows f3_writes=0 here and
  f3_total=1. 'absent' = CHESTNUT_ABSENT_READS (~10 s) with no bridge in sysfs; 'unreadable' = the bridge is enumerated but answered no
  read by the deadline (2026-09-16 cold boot: enumerated at 5000 Mb/s from boot, silent for ~30 s); 'no_12v' = the
  supply read under 5 V for CHESTNUT_NO_12V_GRACE_S, or still did when the budget ended (VBSM_GPU_12V_GRACE);
  'fw_mismatch' / 'flashing' = an unreadable bridge is on old, ROM or mid-flash firmware (VBSM_GPU_FW_GATE)."""
  t0 = time.monotonic()
  boot_age0 = deadline = cold_deadline = None
  reads = f3 = misses = unreadable = 0
  unpowered_since = None   # VBSM_GPU_12V_GRACE: start of the current run of reads under 5 V
  raw = None
  reason = "timeout"

  def extension_s() -> float:
    # VBSM_GPU_COLDWAIT: seconds past the plain deadline, only when the extension was in effect; rounded up. Plus
    # the seconds the plain deadline itself sits past main()+22 (a vision-stream wait over 20 s moves it to t0+2),
    # which ui_watchdog's load deadline would not cover otherwise
    if deadline is None:
      return 0.0
    ext = max(0.0, deadline - (t_main + CHESTNUT_READY_END_BY_S))
    if unreadable and cold_deadline is not None and cold_deadline > deadline:
      ext += max(0.0, time.monotonic() - deadline)
    return math.ceil(ext * 10) / 10

  def tell_watchdog(ext: float) -> None:
    # VBSM_GPU_COLDWAIT: before the 60 s loader budget starts, so ui_watchdog's load deadline moves with us
    if ext > 0:
      try:
        with open(CHESTNUT_PREFLIGHT_EXT_FILE, "w") as f:
          f.write(f"{os.getpid()} {ext}")
      except OSError:
        cloudlog.exception("chestnut preflight extension marker write failed")

  try:
    boot_age0 = time.clock_gettime(time.CLOCK_BOOTTIME)
    deadline = min(t0 + CHESTNUT_READY_TIMEOUT_S, t_main + CHESTNUT_READY_END_BY_S)
    deadline = max(deadline, t0 + 2.0)
    # VBSM_GPU_COLDWAIT: only a bridge that has read unreadable, early in the boot, gets the extension
    cold_deadline = max(deadline, min(t_main + CHESTNUT_COLD_MAX_S, t0 + CHESTNUT_COLD_BOOT_AGE_S - boot_age0))
    while True:
      raw = chestnut_raw()
      reads += 1
      if raw is None:
        if chestnut_enumerated():
          unreadable += 1   # present on the bus, not answering: keep the whole budget, it may wake up
          # VBSM_GPU_FW_GATE: so does a bridge that came up after main()'s gate on old, ROM or mid-flash firmware
          # (chestnut_raw refuses it), which no wait or cycle changes: final now. Every 4th unreadable read (~2 s);
          # sysfs and /proc only, plus the 1 s confirm read on a mismatch
          if unreadable % 4 == 1 and (fw := chestnut_firmware_busy()):
            reason = fw
            break
        else:
          misses += 1
          if misses >= CHESTNUT_ABSENT_READS:
            reason = "absent"
            break
      else:
        mv, _, lt = raw
        if lt == CHESTNUT_PCIE_L0:
          reason = "ready"
          break
        if mv < CHESTNUT_POWERED_MV:
          if unpowered_since is None:
            unpowered_since = time.monotonic()
          if time.monotonic() - unpowered_since >= CHESTNUT_NO_12V_GRACE_S:   # VBSM_GPU_12V_GRACE
            reason = "no_12v"
            break
        else:
          unpowered_since = None
        never_written = _last_f3 == float("-inf")
        in_detect = lt in (0x00, 0x01)   # an unpowered link reads 0x00 or 0x01 (bench 2026-09-14)
        if mv >= CHESTNUT_POWERED_MV and (never_written or (in_detect and time.monotonic() - _last_f3 >= CHESTNUT_F3_RESEND_S)):
          f3 += 1
          chestnut_f3_on()
      if time.monotonic() >= (cold_deadline if unreadable else deadline):
        if raw is None and unreadable:
          reason = chestnut_firmware_busy() or "unreadable"   # VBSM_GPU_FW_GATE: the last look before a cycle
        elif unpowered_since is not None:
          reason = "no_12v"   # VBSM_GPU_12V_GRACE: out of budget with no 12 V yet
        else:
          reason = "timeout"
        break
      time.sleep(CHESTNUT_READY_POLL_S)
  except Exception:
    cloudlog.exception("chestnut link probe error")
    tell_watchdog(extension_s())   # probe_error still starts the loader
    return "probe_error"
  extended_s = extension_s()
  tell_watchdog(extended_s)
  fields = dict(reason=reason, reads=reads, unreadable=unreadable, f3_writes=f3, f3_total=_f3_writes, wait_s=round(time.monotonic() - t0, 1),
                main_age_s=round(time.monotonic() - t_main, 1), pid_age_s=round(time.monotonic() - PROC_START, 1),
                wait_boot_age_s=round(boot_age0, 1), extended_s=extended_s, **chestnut_fields(raw))
  if reason == "ready":
    cloudlog.event("chestnut link ready", **fields)  # VBSM_LOG_LEVEL: no error kwarg on success
  else:
    cloudlog.event("chestnut link not ready", error=True, **fields)
  return reason


def main(demo=False):
  t_main = time.monotonic()   # VBSM_GPU_READY: ui_watchdog's clock starts at the setproctitle() just below
  cloudlog.warning("modeld init")

  cloudlog.bind(daemon=PROCESS_NAME)
  setproctitle(PROCESS_NAME)

  CHESTNUT = _find_driving_pkl(get_active_bundle(chestnut=True)) is not None and (chestnut_present() or cable_connected())
  if CHESTNUT:
    from tinygrad.runtime.ops_amd import AMDDevice
    AMDDevice.wait_timeout_ms = 3000

  params = Params()
  # VBSM_GPU_FALLBACK: upstream never restarts modeld inside an onroad session, so it only ever tries the eGPU at
  # the start of one, and its retry cycle can only follow that first load. The fork restarts a dead modeld
  # (VBSM_RESTART), so keep upstream's rule: ChestnutLoading is cleared at every onroad->offroad edge, at
  # ignition-on and at manager start, and every modeld writes it below. Already set = an earlier modeld of THIS
  # session ran (crash, wedge exit, ui_watchdog wedge kill): run the SoC model until the next onroad start. No
  # mid-drive eGPU load (25-60 s with no model while moving), and no onroad cycle requested at speed.
  loading_prev, active_prev = params.get("ChestnutLoading"), params.get("ChestnutActive")
  respawned = CHESTNUT and loading_prev is not None
  firmware_busy = chestnut_firmware_busy() if CHESTNUT and not respawned else None
  if respawned:
    cloudlog.event("eGPU skipped: modeld restarted inside this onroad session; SoC model until the next one",
                   error=True, loading_prev=loading_prev, active_prev=active_prev)
    CHESTNUT = False
  elif firmware_busy:
    # VBSM_GPU_FW_GATE: the bridge is being flashed, is in the ROM bootloader or still runs old firmware, and none
    # of that changes during this session (see chestnut_firmware_busy). Run the SoC model without touching the
    # bridge, upstream's retry or the alerts: ChestnutActive stays absent, so no "Restart the car to retry" (a car
    # restart does not activate firmware), and no onroad cycle. hardwared sends no rail command meanwhile.
    cloudlog.event("eGPU skipped: chestnut firmware not current; SoC model this session", error=True, reason=firmware_busy)
    CHESTNUT = False
  lost = respawned and (loading_prev or active_prev is not None)
  if lost:
    # this session tried the big model and lost it: report it, queued BEFORE ChestnutLoading (one FIFO writer), so
    # selfdrived never reads "load done, ChestnutActive still True" in between and flashes "Big Model Ready"
    params.put_bool("ChestnutActive", False)
  params.put_bool("ChestnutLoading", CHESTNUT)
  if not lost:
    params.remove("ChestnutActive")
  try:
    os.remove(CHESTNUT_PREFLIGHT_EXT_FILE)   # VBSM_GPU_COLDWAIT: every process decides afresh
  except OSError:
    pass
  if CHESTNUT:
    # VBSM_GPU_READY preflight: record the bridge state this process starts from and, with
    # 12 V present and the link down, send the single PCIe power-on now so link training
    # overlaps the vision-stream wait below. Read-first: a warm start writes nothing.
    try:
      raw = chestnut_raw()
      f3_written = None
      if raw is not None and raw[0] >= CHESTNUT_POWERED_MV and raw[2] != CHESTNUT_PCIE_L0:
        f3_written = chestnut_f3_on()
      cloudlog.event("chestnut preflight", dev=os.environ.get("DEV"), amd_opened=("AMD" in Device._opened_devices),
                     lock_fds=len(chestnut_lock_fds()), pid_age_s=round(time.monotonic() - PROC_START, 1), f3_written=f3_written,
                     ltssm_after=(f"0x{r[2]:02x}" if f3_written and (r := chestnut_raw()) else None), **chestnut_fields(raw))  # VBSM_LOG_LEVEL
    except Exception:
      cloudlog.exception("chestnut preflight failed")

  gc.disable()

  # visionipc clients
  while True:
    available_streams = VisionIpcClient.available_streams("camerad", block=False)
    if available_streams:
      use_extra_client = VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams and VisionStreamType.VISION_STREAM_NARROW_ROAD in available_streams
      main_wide_camera = VisionStreamType.VISION_STREAM_NARROW_ROAD not in available_streams
      break
    time.sleep(.1)

  vipc_client_main_stream = VisionStreamType.VISION_STREAM_WIDE_ROAD if main_wide_camera else VisionStreamType.VISION_STREAM_NARROW_ROAD
  vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True)
  vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False)
  cloudlog.warning(f"vision stream set up, main_wide_camera: {main_wide_camera}, use_extra_client: {use_extra_client}")

  while not vipc_client_main.connect(False):
    time.sleep(0.1)
  while use_extra_client and not vipc_client_extra.connect(False):
    time.sleep(0.1)

  cloudlog.warning(f"connected main cam with buffer size: {vipc_client_main.buffer_len} ({vipc_client_main.width} x {vipc_client_main.height})")
  if use_extra_client:
    cloudlog.warning(f"connected extra cam with buffer size: {vipc_client_extra.buffer_len} ({vipc_client_extra.width} x {vipc_client_extra.height})")

  cloudlog.warning("loading model")
  st = time.monotonic()

  model = None
  if CHESTNUT:
    def apply_ppt_cap():
      # VBSM_GPU_PPT: cap package power BEFORE the model transfer -- the
      # accessory-outlet 12V path (~0.45 ohm measured) browns out at stock
      # boost transients, during load as well as inference. 80W ~ 6A fits.
      # Tune via /data/vbsm_gpu_ppt_w (chestnut_ppt_limit_w(); 0 = leave upstream's
      # AM_POWER_LIMIT, 100 W since b50b82f9). The same value is exported as
      # AM_POWER_LIMIT at import, so the AMD boot inside this first open already
      # applies it; this write and read-back confirm it in the log.
      limit_w = chestnut_ppt_limit_w()
      if limit_w <= 0:
        return
      t_open = time.monotonic()
      smu = Device["AMD"].iface.dev_impl.smu
      open_ms = int((time.monotonic() - t_open) * 1000)
      smu._send_msg(smu.smu_mod.PPSMC_MSG_SetPptLimit, limit_w, timeout=100)
      applied = smu._send_msg(smu.smu_mod.PPSMC_MSG_GetPptLimit, 0, read_back_arg=True, timeout=100)
      cloudlog.event("chestnut ppt limit", requested=limit_w, applied=int(applied), open_ms=open_ms)  # VBSM_LOG_LEVEL

    big_model = None
    cache_cold = None   # VBSM_GPU_COLDCACHE: True once the loader found a cold cache AND told ui_watchdog
    def load_big():
      nonlocal big_model, cache_cold
      cloudlog.bind(daemon=PROCESS_NAME)   # the daemon tag is thread-local; without this the loader's lines carry none
      try:
        # VBSM_GPU_COLDCACHE: decided here, inside the loader budget, so it never delays the readiness wait or
        # the join. It computes the frame-fusion digest exactly as load_model() is about to (headers and kernels,
        # not the weights; check_s in the event), and the tinygrad source hash, which load_model() reuses
        t_chk = time.monotonic()
        cold = chestnut_cache_cold(_find_driving_pkl(get_active_bundle(chestnut=True)),
                                   fused_with_warp(vipc_client_main.width, vipc_client_main.height, True))
        cache_cold = cold and chestnut_watchdog_ext_add(CHESTNUT_COLD_CACHE_EXTRA_S)
        cloudlog.event("chestnut cache", cold=cold, extended_s=CHESTNUT_COLD_CACHE_EXTRA_S if cache_cold else 0.0,  # VBSM_LOG_LEVEL
                       check_s=round(time.monotonic() - t_chk, 2))
        wait_for_chestnut()
        apply_ppt_cap()   # VBSM_GPU_PPT: after the bridge is up, before the transfer
        m = ModelState(cam_w=vipc_client_main.width, cam_h=vipc_client_main.height, chestnut=True)
        m.warmup()
        big_model = m
      except Exception:
        cloudlog.exception("chestnut load failed")
    loader = threading.Thread(target=load_big, daemon=True)
    # VBSM_GPU_READY: probe in the main thread so the 60 s loader budget still covers
    # the load alone; an unstarted loader reads as "finished, no model" (a failed load)
    ready_reason = wait_chestnut_ready(t_main)
    if ready_reason in ("ready", "probe_error"):
      loader.start()
      loader.join(BIG_MODEL_TIMEOUT)
      if loader.is_alive() and cache_cold is True:
        loader.join(CHESTNUT_COLD_CACHE_EXTRA_S)   # VBSM_GPU_COLDCACHE: the first load on this tinygrad builds the cache
    else:
      cloudlog.event("eGPU load skipped; link not ready", reason=ready_reason, error=True)   # nothing opened
      if ready_reason in CHESTNUT_FINAL_REASONS or ready_reason in CHESTNUT_FW_REASONS:
        # VBSM_GPU_FALLBACK: final at once. Marking upstream's retry as used sends the branch below to upstream's
        # own verdict (ChestnutActive False: "Big Model Failed / Restart the car to retry", true here: a car
        # restart is what brought the outlet back on 2026-10-18) with no onroad cycle. block: read back at once.
        # VBSM_GPU_FW_GATE: firmware found during the wait ends the same way (no cycle, so no flash can start in
        # its offroad window); only the gate at main() is silent.
        params.put_bool("ChestnutRetry", True, block=True)
    model = big_model
    if model is None and not params.get_bool("ChestnutRetry"):
      # a crank can cut the chestnut power during the load, retry once with an onroad cycle
      cloudlog.error("big model load failed, cycling onroad to retry")
      params.put_bool("ChestnutRetry", True)
      params.put_bool("OnroadCycleRequested", True)
      # VBSM_GPU_FALLBACK: the cycle stops this process within ~1 s (SIGINT -> VBSM_EXIT; the manager SIGKILLs it
      # 5 s later if it cannot run Python). Load nothing behind a loader that may be wedged in a C call (route 7f:
      # the in-process small-model load hung behind it) and write nothing the cycle would have to clear. If the
      # cycle never comes, exit: the VBSM_RESTART respawn finds ChestnutLoading set and runs the SoC model. The two
      # puts above are queued FIFO on libparams' writer thread, which this wait leaves time to flush.
      cloudlog.event("eGPU load failed; waiting for the onroad cycle", error=True, reason=ready_reason,
                     loader_alive=loader.is_alive())
      for _ in range(CHESTNUT_CYCLE_WAIT_S):
        time.sleep(1)
      cloudlog.event("eGPU onroad cycle did not come; exiting for a SoC respawn", error=True)
      os._exit(1)
    else:
      params.put_bool("ChestnutActive", model is not None)
      if model is None and loader.is_alive() and cache_cold is True:
        # VBSM_GPU_COLDCACHE: final attempt and the loader is still building a cold cache: leave it running
        # (upstream's own final behaviour) so <pkl>.cache is written for the next ignition, and load the SoC model
        # in-process. If the loader is wedged instead, that load can hang behind it (route 7f): ChestnutLoading then
        # stays True and ui_watchdog replaces the process at its extended deadline (VBSM_GPU_WEDGE).
        cloudlog.event("eGPU cache build still running after the final attempt; SoC model in-process, build continues",
                       error=True, reason=ready_reason)
      elif model is None and loader.is_alive():
        # VBSM_GPU_FALLBACK: final attempt, loader still inside a C call: the in-process SoC load can hang
        # behind it (route 7f), so replace the process; the respawn sees ChestnutLoading set and runs SoC
        cloudlog.event("eGPU load wedged; exiting for a SoC respawn", error=True, reason=ready_reason)
        params.put_bool("ChestnutActive", False, block=True)   # flushed before os._exit
        os._exit(1)

  if CHESTNUT:
    # VBSM_GPU_COLDCACHE: the SoC model loads next with ChestnutLoading still True on every eGPU path that gets here
    # (big model loaded, final-at-once skip, final failure, heal), and modeld_warmup never pre-builds a native
    # model, so a cold SoC cache (drive 1 after a tinygrad bump, any small-model change) moves ui_watchdog's load
    # deadline too. Every path gets here >= ~13 s before that deadline (DECISION, integration addendum).
    t_chk = time.monotonic()
    cold_small = chestnut_cache_cold(_find_driving_pkl(get_active_bundle(chestnut=False)),
                                     fused_with_warp(vipc_client_main.width, vipc_client_main.height, False))
    small_ext = cold_small and chestnut_watchdog_ext_add(CHESTNUT_COLD_CACHE_EXTRA_S)
    cloudlog.event("chestnut small cache", cold=cold_small, extended_s=CHESTNUT_COLD_CACHE_EXTRA_S if small_ext else 0.0,  # VBSM_LOG_LEVEL
                   check_s=round(time.monotonic() - t_chk, 2))
  small_model = ModelState(cam_w=vipc_client_main.width, cam_h=vipc_client_main.height, chestnut=False) if model is None or CHESTNUT else None
  if model is None:
    model = small_model
  params.put_bool("ChestnutLoading", False)
  assert model is not None
  cloudlog.warning(f"models loaded in {time.monotonic() - st:.1f}s, modeld starting")

  config_realtime_process(7, 54)

  # messaging
  pub_socks = ["modelV2", "drivingModelData", "cameraOdometry", "modelDataV2SP"] + (["chestnutGpuState"] if CHESTNUT else [])
  pm = PubMaster(pub_socks)
  sm = SubMaster(["deviceState", "carState", "narrowRoadCameraState", "extrinsicsCalibration", "driverMonitoringState",
                  "carControl", "controlsState", "lateralDelay"])

  publish_state = PublishState()
  chestnut_state = ChestnutGpuState(pm, model.chestnut) if CHESTNUT else None

  # setup filter to track dropped frames
  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / model.constants.MODEL_FREQ)
  frame_id = 0
  last_vipc_frame_id = 0
  run_count = 0

  model_transform_main = np.zeros((3, 3), dtype=np.float32)
  model_transform_extra = np.zeros((3, 3), dtype=np.float32)
  live_calib_seen = False
  buf_main, buf_extra = None, None
  meta_main = FrameMeta()
  meta_extra = FrameMeta()
  camera_offset_helper = CameraOffsetHelper()


  if demo:
    CP = get_demo_car_params()
  else:
    CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  cloudlog.info("modeld got CarParams: %s", CP.brand)

  # TODO Move smooth seconds to action function
  long_delay = CP.longitudinalActuatorDelay + model.LONG_SMOOTH_SECONDS
  prev_action = log.ModelDataV2.Action()

  DH = DesireHelper()
  meta_constants = load_meta_constants()
  RELC = RoadEdgeLaneChangeController()

  while True:
    # Keep receiving frames until we are at least 1 frame ahead of previous extra frame
    while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
      buf_main = vipc_client_main.recv()
      meta_main = FrameMeta(vipc_client_main)
      if buf_main is None:
        break

    if buf_main is None:
      cloudlog.debug("vipc_client_main no frame")
      continue

    if use_extra_client:
      # Keep receiving extra frames until frame id matches main camera
      while True:
        buf_extra = vipc_client_extra.recv()
        meta_extra = FrameMeta(vipc_client_extra)
        if buf_extra is None or meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
          break

      if buf_extra is None:
        cloudlog.debug("vipc_client_extra no frame")
        continue

      if abs(meta_main.timestamp_sof - meta_extra.timestamp_sof) > 10000000:
        cloudlog.error(f"frames out of sync! main: {meta_main.frame_id} ({meta_main.timestamp_sof / 1e9:.5f}),\
                       extra: {meta_extra.frame_id} ({meta_extra.timestamp_sof / 1e9:.5f})")

    else:
      # Use single camera
      buf_extra = buf_main
      meta_extra = meta_main

    sm.update(0)
    desire = DH.desire
    is_rhd = sm["driverMonitoringState"].isRHD
    frame_id = sm["narrowRoadCameraState"].frameId
    v_ego = max(sm["carState"].vEgo, 0.)
    if sm.frame % 60 == 0:
      model.lat_delay = get_lat_delay(params, sm["lateralDelay"].lateralDelay)
      model.PLANPLUS_CONTROL = params.get("PlanplusControl", return_default=True)
      camera_offset_helper.set_offset(params.get("CameraOffset", return_default=True))
    lat_delay = model.lat_delay + model.LAT_SMOOTH_SECONDS
    if sm.updated["extrinsicsCalibration"] and sm.seen['narrowRoadCameraState'] and sm.seen['deviceState']:
      device_from_calib_euler = np.array(sm["extrinsicsCalibration"].rpyCalib, dtype=np.float32)
      dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['narrowRoadCameraState'].sensor))]
      main_intrinsics = dc.wide_road.intrinsics if main_wide_camera else dc.narrow_road.intrinsics
      model_transform_main = get_warp_matrix(device_from_calib_euler, main_intrinsics, False).astype(np.float32)
      model_transform_extra = get_warp_matrix(device_from_calib_euler, dc.wide_road.intrinsics, True).astype(np.float32)
      model_transform_main, model_transform_extra = camera_offset_helper.update(model_transform_main, model_transform_extra, sm, main_wide_camera)
      live_calib_seen = True

    traffic_convention = np.zeros(2)
    traffic_convention[int(is_rhd)] = 1

    vec_desire = np.zeros(model.constants.DESIRE_LEN, dtype=np.float32)
    if desire >= 0 and desire < model.constants.DESIRE_LEN:
      vec_desire[desire] = 1

    # tracked dropped frames
    vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
    frames_dropped = frame_dropped_filter.update(min(vipc_dropped_frames, 10))
    if run_count < 10: # let frame drops warm up
      frame_dropped_filter.x = 0.
      frames_dropped = 0.
    run_count = run_count + 1

    frame_drop_ratio = frames_dropped / (1 + frames_dropped)

    bufs = {name: buf_extra if 'big' in name else buf_main for name in model.vision_input_names}
    transforms = {name: model_transform_extra if 'big' in name else model_transform_main for name in model.vision_input_names}

    frame_delay = DT_MDL # compensate for time passed since the frame was captured: current_time - timestamp_eof is 50ms on average
    action_delay = DT_MDL / 2 # middle of the interval between model output (current state) and next frame (expected state)
    lat_action_t = lat_delay + frame_delay + action_delay
    long_action_t = long_delay + frame_delay + action_delay

    inputs:dict[str, np.ndarray] = {
      model.desire_key: vec_desire,
      'traffic_convention': traffic_convention,
    }

    if 'lateral_control_params' in model.numpy_inputs:
      inputs['lateral_control_params'] = np.array([v_ego, lat_delay], dtype=np.float32)

    if 'action_t' in model.numpy_inputs:
      inputs['action_t'] = np.array([lat_action_t, long_action_t], dtype=np.float32)

    mt1 = time.perf_counter()
    try:
      send_chestnut = (chestnut_state is not None and
                       run_count % round(model.constants.MODEL_FREQ / SERVICE_LIST['chestnutGpuState'].frequency) == 0)
      model_output = model.run(bufs, transforms, inputs, chestnut_state.send if send_chestnut else None)
    except Exception:
      if not model.chestnut:
        raise
      cloudlog.exception("big model failed, fall back to small")
      params.put_bool("ChestnutActive", False)
      assert small_model is not None
      model = small_model
      if chestnut_state is not None:
        chestnut_state.big = False
      run_count = 0
      model_output = None
    mt2 = time.perf_counter()
    model_execution_time = mt2 - mt1

    if model_output is not None:
      modelv2_send = messaging.new_message('modelV2')
      drivingdata_send = messaging.new_message('drivingModelData')
      posenet_send = messaging.new_message('cameraOdometry')
      mdv2sp_send = messaging.new_message('modelDataV2SP')

      if not sm['carControl'].latActive:
        prev_action.desiredCurvature = sm['controlsState'].curvature
      action = model.get_action_from_model(model_output, prev_action, lat_action_t, long_action_t, v_ego)
      prev_action = action
      fill_model_msg(drivingdata_send, modelv2_send, model_output, action,
                     publish_state, meta_main.frame_id, meta_extra.frame_id, frame_id,
                     frame_drop_ratio, meta_main.timestamp_eof, model_execution_time, live_calib_seen, meta_constants)
      modelv2_send.modelV2.big = model.chestnut

      desire_state = modelv2_send.modelV2.meta.desireState
      l_lane_change_prob = desire_state[log.Desire.laneChangeLeft]
      r_lane_change_prob = desire_state[log.Desire.laneChangeRight]
      lane_change_prob = l_lane_change_prob + r_lane_change_prob
      left_edge, right_edge = RELC.update_and_fill(modelv2_send.modelV2, mdv2sp_send.modelDataV2SP, v_ego)
      DH.update(sm['carState'], sm['carControl'].latActive, lane_change_prob, left_edge, right_edge)
      modelv2_send.modelV2.meta.laneChangeState = DH.lane_change_state
      modelv2_send.modelV2.meta.laneChangeDirection = DH.lane_change_direction
      mdv2sp_send.valid = modelv2_send.valid
      mdv2sp_send.modelDataV2SP.laneTurnDirection = DH.lane_turn_direction
      drivingdata_send.drivingModelData.meta.laneChangeState = DH.lane_change_state
      drivingdata_send.drivingModelData.meta.laneChangeDirection = DH.lane_change_direction

      fill_pose_msg(posenet_send, model_output, meta_main.frame_id, vipc_dropped_frames, meta_main.timestamp_eof, live_calib_seen)
      pm.send('modelV2', modelv2_send)
      pm.send('drivingModelData', drivingdata_send)
      pm.send('cameraOdometry', posenet_send)
      pm.send('modelDataV2SP', mdv2sp_send)
    last_vipc_frame_id = meta_main.frame_id

if __name__ == "__main__":
  try:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='A boolean for demo mode.')
    args = parser.parse_args()
    main(demo=args.demo)
  except KeyboardInterrupt:
    cloudlog.warning(f"child {PROCESS_NAME} got SIGINT")
    # VBSM_EXIT: the manager's SIGINT lands mid-frame and a normal interpreter
    # exit then walks tinygrad's USB/AMD teardown (atexit finalize + GC frees
    # over USB), which outlives the manager's 5 s grace, so every car-off ended
    # in a SIGKILL. Skipping it is safe: the usbdevfs claim and lock die with
    # the process, hardwared cuts the rails 120 s into the park (a power cycle
    # clears everything), and a re-open before that finds SCRATCH_REG6 set and
    # takes tinygrad's own full-reset path (mode1 reset, ~1-2 s slower open),
    # exactly what a ui_watchdog wedge SIGKILL already produces. Same exit as
    # the wedge paths above. The retry cycle relies on it: no 5 s SIGKILL wait.
    time.sleep(0.2)
    os._exit(0)
  except Exception:
    cloudlog.exception("modeld exception")
    raise
