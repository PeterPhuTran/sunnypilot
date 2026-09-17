#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from collections.abc import Callable
import os
os.environ['GMMU'] = '0'
# VBSM_GPU_READY: tinygrad resolves Device.DEFAULT by probing every backend in order
# (AMD before QCOM) the first time anything asks for it, and modeld_v2's import chain
# asks at import time (compile_modeld.py declares `device: str = Device.DEFAULT` as a
# default argument). On a cold boot that probe opens the AMD device before the PCIe
# link is up, fails, and leaks tinygrad's lock fd and the libusb claim, after which
# this process can never open the eGPU (verified on-device 2026-09-14). Pinning DEV
# makes DEFAULT resolve without probing; every tensor in modeld_v2 names its device
# explicitly, and the eGPU is opened deliberately in load_big().
os.environ.setdefault("DEV", "QCOM")
import numpy as np
import threading
import time
import traceback
from setproctitle import setproctitle
from tinygrad.tensor import Tensor

import openpilot.cereal.messaging as messaging
from openpilot.common.hardware import COMMA_HARDWARE
from openpilot.selfdrive.modeld.helpers import chestnut_present
from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.cereal.services import SERVICE_LIST
from openpilot.cereal.messaging import PubMaster, SubMaster
from openpilot.cereal.visionipc import VisionStreamType
from msgq.visionipc import VisionIpcClient, VisionBuf
from opendbc.car.car_helpers import get_demo_car_params
from openpilot.common.file_chunker import open_file_chunked
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.system import sentry
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, smooth_value
from openpilot.selfdrive.modeld.modeld import ChestnutState

from openpilot.selfdrive.modeld.compile_modeld import (
  MODELD_INPUTS,
  make_input_queues as make_stock_input_queues,
)
from openpilot.sunnypilot.modeld_v2.fill_model_msg import fill_model_msg, fill_pose_msg, PublishState, get_curvature_from_output
from openpilot.sunnypilot.modeld_v2.parse_model_outputs import Parser
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants, Plan
from openpilot.sunnypilot.modeld_v2.meta_helper import load_meta_constants
from openpilot.sunnypilot.modeld_v2.camera_offset_helper import CameraOffsetHelper
from openpilot.sunnypilot.modeld_v2.compile_modeld import (derive_frame_skip, make_split_input_queues,
                                                           make_supercombo_input_queues, nv12_copy_size,
                                                           WARP_INPUTS, POLICY_INPUTS)
from openpilot.sunnypilot.livedelay.helpers import get_lat_delay
from openpilot.sunnypilot.modeld_v2.modeld_base import ModelStateBase
from openpilot.sunnypilot.modeld_v2.helpers import load_oob
from openpilot.sunnypilot.models.helpers import get_active_bundle
from openpilot.sunnypilot.selfdrive.controls.lib.relc import RoadEdgeLaneChangeController

PROCESS_NAME = "openpilot.selfdrive.modeld.modeld_tinygrad"
BIG_MODEL_TIMEOUT = 60


def _pkl_exists(path):
  from openpilot.common.file_chunker import get_manifest_path
  return os.path.exists(path) or os.path.exists(get_manifest_path(path))


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
    jits = load_oob(open_file_chunked(pkl_path))

    metadata = jits['metadata']
    self.WARP_DEV = metadata.get('warp_dev', 'QCOM') if COMMA_HARDWARE else 'CPU'
    self.DEV = ('AMD' if self.chestnut else 'QCOM') if COMMA_HARDWARE else 'CPU'
    self.QUEUE_DEV = self.DEV
    self.is_run_model = 'run_model' in jits

    nv12_info = get_nv12_info(cam_w, cam_h)
    self.frame_copy_size = nv12_copy_size(*nv12_info[:3])
    self.full_frames: dict = {}
    self._blob_cache: dict = {}
    self.frame_buffers: dict = {}

    if self.is_run_model or 'model' in metadata:
      model_metadata = metadata.get('model', metadata)
      self.input_shapes = model_metadata['input_shapes']
      self.vision_output_slices = model_metadata['output_slices']
      self.policy_output_slices = {}
      self._policy_slices_list = []
      self._combined_model_type = 'supercombo'
      self._vision_input_names = [key for key in self.input_shapes if 'img' in key]
      self.frame_skip = derive_frame_skip({}, self.input_shapes)
      if self.is_run_model:
        self.input_queues, self.numpy_inputs, self.frame_buffers = make_stock_input_queues(
          self.input_shapes, self.frame_skip, device=self.DEV, frame_copy_size=self.frame_copy_size)
        self.frame_views, self.npy = self.frame_buffers, self.numpy_inputs
        self.run_model, self.run_policy, self.warp = jits['run_model'][(cam_w, cam_h)], None, None
      else:
        self.input_queues, self.numpy_inputs = make_supercombo_input_queues(self.input_shapes, self.frame_skip, device=self.QUEUE_DEV)
        self.run_model, self.run_policy, self.warp = None, jits['run_policy'], jits[(cam_w, cam_h)]
    else:
      self.run_model, self.run_policy, self.warp = None, jits['run_policy'], jits[(cam_w, cam_h)]
      vision_metadata = metadata['vision']
      policy_keys = [k for k in metadata if k not in ('vision', 'warp_dev')]
      self._combined_model_type = 'split' if policy_keys == ['policy'] else 'multi_policy'
      self.vision_output_slices = vision_metadata['output_slices']
      self._policy_keys = policy_keys
      self._policy_slices_list = [metadata[k]['output_slices'] for k in policy_keys]
      self.policy_output_slices = self._policy_slices_list[0]
      self._has_on_policy = any('on' in k.lower() for k in policy_keys)
      self._vision_input_names = [key for key in vision_metadata['input_shapes'] if 'img' in key]
      first_policy_meta = metadata[policy_keys[0]]
      frame_skip = derive_frame_skip(vision_metadata['input_shapes'], first_policy_meta['input_shapes'])
      self.input_queues, self.numpy_inputs = make_split_input_queues(vision_metadata['input_shapes'],
                                                                     first_policy_meta['input_shapes'],
                                                                     frame_skip, device=self.QUEUE_DEV)

    self._desire_key = next(key for key in self.numpy_inputs if key.startswith('desire'))
    self._road_key = next(key for key in self._vision_input_names if 'big' not in key)
    self._wide_key = next(key for key in self._vision_input_names if 'big' in key)
    self.frame_buf_params = dict.fromkeys(self._vision_input_names, nv12_info)

    is_20hz = bundle.is20hz if bundle else self._combined_model_type in ('split', 'multi_policy')
    if is_20hz:
      from openpilot.sunnypilot.models.split_model_constants import SplitModelConstants
      self.constants = SplitModelConstants()
    else:
      self.constants = ModelConstants()

    self.parser = Parser()
    self.prev_desire = np.zeros(self.constants.DESIRE_LEN, dtype=np.float32)

    if self.warp is not None:
      self.full_frames = {k: Tensor(np.zeros(nv12_info[3], dtype=np.uint8), device=self.WARP_DEV).contiguous().realize() for k in self._vision_input_names}
      self.warp(**{k: self.input_queues[k] for k in WARP_INPUTS}, frame=self.full_frames[self._road_key], big_frame=self.full_frames[self._wide_key])

  def warmup(self) -> None:
    dummy_size = self.frame_copy_size if self.is_run_model else self.frame_buf_params[self._road_key][3]
    dummy_frames = {k: np.zeros(dummy_size, dtype=np.uint8) for k in self._vision_input_names}
    transforms = {k: np.eye(3, dtype=np.float32) for k in [self._road_key, self._wide_key] if k}
    dummy_inputs = {k: np.zeros(v.shape, dtype=v.dtype) for k, v in self.numpy_inputs.items() if k not in ['tfm', 'big_tfm', 'prev_feat']}
    self.run(dummy_frames, transforms, dummy_inputs)
    if self.is_run_model:
      self.input_queues, self.numpy_inputs, self.frame_buffers = make_stock_input_queues(
        self.input_shapes, self.frame_skip, device=self.DEV, frame_copy_size=self.frame_copy_size)
      self.frame_views = self.frame_buffers
      self.npy = self.numpy_inputs
    else:
      for v in self.numpy_inputs.values():
        v[:] = 0
      self.full_frames.clear()
      self._blob_cache.clear()
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
    if self.is_run_model:
      for key, buf in bufs.items():
        data = buf.data if hasattr(buf, 'data') else buf
        np.copyto(self.frame_buffers[key], np.frombuffer(data, dtype=np.uint8, count=self.frame_copy_size))
    else:
      for key, buf in bufs.items():
        ptr = np.frombuffer(buf.data, dtype=np.uint8).ctypes.data
        cache_key = (key, ptr)
        if cache_key not in self._blob_cache:
          self._blob_cache[cache_key] = Tensor.from_blob(ptr, (self.frame_buf_params[key][3],), dtype='uint8', device=self.WARP_DEV)
        self.full_frames[key] = self._blob_cache[cache_key]

    desire_key = self.desire_key
    inputs[desire_key][0] = 0
    self.numpy_inputs[desire_key][:] = np.where(inputs[desire_key] - self.prev_desire > .99, inputs[desire_key], 0)
    self.prev_desire[:] = inputs[desire_key]

    for key in ('traffic_convention', 'lateral_control_params', 'action_t'):
      if key in self.numpy_inputs and key in inputs:
        self.numpy_inputs[key][:] = inputs[key]

    self.numpy_inputs['tfm'][:, :] = transforms[self._road_key].reshape(3, 3)
    self.numpy_inputs['big_tfm'][:, :] = transforms[self._wide_key].reshape(3, 3)

    if self.run_model is not None:
      outs, = self.run_model(**{k: self.input_queues[k] for k in MODELD_INPUTS})
      raw_outputs = outs
    else:
      assert self.warp is not None and self.run_policy is not None
      warped = self.warp(**{k: self.input_queues[k] for k in WARP_INPUTS}, frame=self.full_frames[self._road_key], big_frame=self.full_frames[self._wide_key])
      raw_outputs = self.run_policy(**{k: self.input_queues[k] for k in POLICY_INPUTS if k in self.input_queues}, warped=warped)

    if after_enqueue is not None:
      after_enqueue()

    if self._combined_model_type == 'supercombo':
      model_output = raw_outputs.numpy().flatten()
      if self.chestnut and not np.all(np.isfinite(model_output)):
        raise RuntimeError("model output not finite")
      sliced = {k: model_output[np.newaxis, v] for k, v in self.vision_output_slices.items()}
      outputs = self.parser.parse_outputs(sliced)
      if 'prev_feat' in self.numpy_inputs and 'hidden_state' in self.vision_output_slices:
        self.numpy_inputs['prev_feat'][:] = model_output[self.vision_output_slices['hidden_state']]
    else:
      vision_output = raw_outputs[0].numpy().flatten()
      vision_sliced = {k: vision_output[np.newaxis, v] for k, v in self.vision_output_slices.items()}
      outputs = self.parser.parse_vision_outputs(vision_sliced)

      if 'prev_feat' in self.numpy_inputs and 'hidden_state' in self.vision_output_slices:
        self.numpy_inputs['prev_feat'][:] = vision_output[self.vision_output_slices['hidden_state']]

      for i, policy_slices in enumerate(self._policy_slices_list):
        policy_output = raw_outputs[i + 1].numpy().flatten()
        policy_sliced = {k: policy_output[np.newaxis, v] for k, v in policy_slices.items()}
        parsed = self.parser.parse_policy_outputs(policy_sliced)
        if ('off' in self._policy_keys[i]
          and self._has_on_policy
          and any('plan' in self._policy_slices_list[j] for j, k in enumerate(self._policy_keys) if 'on' in k.lower())):

          parsed.pop('plan', None)

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
# bounded, for L0. A dead outlet or an absent bridge fails fast without a strike; a
# link that never trains with 12 V present skips the open (nothing to leak) and takes
# the SoC fallback -- the ui_watchdog kick stays the safety net. The wait ends 22 s
# after main() starts, so 22 + 60 (loader budget) + ~3 (small model) stays under
# ui_watchdog's 90 s load deadline: the watchdog matches ".modeld" in the cmdline,
# which only appears at setproctitle() a few ms into main(), and polls every 2 s, so
# its clock starts at or after ours. Imports (4-11 s warm, longer cold) come before
# main() and must not shrink the link budget (they did in e8b7e968, which counted
# from process start).
CHESTNUT_READY_TIMEOUT_S = 20.0
CHESTNUT_READY_END_BY_S = 22.0   # after main() entry, see above
CHESTNUT_READY_POLL_S = 0.5
CHESTNUT_F3_TIMEOUT_MS = 10000   # tinygrad's own timeout for the PCIe power write
CHESTNUT_F3_RESEND_S = 10.0      # re-send F3=1 only if the LTSSM is still in Detect this long after the last write
CHESTNUT_POWERED_MV = 5000       # helpers.CHESTNUT_POWERED_VOLTAGE
CHESTNUT_PCIE_L0 = 0x78
CHESTNUT_LOCK_DIR = "/tmp"
# VBSM_GPU_LATE: after a skipped eGPU the bridge is re-read from a thread at 1 Hz until it shows 12 V and L0;
# ui_watchdog gates its automatic kick on the ready marker so a kick never buys a second skip.
CHESTNUT_LINK_WAIT_FILE = "/dev/shm/vbsm_gpu_link_wait"
CHESTNUT_LINK_READY_FILE = "/dev/shm/vbsm_gpu_link_ready"
CHESTNUT_LATE_PROBE_S = 1.0
CHESTNUT_LATE_F3_MAX = 3         # PCIe power-on writes the late probe may spend (same resend rule as the preflight)
_last_f3 = float("-inf")
_f3_writes = 0


def proc_start_monotonic() -> float:
  """time.monotonic() at which this process started (from /proc/self/stat), so the probe budget is measured the
  way ui_watchdog measures its 90 s load deadline: from pid age, imports included. Never the wall clock: it jumps
  at the boot-time NTP sync."""
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


def exc_summary(e: BaseException) -> str:
  """one line per (sub-)exception: type@file:line: message -- tinygrad wraps the real error in an ExceptionGroup"""
  parts = []
  for sub in getattr(e, "exceptions", [e]):
    tb = traceback.extract_tb(sub.__traceback__)
    loc = f"{os.path.basename(tb[-1].filename)}:{tb[-1].lineno}" if tb else "?"
    parts.append(f"{type(sub).__name__}@{loc}: {str(sub)[:120]}")
  return " | ".join(parts)[:600]


def failed_at_flock(e: BaseException) -> bool:
  """True only if a sub-exception was raised inside tinygrad's flock_acquire (the flock itself, nothing after it)"""
  for sub in getattr(e, "exceptions", [e]):
    tb = traceback.extract_tb(sub.__traceback__)
    if tb and tb[-1].name == "flock_acquire" and "Failed to acquire lock" in str(sub):
      return True
  return False


def chestnut_raw() -> tuple[int, int, int] | None:
  """READ-ONLY (supply_mv, supply_ma, ltssm) from the bridge over usbdevfs EP0 (same reads as chestnut_power.py);
  own fd, closed in finally; None on any exception. No F3 write, no tinygrad, no flock."""
  import ctypes
  import fcntl
  import struct
  from openpilot.system.hardware.chestnut.flash import Ctrl, USBDEVFS_CONTROL, find_chestnut, open_device
  try:
    path, _, _ = find_chestnut()
    if not path:
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
  """sysfs lists the bridge (VID:PID and product string): the presence test hardwared and the watchdog use. On
  2026-09-16 the bridge was enumerated at SuperSpeed from boot but answered no control transfer for ~30 s."""
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
    path, _, _ = find_chestnut()
    if not path:
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


def chestnut_late_probe(skip_reason: str) -> None:
  """VBSM_GPU_LATE: after the eGPU was skipped, keep reading the bridge at 1 Hz (read-only usbdevfs, own fd; fcntl.ioctl
  releases the GIL, so this daemon thread cannot stall the SoC model). With 12 V and the link in Detect, write the PCIe
  power bit under the preflight's resend rule, at most CHESTNUT_LATE_F3_MAX times. Once 12 V and L0 are seen, write the
  ready marker for ui_watchdog's kick and stop. Never raises; never touches tinygrad."""
  cloudlog.bind(daemon=PROCESS_NAME)
  t0 = time.monotonic()
  reads = f3 = 0
  powered_logged = False
  try:
    with open(CHESTNUT_LINK_WAIT_FILE, "w") as f:
      f.write(f"{skip_reason} {int(time.time())}")
  except OSError:
    pass
  while True:
    try:
      time.sleep(CHESTNUT_LATE_PROBE_S)
      raw = chestnut_raw()
      reads += 1
      if raw is None:
        continue
      mv, _, lt = raw
      if mv >= CHESTNUT_POWERED_MV and not powered_logged:
        powered_logged = True
        cloudlog.event("chestnut late probe sees 12v", skip_reason=skip_reason, reads=reads, wait_s=round(time.monotonic() - t0, 1),
                       **chestnut_fields(raw))
      if mv >= CHESTNUT_POWERED_MV and lt == CHESTNUT_PCIE_L0:
        try:
          with open(CHESTNUT_LINK_READY_FILE, "w") as f:
            f.write(f"{skip_reason} {int(time.time())}")
        except OSError:
          pass
        cloudlog.event("chestnut link ready late", skip_reason=skip_reason, reads=reads, f3_writes=f3, f3_total=_f3_writes,
                       wait_s=round(time.monotonic() - t0, 1), **chestnut_fields(raw))
        return
      if (mv >= CHESTNUT_POWERED_MV and lt in (0x00, 0x01) and f3 < CHESTNUT_LATE_F3_MAX
          and time.monotonic() - _last_f3 >= CHESTNUT_F3_RESEND_S):
        f3 += 1
        ok = chestnut_f3_on()
        cloudlog.event("chestnut late f3", skip_reason=skip_reason, written=ok, f3_writes=f3, f3_total=_f3_writes,
                       wait_s=round(time.monotonic() - t0, 1), **chestnut_fields(raw))
    except Exception:
      cloudlog.exception("chestnut late probe error")
      return


def chestnut_fields(raw) -> dict:
  return {"supply_mv": raw[0], "supply_ma": raw[1], "ltssm": f"0x{raw[2]:02x}"} if raw else {"supply_mv": None, "supply_ma": None, "ltssm": None}


def wait_chestnut_ready(t_main: float) -> str:
  """'ready' | 'timeout' | 'no_12v' | 'absent' | 'unreadable' | 'probe_error'. Never raises. Read-first: F3=1 only with
  12 V and the link down; the preflight's write counts (f3_total), so a cold boot normally shows f3_writes=0 here and
  f3_total=1. 'absent' = six reads with no bridge in sysfs; 'unreadable' = the bridge is enumerated but answered no
  read by the deadline (2026-09-16 cold boot: enumerated at 5000 Mb/s from boot, silent for ~30 s)."""
  t0 = time.monotonic()
  reads = f3 = misses = unpowered = unreadable = 0
  raw = None
  reason = "timeout"
  try:
    deadline = min(t0 + CHESTNUT_READY_TIMEOUT_S, t_main + CHESTNUT_READY_END_BY_S)
    deadline = max(deadline, t0 + 2.0)
    while True:
      raw = chestnut_raw()
      reads += 1
      if raw is None:
        if chestnut_enumerated():
          unreadable += 1   # present on the bus, not answering: keep the whole budget, it may wake up
        else:
          misses += 1
          if misses >= 6:
            reason = "absent"
            break
      else:
        mv, _, lt = raw
        if lt == CHESTNUT_PCIE_L0:
          reason = "ready"
          break
        unpowered = unpowered + 1 if mv < CHESTNUT_POWERED_MV else 0
        if unpowered >= 3:
          reason = "no_12v"
          break
        never_written = _last_f3 == float("-inf")
        in_detect = lt in (0x00, 0x01)   # an unpowered link reads 0x00 or 0x01 (bench 2026-09-14)
        if mv >= CHESTNUT_POWERED_MV and (never_written or (in_detect and time.monotonic() - _last_f3 >= CHESTNUT_F3_RESEND_S)):
          f3 += 1
          chestnut_f3_on()
      if time.monotonic() >= deadline:
        reason = "unreadable" if raw is None and unreadable else "timeout"
        break
      time.sleep(CHESTNUT_READY_POLL_S)
  except Exception:
    cloudlog.exception("chestnut link probe error")
    return "probe_error"
  fields = dict(reason=reason, reads=reads, unreadable=unreadable, f3_writes=f3, f3_total=_f3_writes, wait_s=round(time.monotonic() - t0, 1),
                main_age_s=round(time.monotonic() - t_main, 1), pid_age_s=round(time.monotonic() - PROC_START, 1), **chestnut_fields(raw))
  if reason == "ready":
    cloudlog.event("chestnut link ready", error=False, **fields)
  else:
    cloudlog.event("chestnut link not ready", error=True, **fields)
  return reason


def main(demo=False):
  t_main = time.monotonic()   # ui_watchdog's clock starts at the setproctitle() just below (see VBSM_GPU_READY)
  cloudlog.warning("modeld init")

  sentry.set_tag("daemon", PROCESS_NAME)
  cloudlog.bind(daemon=PROCESS_NAME)
  setproctitle(PROCESS_NAME)
  config_realtime_process(7, 54)

  CHESTNUT = chestnut_present()
  # VBSM_GPU_FALLBACK: set when an eGPU load fails twice or an eGPU-active
  # modeld dies (accessory-outlet power limits); tmpfs clears it each boot.
  # With the veto the SoC path runs, and the per-hardware bundle slots make
  # ModelState(chestnut=False) select the Qualcomm slot on its own.
  if CHESTNUT and os.path.exists('/dev/shm/vbsm_usbgpu_veto'):
    cloudlog.event("eGPU vetoed earlier this boot; running SoC fallback", error=True)
    CHESTNUT = False
  if CHESTNUT:
    os.environ['HCQDEV_WAIT_TIMEOUT_MS'] = '3000'

  params = Params()
  params.put_bool("ChestnutLoading", CHESTNUT)
  params.remove("ChestnutActive")
  for stale in (CHESTNUT_LINK_WAIT_FILE, CHESTNUT_LINK_READY_FILE):   # VBSM_GPU_LATE: every process decides afresh
    try:
      os.remove(stale)
    except OSError:
      pass
  if CHESTNUT:
    # VBSM_GPU_READY preflight: record the bridge state this process starts from and, with
    # 12 V present and the link down, send the single PCIe power-on now so link training
    # overlaps the vision-stream wait below. Read-first: a warm start writes nothing.
    try:
      from tinygrad.device import Device
      raw = chestnut_raw()
      f3_written = None
      if raw is not None and raw[0] >= CHESTNUT_POWERED_MV and raw[2] != CHESTNUT_PCIE_L0:
        f3_written = chestnut_f3_on()
      cloudlog.event("chestnut preflight", dev=os.environ.get("DEV"), amd_opened=("AMD" in Device._opened_devices),
                     lock_fds=len(chestnut_lock_fds()), pid_age_s=round(time.monotonic() - PROC_START, 1), f3_written=f3_written,
                     ltssm_after=(f"0x{r[2]:02x}" if f3_written and (r := chestnut_raw()) else None), error=False, **chestnut_fields(raw))
    except Exception:
      cloudlog.exception("chestnut preflight failed")

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
      # VBSM_GPU_PPT: cap package power BEFORE the 1.7GB transfer -- the
      # accessory-outlet 12V path (~0.45 ohm measured) browns out at stock
      # boost transients, during load as well as inference. 80W ~ 6A fits.
      # Tune via /data/vbsm_gpu_ppt_w (watts; 0 disables), clamp 40..220.
      limit_w = 80
      try:
        with open("/data/vbsm_gpu_ppt_w") as f:
          limit_w = int(f.read().strip())
      except (OSError, ValueError):
        pass
      if limit_w <= 0:
        return
      limit_w = max(40, min(220, limit_w))
      from tinygrad.device import Device
      t_open = time.monotonic()
      smu = Device["AMD"].iface.dev_impl.smu
      open_ms = int((time.monotonic() - t_open) * 1000)
      smu._send_msg(smu.smu_mod.PPSMC_MSG_SetPptLimit, limit_w, timeout=100)
      applied = smu._send_msg(smu.smu_mod.PPSMC_MSG_GetPptLimit, 0, read_back_arg=True, timeout=100)
      cloudlog.event("chestnut ppt limit", requested=limit_w, applied=int(applied), open_ms=open_ms, error=False)

    # VBSM_GPU_FALLBACK: load into a separate name so a late-completing or
    # wedged loader thread can never clobber state after the timeout fires
    big_model = None
    # VBSM_GPU_LOCK_RETRY (revised 2026-09-14): retry only a failure raised inside
    # tinygrad's flock itself while this process held no lock fd beforehand -- with the
    # DEV pin that means an external holder (the unexplained 2026-09-13 case). Any
    # failure after the flock leaks the flock and the libusb claim and is never
    # retried (verified: "Resource busy" on every later open in the process).
    GPU_LOCK_RETRIES = 2
    GPU_LOCK_RETRY_S = 2.0

    def _log_lock_holder(attempt: int) -> None:
      # root-visible: an unprivileged fuser cannot see a root process's fds
      try:
        import glob as _glob
        import subprocess as _sp
        for lk in _glob.glob(f"{CHESTNUT_LOCK_DIR}/am_usb:*.lock"):
          r = _sp.run(["sudo", "fuser", "-v", lk], capture_output=True, text=True, timeout=5)
          cloudlog.event("eGPU lock holder", lock=lk, attempt=attempt, fuser=(r.stdout + r.stderr)[:500], error=True)
      except Exception:
        pass

    def load_big():
      nonlocal big_model
      cloudlog.bind(daemon=PROCESS_NAME)   # the daemon tag is thread-local; without this the loader's lines carry none
      for attempt in range(1, GPU_LOCK_RETRIES + 1):
        fds_before = chestnut_lock_fds()
        t_attempt = time.monotonic()
        try:
          apply_ppt_cap()
          m = ModelState(cam_w=vipc_client_main.width, cam_h=vipc_client_main.height, chestnut=True)
          m.warmup()
          big_model = m
          if attempt > 1:
            cloudlog.event("eGPU load succeeded after lock retry", attempts=attempt, error=False)
          return
        except Exception as e:
          fds_after = chestnut_lock_fds()
          at_flock = failed_at_flock(e)
          cloudlog.event("eGPU open failed", attempt=attempt, at_flock=at_flock, lock_fds_before=len(fds_before), lock_fds_after=len(fds_after),
                         elapsed_ms=int((time.monotonic() - t_attempt) * 1000), summary=exc_summary(e), error=True)
          if at_flock:
            _log_lock_holder(attempt)
            if not fds_before and attempt < GPU_LOCK_RETRIES:
              for fd in set(fds_after) - set(fds_before):   # opened before the flock, hold nothing, no libusb handle yet
                try:
                  os.close(fd)
                except OSError:
                  pass
              cloudlog.warning(f"eGPU lock held elsewhere, retry {attempt}/{GPU_LOCK_RETRIES - 1} in {GPU_LOCK_RETRY_S:.0f}s")
              time.sleep(GPU_LOCK_RETRY_S)
              continue
          cloudlog.exception("chestnut load failed")
          return
    loader = threading.Thread(target=load_big, daemon=True)
    # VBSM_GPU_READY: probe in the main thread so the 60 s loader budget still covers
    # the load alone; an unstarted loader reads as "finished, no model"
    ready_reason = wait_chestnut_ready(t_main)
    if ready_reason in ("ready", "probe_error"):
      loader.start()
      loader.join(BIG_MODEL_TIMEOUT)
    else:
      cloudlog.event("eGPU load skipped; link not ready", reason=ready_reason, error=True)   # nothing opened: no lock, no claim
    model = big_model
    if model is None and ready_reason in ("no_12v", "absent", "unreadable", "timeout"):
      # VBSM_GPU_LATE: keep watching the bridge; ui_watchdog kicks only once it reports 12 V + L0
      threading.Thread(target=chestnut_late_probe, args=(ready_reason,), daemon=True).start()
    if model is None and ready_reason in ("no_12v", "absent", "unreadable"):
      # nothing was attempted: no strike toward the boot-scoped veto; the kick may retry later
      params.put_bool("ChestnutModelError", True)
    elif model is None:
      # VBSM_GPU_FALLBACK: count the failure so a persistently failing load
      # cannot be retried forever by ui_watchdog's GPU kick -- the second
      # failure this boot vetoes the eGPU. A loader still alive here is wedged
      # inside a C call (field-verified: it ignored SIGINT and needed SIGKILL);
      # nothing in this process is safe behind it, so replace the process. A
      # loader that finished and failed cleanly falls through to upstream's
      # in-process small-model path: no restart, no manager budget spent.
      fails = 1
      try:
        with open('/dev/shm/vbsm_gpu_load_fails') as f:
          fails = int(f.read().strip()) + 1
      except (OSError, ValueError):
        pass
      try:
        with open('/dev/shm/vbsm_gpu_load_fails', 'w') as f:
          f.write(str(fails))
      except OSError:
        pass
      if fails >= 2:
        try:
          with open('/dev/shm/vbsm_usbgpu_veto', 'w') as f:
            f.write("load")
        except OSError:
          pass
      if loader.is_alive():
        cloudlog.event("eGPU load wedged; exiting for respawn", error=True, attempt=fails, vetoed=bool(fails >= 2))
        # Params.put is write+rename, so these survive os._exit: selfdrived's
        # bigModelFailed input stays deterministic across the respawn window
        params.put_bool("ChestnutModelError", True)
        params.put_bool("ChestnutActive", False)
        time.sleep(0.2)
        os._exit(1)
      cloudlog.event("eGPU load failed; running SoC model in-process", error=True, attempt=fails, vetoed=bool(fails >= 2))
      params.put_bool("ChestnutModelError", True)
    params.put_bool("ChestnutActive", model is not None)
    if model is not None:
      params.remove("ChestnutModelError")
      # a good load clears the ladder: the counter was never reset on success,
      # so a stale count could turn the next transient into a boot-long veto
      try:
        os.remove('/dev/shm/vbsm_gpu_load_fails')
      except OSError:
        pass

  small_model = ModelState(cam_w=vipc_client_main.width, cam_h=vipc_client_main.height, chestnut=False) if model is None or CHESTNUT else None
  if model is None:
    model = small_model
  params.put_bool("ChestnutLoading", False)
  assert model is not None
  cloudlog.warning(f"models loaded in {time.monotonic() - st:.1f}s, modeld starting")

  # messaging
  pub_socks = ["modelV2", "drivingModelData", "cameraOdometry", "modelDataV2SP"] + (["chestnutState"] if CHESTNUT else [])
  pm = PubMaster(pub_socks)
  sm = SubMaster(["deviceState", "carState", "narrowRoadCameraState", "extrinsicsCalibration", "driverMonitoringState", "carControl", "lateralDelay"])

  publish_state = PublishState()
  chestnut_state = ChestnutState(pm, model.chestnut) if CHESTNUT else None

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
                       run_count % round(model.constants.MODEL_FREQ / SERVICE_LIST['chestnutState'].frequency) == 0)
      model_output = model.run(bufs, transforms, inputs, chestnut_state.send if send_chestnut else None)
    except Exception as e:
      if not params.get_bool("ChestnutActive"):
        raise
      # VBSM_GPU_FALLBACK: upstream swaps to the already-loaded small model
      # in-process -- no restart and no no-model window, which retires the
      # fork's exit-for-respawn path (30.7s of no model on 2026-09-01). The
      # fork's veto bookkeeping stays: ui_watchdog scopes the veto to the
      # drive, retries the eGPU once the rail recovers, and any later respawn
      # must not re-attempt a browned-out GPU. The payload records what raised
      # because run() spans BOTH devices (warp on QCOM, policy on AMD): a USB
      # bulk I/O error from a bad cable was diagnosed from exactly this string.
      hangs = 1
      try:
        with open('/dev/shm/vbsm_gpu_hangs') as f:
          hangs = int(f.read().strip()) + 1
      except (OSError, ValueError):
        pass
      try:
        with open('/dev/shm/vbsm_gpu_hangs', 'w') as f:
          f.write(str(hangs))
      except OSError:
        pass
      vetoed = True
      try:
        with open('/dev/shm/vbsm_usbgpu_veto', 'w') as f:
          f.write(f"hang {type(e).__name__}: {e}"[:200])
      except OSError as ve:
        vetoed = False
        cloudlog.error(f"eGPU veto write failed, a respawn may retry the GPU: {ve}")
      cloudlog.exception(f"chestnut failed, falling back to small (vetoed={vetoed}, strike {hangs})")
      params.put_bool("ChestnutModelError", True)
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
  except Exception:
    sentry.capture_exception()
    raise
