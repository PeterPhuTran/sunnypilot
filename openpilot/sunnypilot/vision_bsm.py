#!/usr/bin/env python3
"""Camera blind spot monitor - build-free variant.

Detects vehicles in the blind spots using the driver facing camera, which looks
out through both rear side windows, for cars without factory BSM.

This variant deliberately avoids anything that needs compiling, so it can be
dropped into a prebuilt install: no new params (config comes from CONFIG_PATH)
and no new cereal message (state goes to STATE_PATH on tmpfs). The cereal
version of this lives on the vision-bsm branch.

Two detectors run together, because each fails where the other works:

  motion  background subtraction, every frame, effectively free. Sees anything
          that moves against the scene, so it reacts fast - but a car pacing you
          holds still relative to the window and gets absorbed into the rolling
          background, which is exactly the car you must not miss.
  model   YOLOv10n over ONNX Runtime on the CPU. Sees parked and pacing cars
          that motion cannot, and can tell a car from a shadow. Costs ~450 ms
          per zone on this device, so only one zone is checked at a time.

The model runs on the CPU on purpose. The GPU belongs to modeld at 20Hz and a
blind spot monitor must never compete with the model that drives the car.

Occupants are rejected by geometry, not by blanking pixels. The crop around a
window also contains the driver, whom the model reports as a person in nearly
every frame, and the car's own interior, which it reports as a car. Blanking
everything outside the traced glass removes both, but it also removes the
surroundings the model needs to recognise a car at all - measured on real
frames, masking took a confident 0.71 detection down to nothing. So the model
gets the full crop with generous context, and a detection only counts when
enough of its box lies on the glass. On recorded frames that dropped 41 of 41
occupant boxes and every ego-car box (all at 8% inside or less) while keeping
the real traffic, which sits at 55-83%.

CONFIG_PATH: {"enabled": bool, "camera_view": bool,
              "zones": {"left": [[[x,y],...],...], "right": [...]}}
  coordinates normalized 0..1 of the driver camera frame, one polygon per pane.
STATE_PATH:  {"left": bool, "right": bool, "night": bool, "model": bool,
              "ts": <CLOCK_BOOTTIME seconds>}
"""
import json
import os
import sys
import threading
import time

import numpy as np

from msgq.visionipc import VisionIpcClient, VisionStreamType
from PIL import Image, ImageDraw

import openpilot.cereal.messaging as messaging
from openpilot.common.realtime import Priority, config_realtime_process
from openpilot.common.swaglog import cloudlog

CONFIG_PATH = "/data/vision_bsm.json"
STATE_PATH = "/dev/shm/vision_bsm_state"

# onnxruntime lives under /data so an AGNOS update cannot remove it. Appended,
# not inserted, so the system numpy stays ahead of the copy pip pulled in.
SITE_PACKAGES = "/data/vbsm/site-packages"
MODEL_PATH = "/data/vbsm_models/yolov10n.onnx"

BACKGROUND_ALPHA = 0.1
BRIGHT_FRACTION_THRESHOLD = 0.10
BRIGHT_VALUE = 210
DEVIATION_THRESHOLD = 0.15
DOWNSAMPLE = 4
FRAME_SKIP = 4
HOLD_TIME = 1.0
NIGHT_LEVEL = 60
PUBLISH_RATE = 5.0
RAISE_FRAMES = 3
RECONNECT_TIMEOUT = 50
REFERENCE_ZONE = [(0.36, 0.42), (0.60, 0.42), (0.60, 0.62), (0.36, 0.62)]
CONFIG_CHECK_TIME = 5.0
WARMUP_FRAMES = 20

# model
MODEL_NET = 640
MODEL_THREADS = 2
MODEL_SCORE = 0.25
MODEL_HOLD = 2.5           # a confirmed vehicle stays raised this long
# Generous context around the window: the model scores a car in the next lane
# roughly three times higher at 0.30 than at 0.06, because it can see what
# surrounds the car rather than just the car.
MODEL_PAD = 0.30
MODEL_GREY = 114
# Fraction of a detection box that must lie on the traced glass to count. Real
# traffic measured 55-83%; the driver and the ego car's own interior, which the
# model reports as a person and a car respectively, never exceeded 8%.
MODEL_INSIDE = 0.30
# how long continued movement keeps a confirmed car up for, and the ceiling on
# that. Without a ceiling the warning latches: motion is active on ~74% of left
# frames, so scenery alone would keep re-extending a car that has long gone.
MOTION_EXTEND = 1.2
MAX_HOLD = 4.0
FUSION_DEFAULT = "model"        # "model" | "either" | "motion"
# how often a zone may be checked, by what the driver is doing
MODEL_INTERVAL_SIGNALLING = 0.5
MODEL_INTERVAL_ACTIVE = 1.5
MODEL_INTERVAL_IDLE = 3.0
# COCO ids
VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PERSON_CLASS = 0


class PolygonZone:
  def __init__(self, points):
    self.points = points
    self.frame_shape = None
    self.mask = None
    self.background = None

  def prepare(self, frame_shape):
    if frame_shape == self.frame_shape:
      return
    height, width = frame_shape
    mask_img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask_img).polygon([(x * width, y * height) for x, y in self.points], fill=1)
    self.mask = np.array(mask_img, dtype=bool)[::DOWNSAMPLE, ::DOWNSAMPLE]
    self.frame_shape = frame_shape
    self.background = None

  def detect(self, y_plane, learn_only=False):
    self.prepare(y_plane.shape)
    zone = y_plane[::DOWNSAMPLE, ::DOWNSAMPLE][self.mask].astype(np.float32)
    if zone.size == 0:
      return False

    if learn_only or self.background is None or self.background.shape != zone.shape:
      self.background = zone
      return False

    deviation = float(np.mean(np.abs(zone - self.background))) / (float(np.mean(self.background)) + 1.0)
    bright_fraction = float(np.mean(zone > BRIGHT_VALUE))
    night = float(np.median(self.background)) < NIGHT_LEVEL
    detected = deviation > DEVIATION_THRESHOLD or (night and bright_fraction > BRIGHT_FRACTION_THRESHOLD)
    if not detected:
      self.background = (1 - BACKGROUND_ALPHA) * self.background + BACKGROUND_ALPHA * zone
    return detected


class SideState:
  def __init__(self, polygons):
    self.zones = [PolygonZone(points) for points in polygons]
    self.positive_streak = 0
    self.last_positive = -HOLD_TIME
    # model
    self.last_model_run = -1e9
    self.last_model_hit = -1e9
    self.last_model_result = -1e9
    self.occluded = False
    self.hold_until = -1e9
    self.model_runs = 0
    self.model_hits = 0

  def update(self, y_plane, now, learn_only=False):
    results = [zone.detect(y_plane, learn_only) for zone in self.zones]
    if learn_only:
      self.positive_streak = 0
    else:
      self.positive_streak = self.positive_streak + 1 if any(results) else 0
      if self.positive_streak >= RAISE_FRAMES:
        self.last_positive = now
    return now - self.last_positive < HOLD_TIME


def polygon_bbox(polygons, pad=MODEL_PAD):
  points = [p for polygon in polygons for p in polygon]
  xs = [p[0] for p in points]
  ys = [p[1] for p in points]
  return (max(0.0, min(xs) - pad), max(0.0, min(ys) - pad),
          min(1.0, max(xs) + pad), min(1.0, max(ys) + pad))


def nv12_to_rgb(buf, box):
  """RGB crop of an NV12 buffer, without converting the whole 1928x1208 frame."""
  x0, y0, x1, y1 = box
  # chroma is subsampled 2x, so align the crop to even pixels
  x0 &= ~1
  y0 &= ~1
  x1 = min(buf.width, (x1 + 1) & ~1)
  y1 = min(buf.height, (y1 + 1) & ~1)
  if x1 - x0 < 2 or y1 - y0 < 2:
    return None

  data = np.frombuffer(buf.data, dtype=np.uint8)
  y = data[:buf.uv_offset].reshape((-1, buf.stride))[y0:y1, x0:x1].astype(np.float32)
  uv = data[buf.uv_offset:].reshape((-1, buf.stride))[y0 // 2:y1 // 2, x0:x1]
  u = uv[:, 0::2].astype(np.float32) - 128.0
  v = uv[:, 1::2].astype(np.float32) - 128.0
  # nearest neighbour upsample back to luma resolution
  u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)[:y.shape[0], :y.shape[1]]
  v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)[:y.shape[0], :y.shape[1]]

  rgb = np.empty(y.shape + (3,), dtype=np.float32)
  rgb[..., 0] = y + 1.402 * v
  rgb[..., 1] = y - 0.344136 * u - 0.714136 * v
  rgb[..., 2] = y + 1.772 * u
  return np.clip(rgb, 0, 255).astype(np.uint8)


class ModelDetector:
  """YOLOv10n on the CPU. Absent or broken, the daemon runs on motion alone."""

  def __init__(self, model_path=MODEL_PATH):
    self.session = None
    self.input_name = None
    self.masks = {}
    if SITE_PACKAGES not in sys.path:
      sys.path.append(SITE_PACKAGES)
    try:
      import onnxruntime as ort
      opts = ort.SessionOptions()
      opts.intra_op_num_threads = MODEL_THREADS
      opts.inter_op_num_threads = 1
      # ORT_ENABLE_ALL fuses a QuickGelu kernel that throws "GetElementType is
      # not implemented" on this build, and a throwing inference is a missed
      # car. EXTENDED keeps the optimizations that matter without that fusion.
      opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
      self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
      self.input_name = self.session.get_inputs()[0].name
      cloudlog.info(f"vision_bsm: model loaded from {model_path}")
    except Exception:
      cloudlog.exception("vision_bsm: model unavailable, falling back to motion only")
      self.session = None

  @property
  def available(self):
    return self.session is not None

  def _mask_for(self, side, polygons, box, shape, frame_size):
    """Cached polygon mask in crop pixels, so occupants never reach the model."""
    key = (side, shape, box, frame_size)
    mask = self.masks.get(key)
    if mask is None:
      frame_width, frame_height = frame_size
      height, width = shape
      img = Image.new("L", (width, height), 0)
      draw = ImageDraw.Draw(img)
      for polygon in polygons:
        draw.polygon([((px * frame_width) - box[0], (py * frame_height) - box[1])
                      for px, py in polygon], fill=255)
      mask = np.array(img, dtype=bool)
      self.masks[key] = mask
      # geometry only changes when the zones are re-traced, so never grow past
      # the two sides currently in play
      if len(self.masks) > 4:
        self.masks = {key: mask}
    return mask

  @staticmethod
  def _inside_fraction(mask, box):
    """How much of a detection box lands on the traced glass."""
    height, width = mask.shape
    x0 = int(max(0, min(width - 1, box[0])))
    y0 = int(max(0, min(height - 1, box[1])))
    x1 = int(max(1, min(width, box[2])))
    y1 = int(max(1, min(height, box[3])))
    if x1 <= x0 or y1 <= y0:
      return 0.0
    return float(mask[y0:y1, x0:x1].mean())

  def prepare(self, buf, side, polygons):
    """Cut the crop out of the shared camera buffer, on the sampling thread.

    This has to happen here rather than in the worker: the VisionIPC buffer is
    recycled as soon as the next frame arrives, so handing one to another thread
    would race. nv12_to_rgb copies, so what the worker gets is its own.
    """
    x0, y0, x1, y1 = polygon_bbox(polygons)
    box = (int(x0 * buf.width), int(y0 * buf.height),
           int(x1 * buf.width), int(y1 * buf.height))
    crop = nv12_to_rgb(buf, box)
    if crop is None:
      return None, None
    mask = self._mask_for(side, polygons, box, crop.shape[:2], (buf.width, buf.height))
    return crop, mask

  def detect(self, buf, side, polygons):
    """(vehicle_seen, person_on_the_glass) for one window - synchronous."""
    crop, mask = self.prepare(buf, side, polygons)
    if crop is None:
      return False, False
    return self.infer(crop, mask)

  def infer(self, crop, mask):
    """(vehicle_seen, person_on_the_glass); safe to call off the main thread."""
    img = Image.fromarray(crop)
    scale = min(MODEL_NET / img.width, MODEL_NET / img.height)
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    offset = ((MODEL_NET - new_size[0]) // 2, (MODEL_NET - new_size[1]) // 2)
    canvas = Image.new("RGB", (MODEL_NET, MODEL_NET), (MODEL_GREY,) * 3)
    canvas.paste(img.resize(new_size, Image.BILINEAR), offset)

    # transpose only restrides the view; some fused kernels mishandle a
    # non-contiguous input, so hand the runtime a real contiguous buffer
    arr = np.ascontiguousarray(
        (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None])
    # YOLOv10 is NMS free: (1, 300, 6) of x0,y0,x1,y1,score,class, score sorted
    out = self.session.run(None, {self.input_name: arr})[0][0]

    vehicle = person = False
    for det in out:
      score = float(det[4])
      if score < MODEL_SCORE:
        break                                  # sorted, so nothing below matters
      cls = int(det[5])
      if cls not in VEHICLE_CLASSES and cls != PERSON_CLASS:
        continue
      # letterboxed network pixels back to crop pixels
      det_box = ((det[0] - offset[0]) / scale, (det[1] - offset[1]) / scale,
                 (det[2] - offset[0]) / scale, (det[3] - offset[1]) / scale)
      if self._inside_fraction(mask, det_box) < MODEL_INSIDE:
        continue                               # the driver, or our own bodywork
      if cls == PERSON_CLASS:
        person = True
      else:
        vehicle = True
    return vehicle, person


class ModelWorker:
  """Runs the model off the sampling loop.

  An inference takes about half a second. Doing it inline would stall frame
  sampling for that long, dropping the motion detector from 5Hz to roughly 2Hz
  during a lane change - precisely when it is most needed. The worker takes the
  latest crop and leaves the loop free.
  """

  def __init__(self, model):
    self.model = model
    self.lock = threading.Lock()
    self.pending = None                  # (side, buf) - only the newest matters
    self.wake = threading.Event()
    self.results = {}                    # side -> (vehicle, person, timestamp)
    self.busy = False
    self.running = True
    self.thread = threading.Thread(target=self._run, daemon=True)
    self.thread.start()

  def stop(self):
    """Let an in-flight inference finish before the session is torn down.

    Killing the thread mid-run takes the process down with
    'terminate called without an active exception'.
    """
    self.running = False
    self.wake.set()
    self.thread.join(timeout=5.0)

  def submit(self, side, crop, mask):
    with self.lock:
      if self.busy:
        return False
      self.pending = (side, crop, mask)
      self.busy = True
    self.wake.set()
    return True

  def take(self):
    with self.lock:
      return dict(self.results)

  def _run(self):
    while self.running:
      self.wake.wait()
      self.wake.clear()
      if not self.running:
        break
      with self.lock:
        job = self.pending
        self.pending = None
      if job is None:
        with self.lock:
          self.busy = False
        continue

      side, crop, mask = job
      try:
        vehicle, person = self.model.infer(crop, mask)
      except Exception:
        cloudlog.exception("vision_bsm: model inference failed")
        vehicle = person = False
      with self.lock:
        self.results[side] = (vehicle, person, time.monotonic())
        self.busy = False


class Detector:
  def __init__(self, zones, model=None, fusion=FUSION_DEFAULT):
    self.polygons = zones
    self.fusion = fusion if fusion in ("model", "either", "motion") else FUSION_DEFAULT
    self.states = {side: SideState(zones[side]) for side in ("left", "right")}
    self.reference = PolygonZone(REFERENCE_ZONE)
    self.warmup = WARMUP_FRAMES
    self.night = False
    self.model = model
    self.worker = ModelWorker(model) if model is not None and model.available else None
    self.next_side = "left"

  def update_night(self):
    """True when the windows are too dark to detect through.

    The IR illuminators light the cabin, not the world outside the glass, so
    after dark the zones are lit only by street lights and headlights.
    """
    levels = [float(np.median(z.background)) for s in self.states.values()
              for z in s.zones if z.background is not None and z.background.size]
    if levels:
      self.night = max(levels) < NIGHT_LEVEL

  def _model_interval(self, side, blinkers):
    if blinkers.get(side):
      return MODEL_INTERVAL_SIGNALLING
    if any(blinkers.values()):
      return MODEL_INTERVAL_IDLE     # the other side is what matters right now
    return MODEL_INTERVAL_ACTIVE

  def run_model(self, buf, now, blinkers):
    """Collect finished results and queue at most one new zone."""
    if self.worker is None:
      return

    for side, (vehicle, person, done_at) in self.worker.take().items():
      state = self.states[side]
      if done_at <= state.last_model_result:
        continue                            # already folded this one in
      state.last_model_result = done_at
      state.occluded = person
      state.model_runs += 1
      if vehicle:
        state.last_model_hit = done_at
        state.model_hits += 1

    # the side being signalled always wins, otherwise alternate
    order = [s for s in ("left", "right") if blinkers.get(s)]
    order += [self.next_side, "left" if self.next_side == "right" else "right"]

    for side in order:
      state = self.states[side]
      if now - state.last_model_run < self._model_interval(side, blinkers):
        continue
      crop, mask = self.model.prepare(buf, side, self.polygons[side])
      if crop is None:
        return
      if self.worker.submit(side, crop, mask):
        state.last_model_run = now
        self.next_side = "left" if side == "right" else "right"
      return

  def process(self, y_plane, now):
    if self.warmup > 0:
      self.warmup -= 1
      self.reference.detect(y_plane, learn_only=True)
      learn_only = True
    else:
      learn_only = self.reference.detect(y_plane)
      if learn_only:
        self.reference.detect(y_plane, learn_only=True)
    left = self.states["left"].update(y_plane, now, learn_only=learn_only)
    right = self.states["right"].update(y_plane, now, learn_only=learn_only)
    self.update_night()
    return left, right

  def combine(self, side, motion, now):
    """The model decides whether there is a car; motion decides how long.

    Motion is not allowed to raise the warning on its own. Replayed against a
    real drive it fired on 73% of left frames - scenery streaming past a window
    looks the same to background subtraction as a car does, and a warning that
    is on three quarters of the time tells the driver nothing.

    What motion is good for is persistence. Once the model has confirmed a car,
    continued movement in that window means it is still there, so motion
    refreshes the hold and the warning stays up between model runs instead of
    flickering at the model's 0.5-1.5s cadence.
    """
    state = self.states[side]
    if self.fusion == "motion":
      return bool(motion and not state.occluded)

    confirmed = now - state.last_model_hit < MODEL_HOLD
    if confirmed:
      state.hold_until = max(state.hold_until, state.last_model_hit + MODEL_HOLD)
    if self.fusion == "either":
      return bool(confirmed or (motion and not state.occluded))

    if confirmed and motion:
      # extend, but never past MAX_HOLD after the model actually saw the car
      state.hold_until = min(max(state.hold_until, now + MOTION_EXTEND),
                             state.last_model_hit + MAX_HOLD)
    return bool(now < state.hold_until)


def read_config():
  try:
    with open(CONFIG_PATH) as f:
      config = json.load(f)
  except (OSError, ValueError):
    return None
  if not config.get("enabled"):
    return None

  try:
    zones = {}
    for side in ("left", "right"):
      polygons = []
      for polygon in config["zones"][side]:
        points = [(float(x), float(y)) for x, y in polygon]
        if len(points) < 3 or not all(0 <= x <= 1 and 0 <= y <= 1 for x, y in points):
          return None
        polygons.append(points)
      if not polygons:
        return None
      zones[side] = polygons
    return {"zones": zones, "fusion": config.get("fusion", FUSION_DEFAULT)}
  except (KeyError, TypeError, ValueError):
    return None


def publish(left, right, night=False, model=False):
  state = {"left": bool(left), "right": bool(right), "night": bool(night),
           "model": bool(model), "ts": time.clock_gettime(time.CLOCK_BOOTTIME)}
  # write-then-rename so a reader never sees a half written file
  tmp = STATE_PATH + ".tmp"
  with open(tmp, "w") as f:
    json.dump(state, f)
  os.replace(tmp, STATE_PATH)


def vision_bsm_thread():
  # Pinning to a big core fails with EINVAL whenever that core is offline, which
  # is how the device sits while parked. Not worth dying over: run unpinned.
  try:
    config_realtime_process(5, Priority.CTRL_LOW)
  except OSError:
    cloudlog.warning("vision_bsm: could not pin to core 5, running unpinned")

  client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_DRIVER, True)
  sm = messaging.SubMaster(['carState'])
  model = ModelDetector()
  detector = None
  connected = False
  config = None
  frame_count = 0
  missed_frames = 0
  last_config_check = -CONFIG_CHECK_TIME
  last_publish = 0.0
  left = right = False

  while True:
    now = time.monotonic()
    if now - last_config_check > CONFIG_CHECK_TIME:
      new_config = read_config()
      if new_config != config:
        config = new_config
        if config is None:
          detector = None
        else:
          # without the model nothing would ever raise a warning in "model"
          # mode, so fall back to motion rather than going silently blind
          fusion = config["fusion"]
          if not model.available and fusion == "model":
            fusion = "motion"
            cloudlog.warning("vision_bsm: no model, falling back to motion only")
          detector = Detector(config["zones"], model, fusion)
          cloudlog.info(f"vision_bsm: zones loaded, fusion={fusion}")
        left = right = False
      last_config_check = now

    if config is None:
      if now - last_publish > 1.0 / PUBLISH_RATE:
        publish(False, False)
        last_publish = now
      time.sleep(1.0 / PUBLISH_RATE)
      continue

    if not connected:
      if not client.connect(False):
        time.sleep(1)
        continue
      connected = True
      detector.warmup = WARMUP_FRAMES

    buf = client.recv()
    if buf is None:
      missed_frames += 1
      if missed_frames > RECONNECT_TIMEOUT:
        client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_DRIVER, True)
        connected = False
        missed_frames = 0
      continue
    missed_frames = 0

    # SubMaster hands back a default carState until one arrives, so this is safe
    # before the car is up; both blinkers simply read False.
    sm.update(0)
    blinkers = {"left": bool(sm['carState'].leftBlinker),
                "right": bool(sm['carState'].rightBlinker)}

    frame_count += 1
    if frame_count % FRAME_SKIP == 0:
      y_plane = np.frombuffer(buf.data[:buf.uv_offset], dtype=np.uint8).reshape((-1, buf.stride))[:buf.height, :buf.width]
      motion_left, motion_right = detector.process(y_plane, now)
      detector.run_model(buf, now, blinkers)
      left = detector.combine("left", motion_left, now)
      right = detector.combine("right", motion_right, now)

    if now - last_publish > 1.0 / PUBLISH_RATE:
      publish(left, right, detector.night, model.available)
      last_publish = now


def main():
  while True:
    try:
      vision_bsm_thread()
    except Exception:
      cloudlog.exception("vision_bsm crashed")
      time.sleep(5)


if __name__ == "__main__":
  main()
