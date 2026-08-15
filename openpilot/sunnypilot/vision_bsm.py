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
          per scale on this device and runs two scales, so only one zone is
          checked at a time and the side being signalled goes first.

The model runs on the CPU on purpose. The GPU belongs to modeld at 20Hz and a
blind spot monitor must never compete with the model that drives the car.

What the model reports has to be filtered by where it is, not by blanking
pixels. The crop around a window also contains the driver, whom it reports as a
person in nearly every frame, and the car's own interior, which it reports as a
car - correctly, that is what it is. Blanking everything outside the traced
glass removes both, but it also removes the surroundings the model needs to
recognise a car at all: measured on real frames, masking took a confident 0.71
detection down to nothing.

So the model gets the full crop, and detections are rejected on geometry. Being
mostly on the glass (MODEL_INSIDE) removes the occupants, whose boxes sit at 8%
or less against real traffic's 55-83%. Not filling the whole crop
(MAX_BOX_COVERAGE) removes the cabin, which the on-glass test cannot catch
because a box covering everything overlaps the window wherever it is.

Thresholds are per side and were calibrated against 48 hand labelled windows,
because the two sides do not behave alike - see MODEL_SCORE_DEFAULT.

CONFIG_PATH: {"enabled": bool, "camera_view": bool,
              "zones": {"left": [[[x,y],...],...], "right": [...]}}
  coordinates normalized 0..1 of the driver camera frame, one polygon per pane.
STATE_PATH:  {"left": bool, "right": bool, "night": bool, "model": bool,
              "ts": <CLOCK_BOOTTIME seconds>}
"""
import ctypes
import gc
import json
import os
import sys
import threading
import time

import numpy as np

from msgq.visionipc import VisionIpcClient, VisionStreamType
from PIL import Image, ImageDraw

import openpilot.cereal.messaging as messaging
from openpilot.common.swaglog import cloudlog

CONFIG_PATH = "/data/vision_bsm.json"
STATE_PATH = "/dev/shm/vision_bsm_state"

# onnxruntime lives under /data so an AGNOS update cannot remove it. Appended,
# not inserted, so the system numpy stays ahead of the copy pip pulled in.
SITE_PACKAGES = "/data/vbsm/site-packages"
# First existing candidate wins: the fine-tuned 320 (4.0x faster on this CPU,
# held-out accuracy parity with the 640, much wider score margins), falling
# back to the stock 640 so a missing model file degrades instead of disabling.
MODEL_PATH_CANDIDATES = ("/data/vbsm_models/vbsm320.onnx",
                         "/data/vbsm_models/yolov10n.onnx")
MODEL_PATH = next((p for p in MODEL_PATH_CANDIDATES if os.path.isfile(p)),
                  MODEL_PATH_CANDIDATES[-1])

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
# Hard ceiling on this process's resident set. Steady state measures ~98 MB on
# device; if it ever passes this, something is leaking and the only safe move
# is to exit before the kernel starts shooting openpilot processes instead.
# manager restarts the daemon within seconds, so a leak costs a blink in
# detection rather than the cascade of faults it caused on 2026-08-08.
RSS_LIMIT_MB = 250
RSS_CHECK_INTERVAL = 10.0

# model
MODEL_NET = 640
MODEL_THREADS = 2
# Per side, because the two windows do not behave alike. Measured through this
# code path on 48 labelled windows, with the ego-car boxes excluded: empty left
# windows reach 0.050 while empty right windows never leave 0.000, and right
# side vehicles top out at 0.123 where left ones reach 0.78. One number cannot
# serve both - it has to thread a gap 0.009 wide - so each side gets a
# threshold with room around it.
#
# Both are low because the scores are small once the cabin is excluded; see
# MAX_BOX_COVERAGE. Override with "model_score" in the config, either a single
# number for both sides or {"left": x, "right": y}.
# Calibrated for the fine-tuned 320 against the held-out labels: left empties
# top out at 0.087 with vehicles up to 0.95; right empties are all exactly
# 0.000 with vehicles from 0.20. Margins on both sides of each threshold.
# (The old 640 model wanted {left 0.08, right 0.03}; if the daemon ever falls
# back to it, set model_score in the config accordingly.)
MODEL_SCORE_DEFAULT = {"left": 0.12, "right": 0.08}
# collect scores down to here so the daemon's own numbers can be swept against
# labels; only the threshold decides whether a warning is raised
MODEL_FLOOR = 0.02
# Floor of the adaptive hold. History: 4s produced stale-hold false tiles,
# and 1.2s once flickered - but only because the old two-scale jobs ran ~5s.
# With the 320 model a run is ~140ms and the intervals above pace the sides at
# well under a second, so 1.2s comfortably covers the cadence again, and the
# adaptive hold (HOLD_GAP_FACTOR x observed gap) still stretches it whenever
# the device is slower than the bench.
MODEL_HOLD = 1.2
# A warning needs two model hits, not one - but confirmation counts RUNS, not
# wall-clock. The first version used a time window sized to the check interval
# and produced a 46-minute drive with zero warnings: consecutive completions
# for one side are bounded by inference time plus side alternation (~1s
# signalling, ~2.2s idle), so the window could never contain two of them while
# idle. Confirm when this run and the previous run of the same side (allowing
# one missed run) both saw the car; the wall-clock cap only guards against
# confirming across genuinely stale gaps.
CONFIRM_MISS_TOLERANCE = 1
# Single-hit bypass for high-confidence detections; see the fold for why.
CONFIRM_BYPASS_SCORE = 0.45
# Pathology guard only, NOT a cadence assumption. Measured under onroad-like
# load, same-side completions arrive 5.6s apart (p90 8.1s) even for the
# signalled side - a cap below that starves confirmation, which silenced two
# whole drives. Runs-counting carries the real confirmation semantics.
CONFIRM_MAX_GAP = 12.0
# Hold scales with the observed per-side completion gap so a confirmed car
# stays warned between checks at any cadence; clamped so a stale hold stays
# bounded well under the 4s regime that produced false tiles.
HOLD_GAP_FACTOR = 1.5
HOLD_MAX = 4.0
# A single crop at 0.30 padding found less than half the cars in the labelled
# set, and a third of them scored exactly zero - invisible, not merely below
# threshold - so no amount of threshold tuning could recover them. The wide crop
# was destroying them: a car filling the window needs the tight crop, a car with
# room around it needs the wide one, and the two disagree on which cars they can
# see. Running both and taking the best is what makes the tight crop usable,
# which in turn is what makes MAX_BOX_COVERAGE necessary.
MODEL_PADS = (0.0, 0.30)
MODEL_GREY = 114
# Fraction of a detection box that must lie on the traced glass to count. Real
# traffic measured 55-83%; the driver and the ego car's own interior, which the
# model reports as a person and a car respectively, never exceeded 8%.
MODEL_INSIDE = 0.30
# The model reports the cabin we are sitting in as a car - correctly, it is one -
# with a box spanning the whole crop. On-glass filtering cannot catch that: a box
# covering everything overlaps the window wherever the window is. On the right
# side, where the window is a large share of the tight crop, it sailed through at
# up to 0.66 and drowned the real cars, which score under 0.13 there. A vehicle
# in the next lane never fills the entire frame, so anything that does is us.
MAX_BOX_COVERAGE = 0.95
# how long continued movement keeps a confirmed car up for, and the ceiling on
# that. Without a ceiling the warning latches: motion is active on ~74% of left
# frames, so scenery alone would keep re-extending a car that has long gone.
MOTION_EXTEND = 0.8
MAX_HOLD = 2.0
FUSION_DEFAULT = "model"        # "model" | "either" | "motion"
# how often a zone may be checked, by what the driver is doing
# Retuned for the fine-tuned 320: a single run measures ~140ms parked, so the
# old 0.5s floor - sized for 5s+ two-scale jobs - had become the bottleneck
# itself. Batching both sides into one inference was measured and rejected:
# batch-2 costs 2.03x batch-1 on this 2-thread CPU (no amortisation), and the
# dynamic-batch export scrambles YOLOv10's end-to-end head outputs outright.
# Sides keep alternating; these intervals just let them alternate faster.
MODEL_INTERVAL_SIGNALLING = 0.25
MODEL_INTERVAL_ACTIVE = 0.75
MODEL_INTERVAL_IDLE = 1.5
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
    self.pending_hit = -1e9
    self.runs_since_hit = 999
    self.run_gap = MODEL_HOLD           # observed same-side completion gap
    self.hold_span = MODEL_HOLD
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


def polygon_bbox(polygons, pad):
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

  # Slice both planes by their true extent, never by "the rest of the buffer".
  # Real VisionIPC buffers carry alignment padding after the UV plane, so
  # data[uv_offset:] is not a whole number of rows - reshaping it crashed on
  # every real frame, which no synthetic test buffer ever reproduced. That
  # crash looped the daemon through a fresh ONNX session every ~5s, which was
  # the 140 MB/min "leak" of 2026-08-08 23:40.
  data = np.frombuffer(buf.data, dtype=np.uint8)
  y_len = buf.height * buf.stride
  uv_len = (buf.height // 2) * buf.stride
  y = data[:y_len].reshape((buf.height, buf.stride))[y0:y1, x0:x1].astype(np.float32)
  uv = data[buf.uv_offset:buf.uv_offset + uv_len] \
      .reshape((buf.height // 2, buf.stride))[y0 // 2:y1 // 2, x0:x1]
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

  def __init__(self, model_path=MODEL_PATH, thresholds=None):
    self.session = None
    self.input_name = None
    self.masks = {}
    self.thresholds = dict(thresholds or MODEL_SCORE_DEFAULT)
    self.net = MODEL_NET
    self.vehicle_classes = set(VEHICLE_CLASSES)
    self.person_classes = {PERSON_CLASS}
    if SITE_PACKAGES not in sys.path:
      sys.path.append(SITE_PACKAGES)
    try:
      import onnxruntime as ort
      opts = ort.SessionOptions()
      opts.intra_op_num_threads = MODEL_THREADS
      opts.inter_op_num_threads = 1
      # The arena keeps inference workspace allocated between calls. Measured,
      # it costs 19 MB and saves nothing - 196 ms per inference either way - and
      # this device has run out of memory mid-drive, so it goes.
      opts.enable_cpu_mem_arena = False
      # ORT_ENABLE_ALL fuses a QuickGelu kernel that throws "GetElementType is
      # not implemented" on this build, and a throwing inference is a missed
      # car. EXTENDED keeps the optimizations that matter without that fusion.
      opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
      self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
      self.input_name = self.session.get_inputs()[0].name
      # The model file is the authority on its own geometry and classes.
      # Network size comes from the input shape; the class map comes from the
      # ultralytics names metadata when present (the fine-tuned model ships
      # 0=vehicle 1=person), falling back to the stock COCO ids.
      dim = self.session.get_inputs()[0].shape[2]
      # dynamic exports carry symbolic dims ("height") - int() would throw and
      # silently demote the daemon to motion-only via the except below
      self.net = int(dim) if isinstance(dim, int) else MODEL_NET
      names = {}
      try:
        import ast
        raw = self.session.get_modelmeta().custom_metadata_map.get("names", "")
        names = ast.literal_eval(raw) if raw else {}
      except (ValueError, SyntaxError):
        names = {}
      if names and "vehicle" in names.values():
        self.vehicle_classes = {i for i, n in names.items() if n == "vehicle"}
        self.person_classes = {i for i, n in names.items() if n == "person"}
      else:
        self.vehicle_classes = set(VEHICLE_CLASSES)
        self.person_classes = {PERSON_CLASS}
      cloudlog.info(f"vision_bsm: model loaded from {model_path} "
                    f"net={self.net} vehicle_classes={sorted(self.vehicle_classes)}")
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
      # one entry per side per scale
      if len(self.masks) > 2 * len(MODEL_PADS):
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

  def prepare(self, buf, side, polygons, pads=MODEL_PADS):
    """Cut the crops out of the shared camera buffer, on the sampling thread.

    This has to happen here rather than in the worker: the VisionIPC buffer is
    recycled as soon as the next frame arrives, so handing one to another thread
    would race. nv12_to_rgb copies, so what the worker gets is its own.

    Returns one (crop, mask) per padding - see MODEL_PADS for why there is more
    than one.
    """
    jobs = []
    for pad in pads:
      x0, y0, x1, y1 = polygon_bbox(polygons, pad)
      box = (int(x0 * buf.width), int(y0 * buf.height),
             int(x1 * buf.width), int(y1 * buf.height))
      crop = nv12_to_rgb(buf, box)
      if crop is None:
        continue
      mask = self._mask_for(side, polygons, box, crop.shape[:2], (buf.width, buf.height))
      jobs.append((crop, mask))
    return jobs

  def detect(self, buf, side, polygons):
    """(vehicle_seen, person_on_the_glass) for one window - synchronous."""
    jobs = self.prepare(buf, side, polygons)
    if not jobs:
      return False, False
    return self.infer(jobs, side)

  def infer(self, jobs, side):
    """(vehicle_seen, person_on_the_glass) across every scale; thread safe."""
    score, person = self.score(jobs, side)
    return score >= self.thresholds[side], person

  def score(self, jobs, side):
    """(best vehicle score on the glass, person_on_the_glass) across scales.

    Kept separate from the threshold so the daemon's own numbers can be
    measured against labels, rather than measuring a reimplementation of it and
    hoping the two agree - they did not, the first time this was checked.

    Best-of rather than agreement: the paddings are complementary, so requiring
    both would discard exactly the cars only one of them can see.
    """
    best = 0.0
    person = False
    for crop, mask in jobs:
      s, p = self._infer_one(crop, mask, side)
      best = max(best, s)
      person = person or p
    return best, person

  def _infer_one(self, crop, mask, side):
    """One crop through the model, filtered to detections on the glass."""
    img = Image.fromarray(crop)
    net = self.net
    scale = min(net / img.width, net / img.height)
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    offset = ((net - new_size[0]) // 2, (net - new_size[1]) // 2)
    canvas = Image.new("RGB", (net, net), (MODEL_GREY,) * 3)
    canvas.paste(img.resize(new_size, Image.BILINEAR), offset)

    # transpose only restrides the view; some fused kernels mishandle a
    # non-contiguous input, so hand the runtime a real contiguous buffer
    arr = np.ascontiguousarray(
        (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None])
    # YOLOv10 is NMS free: (1, 300, 6) of x0,y0,x1,y1,score,class, score sorted
    out = self.session.run(None, {self.input_name: arr})[0][0]

    best = 0.0
    person = False
    for det in out:
      score = float(det[4])
      if score < MODEL_FLOOR:
        break                                  # sorted, so nothing below matters
      cls = int(det[5])
      if cls not in self.vehicle_classes and cls not in self.person_classes:
        continue
      # letterboxed network pixels back to crop pixels
      det_box = ((det[0] - offset[0]) / scale, (det[1] - offset[1]) / scale,
                 (det[2] - offset[0]) / scale, (det[3] - offset[1]) / scale)
      area = max(0.0, det_box[2] - det_box[0]) * max(0.0, det_box[3] - det_box[1])
      if area > MAX_BOX_COVERAGE * crop.shape[0] * crop.shape[1]:
        continue                               # the car we are sitting in
      if self._inside_fraction(mask, det_box) < MODEL_INSIDE:
        continue                               # the driver, or our own bodywork
      if cls in self.person_classes:
        person = person or score >= self.thresholds[side]
      else:
        best = max(best, score)
    return best, person


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

  def submit(self, side, jobs):
    with self.lock:
      if self.busy:
        return False
      self.pending = (side, jobs)
      self.busy = True
    self.wake.set()
    return True

  def take(self):
    with self.lock:
      return dict(self.results)

  def _run(self):
    # A thread inherits the scheduling policy of whoever created it, so say
    # plainly that the expensive half of this daemon is the lowest priority
    # thing on the device. If anything else wants the CPU, it gets it.
    try:
      os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
    except (OSError, AttributeError, ValueError):
      pass
    try:
      os.nice(10)
    except OSError:
      pass

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

      side, jobs = job
      try:
        score, person = self.model.score(jobs, side)
      except Exception:
        cloudlog.exception("vision_bsm: model inference failed")
        score, person = 0.0, False
      with self.lock:
        self.results[side] = (score, person, time.monotonic())
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

    for side, (score, person, done_at) in self.worker.take().items():
      vehicle = score >= self.model.thresholds[side]
      state = self.states[side]
      if done_at <= state.last_model_result:
        continue                            # already folded this one in
      if state.last_model_result > 0:
        state.run_gap = min(10.0, done_at - state.last_model_result)
      state.last_model_result = done_at
      state.occluded = person
      state.model_runs += 1
      if vehicle:
        state.model_hits += 1
        # confirm-before-raise: this run plus the previous run of this side
        # (one miss tolerated) - a single WEAK output still cannot chime. A
        # strong one can: a fast overtaker on route 38 was scoreable for only
        # 1.2s of a signal window at 0.65-0.74, and two-hit confirmation ate
        # it. On this model empty windows measure <=0.087, so a single hit at
        # CONFIRM_BYPASS_SCORE is not phantom territory; persistent false
        # sources (parked rows) were going to confirm one run later anyway.
        if (score >= CONFIRM_BYPASS_SCORE
            or (state.runs_since_hit <= CONFIRM_MISS_TOLERANCE
                and done_at - state.pending_hit <= CONFIRM_MAX_GAP)):
          state.last_model_hit = done_at
          state.hold_span = min(HOLD_MAX, max(MODEL_HOLD, HOLD_GAP_FACTOR * state.run_gap))
        state.runs_since_hit = 0
        state.pending_hit = done_at
      else:
        state.runs_since_hit += 1

    # An inference takes ~1s and this runs at 5Hz, so most calls find the
    # worker busy. Bail before building anything: the crops cost tens of
    # megabytes of temporaries, and building them only to throw them away was
    # the bulk of the allocation churn behind the 2026-08-08 leak.
    if self.worker.busy:
      return

    # While signalling, the signalled side gets the worker EXCLUSIVELY - the
    # other window was stealing a third of the cadence exactly when latency
    # matters most. Idle, the sides alternate as before.
    signalled = [s for s in ("left", "right") if blinkers.get(s)]
    if signalled:
      order = signalled
    else:
      order = [self.next_side, "left" if self.next_side == "right" else "right"]

    for side in order:
      state = self.states[side]
      if now - state.last_model_run < self._model_interval(side, blinkers):
        continue
      # One scale per run, alternating: a two-scale job measured ~5.6s under
      # onroad-like load, and cadence is what confirmation and hold live on.
      # Consecutive runs still cover both scales inside one confirmation pair.
      state.pad_flip = getattr(state, "pad_flip", 0) ^ 1
      pads = (MODEL_PADS[state.pad_flip % len(MODEL_PADS)],)
      jobs = self.model.prepare(buf, side, self.polygons[side], pads)
      if not jobs:
        return
      if self.worker.submit(side, jobs):
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

    confirmed = now - state.last_model_hit < state.hold_span
    if confirmed:
      state.hold_until = max(state.hold_until, state.last_model_hit + state.hold_span)
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
    # accepts a single number for both sides or {"left": x, "right": y};
    # anything unparseable or out of range falls back per side
    raw = config.get("model_score", MODEL_SCORE_DEFAULT)
    thresholds = dict(MODEL_SCORE_DEFAULT)
    for side in ("left", "right"):
      value = raw.get(side) if isinstance(raw, dict) else raw
      try:
        value = float(value)
      except (TypeError, ValueError):
        continue
      if 0.0 < value < 1.0:
        thresholds[side] = value
    return {"zones": zones, "fusion": config.get("fusion", FUSION_DEFAULT),
            "model_score": thresholds}
  except (KeyError, TypeError, ValueError):
    return None


def rss_mb():
  try:
    with open("/proc/self/status") as f:
      for line in f:
        if line.startswith("VmRSS:"):
          return int(line.split()[1]) / 1024.0
  except OSError:
    pass
  return 0.0


def publish(left, right, night=False, model=False):
  state = {"left": bool(left), "right": bool(right), "night": bool(night),
           "model": bool(model), "ts": time.clock_gettime(time.CLOCK_BOOTTIME)}
  # write-then-rename so a reader never sees a half written file
  tmp = STATE_PATH + ".tmp"
  with open(tmp, "w") as f:
    json.dump(state, f)
  os.replace(tmp, STATE_PATH)


def tame_glibc():
  """Stop glibc from hoarding the temporaries this process churns through.

  The 2026-08-08 23:40 drive leaked 140 MB/min to 1.5 GB and took openpilot
  down with it, yet the same code soaked flat single-threaded. The difference
  is the worker thread: with inference on one thread and multi-megabyte
  NV12/PIL temporaries churning on another, glibc raises its dynamic mmap
  threshold, the big buffers move onto per-thread heap arenas, and freed
  chunks stay pinned there instead of returning to the kernel.

  Two knobs close that path: cap the arenas, and pin the mmap threshold so
  every multi-megabyte temporary is mmap'd and genuinely returned on free.
  """
  try:
    libc = ctypes.CDLL("libc.so.6")
    libc.mallopt(-8, 2)             # M_ARENA_MAX = 2
    libc.mallopt(-3, 128 * 1024)    # M_MMAP_THRESHOLD pinned, no dynamic growth
  except OSError:
    cloudlog.warning("vision_bsm: mallopt unavailable, relying on the RSS limit")


def vision_bsm_thread():
  tame_glibc()
  # Deliberately NOT config_realtime_process. That helper disables the garbage
  # collector, raises the process to SCHED_FIFO and pins it to one core - all
  # reasonable for the cheap background-subtraction loop this used to be, and all
  # harmful now that it runs a CNN.
  #
  # A drive reported the device running out of memory and openpilot dropping out
  # with take-control alerts. Both follow from those settings: this loop
  # allocates numpy arrays and PIL images every cycle, so a disabled collector
  # lets cycles accumulate, and two threads of 640x640 inference at realtime
  # priority on a single core preempt whatever openpilot needs on it.
  #
  # Nothing here is safety critical - it raises a warning, it does not steer - so
  # it runs at normal priority, unpinned, and yields to everything that matters.
  gc.enable()
  try:
    os.nice(10)
  except OSError:
    pass

  client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_DRIVER, True)
  sm = messaging.SubMaster(['carState'])
  # Loaded on first use, not here: the model and its runtime cost ~100 MB, and
  # building them up front meant switching the mod off in settings left every
  # byte of it resident. Off should mean off.
  model = None
  detector = None
  connected = False
  config = None
  frame_count = 0
  missed_frames = 0
  last_config_check = -CONFIG_CHECK_TIME
  last_publish = 0.0
  left = right = False

  last_rss_check = 0.0
  last_cadence_log = 0.0
  frame_errors = 0

  try:
    while True:
      now = time.monotonic()

      if now - last_rss_check > RSS_CHECK_INTERVAL:
        last_rss_check = now
        rss = rss_mb()
        if rss > RSS_LIMIT_MB:
          cloudlog.error(f"vision_bsm: RSS {rss:.0f} MB over the {RSS_LIMIT_MB} MB "
                         "limit - exiting so manager restarts us clean")
          sys.exit(1)
        if detector is not None and now - last_cadence_log > 60.0:
          last_cadence_log = now
          gaps = {s2: round(detector.states[s2].run_gap, 1) for s2 in ("left", "right")}
          runs = {s2: detector.states[s2].model_runs for s2 in ("left", "right")}
          cloudlog.info(f"vision_bsm cadence: gaps={gaps} runs={runs}")

      if now - last_config_check > CONFIG_CHECK_TIME:
        new_config = read_config()
        if new_config != config:
          config = new_config
          # a replaced Detector must take its worker thread with it, or the old
          # thread keeps the model alive from the shadows
          if detector is not None and detector.worker is not None:
            detector.worker.stop()
          if config is None:
            detector = None
          else:
            if model is None:
              model = ModelDetector()
            # without the model nothing would ever raise a warning in "model"
            # mode, so fall back to motion rather than going silently blind
            fusion = config["fusion"]
            if not model.available and fusion == "model":
              fusion = "motion"
              cloudlog.warning("vision_bsm: no model, falling back to motion only")
            model.thresholds = config["model_score"]
            detector = Detector(config["zones"], model, fusion)
            cloudlog.info(f"vision_bsm: zones loaded, fusion={fusion}, "
                          f"score={model.thresholds}")
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
        # A processing failure must cost one frame, not the process. The old
        # handler tore down and rebuilt everything - client, SubMaster, model -
        # on any exception, and when the exception recurred per frame, that
        # rebuild churn WAS the memory leak that took the device down.
        try:
          y_plane = np.frombuffer(buf.data[:buf.height * buf.stride], dtype=np.uint8) \
                      .reshape((buf.height, buf.stride))[:, :buf.width]
          motion_left, motion_right = detector.process(y_plane, now)
          detector.run_model(buf, now, blinkers)
          left = detector.combine("left", motion_left, now)
          right = detector.combine("right", motion_right, now)
        except Exception:
          frame_errors += 1
          if frame_errors in (1, 10) or frame_errors % 100 == 0:
            cloudlog.exception(f"vision_bsm: frame processing failed ({frame_errors} so far)")

      if now - last_publish > 1.0 / PUBLISH_RATE:
        publish(left, right, detector.night, model is not None and model.available)
        last_publish = now
  finally:
    # whatever ends this loop - exception, RSS exit - the worker thread and
    # the model it references must die with it, not linger into a retry
    if detector is not None and detector.worker is not None:
      detector.worker.stop()


def main():
  # One in-process retry for transient failures, then exit and let manager
  # start a fresh process. The old loop retried forever, and every retry
  # rebuilt the ONNX session while the previous worker thread kept the old one
  # alive - so a repeating exception became a ratchet that leaked a session
  # every five seconds until the kernel started killing openpilot instead.
  # A process exit cannot leak: everything dies with it.
  for attempt in (1, 2):
    try:
      vision_bsm_thread()
      return
    except Exception:
      cloudlog.exception(f"vision_bsm crashed (attempt {attempt})")
      time.sleep(5)
  cloudlog.error("vision_bsm: crashing repeatedly, exiting for a clean restart")
  sys.exit(1)


if __name__ == "__main__":
  main()
