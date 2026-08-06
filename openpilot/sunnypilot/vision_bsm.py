#!/usr/bin/env python3
import time

import numpy as np

from msgq.visionipc import VisionIpcClient, VisionStreamType
from PIL import Image, ImageDraw

from openpilot.cereal import messaging
from openpilot.common.params import Params
from openpilot.common.realtime import Priority, config_realtime_process
from openpilot.common.swaglog import cloudlog

# VisionBsmZones format: {"left": [[[x, y], ...], ...], "right": [[[x, y], ...], ...]}
# where each side holds a list of polygons (3+ vertices, coordinates normalized
# 0..1 of the driver camera frame), one per glass pane. Each polygon is scored
# independently and a side triggers when any of its polygons detects a vehicle.
# REFERENCE_ZONE watches static cabin interior: when it deviates too, the change
# is global (exposure shift, tunnel, dawn) and backgrounds rebase instead of
# detecting; the first WARMUP_FRAMES after camera connect are learn-only.

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
TOGGLE_CHECK_TIME = 5.0
WARMUP_FRAMES = 20


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

  def update(self, y_plane, now, learn_only=False):
    results = [zone.detect(y_plane, learn_only) for zone in self.zones]
    if learn_only:
      self.positive_streak = 0
    else:
      self.positive_streak = self.positive_streak + 1 if any(results) else 0
      if self.positive_streak >= RAISE_FRAMES:
        self.last_positive = now
    return now - self.last_positive < HOLD_TIME


class Detector:
  def __init__(self, zones):
    self.states = {side: SideState(zones[side]) for side in ("left", "right")}
    self.reference = PolygonZone(REFERENCE_ZONE)
    self.warmup = WARMUP_FRAMES

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
    return left, right


def parse_zones(zones):
  try:
    parsed = {}
    for side in ("left", "right"):
      polygons = []
      for polygon in zones[side]:
        points = [(float(x), float(y)) for x, y in polygon]
        if len(points) < 3 or not all(0 <= x <= 1 and 0 <= y <= 1 for x, y in points):
          return None
        polygons.append(points)
      if not polygons:
        return None
      parsed[side] = polygons
    return parsed
  except (KeyError, TypeError, ValueError):
    return None


def publish(pm, left, right):
  msg = messaging.new_message('visionBlindspotSP', valid=True)
  msg.visionBlindspotSP.left = left
  msg.visionBlindspotSP.right = right
  pm.send('visionBlindspotSP', msg)


def vision_bsm_thread():
  config_realtime_process(5, Priority.CTRL_LOW)

  params = Params()
  pm = messaging.PubMaster(['visionBlindspotSP'])

  client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_DRIVER, True)
  detector = None
  connected = False
  zones = None
  frame_count = 0
  missed_frames = 0
  last_toggle_check = -TOGGLE_CHECK_TIME
  last_publish = 0.0
  left = right = False

  while True:
    now = time.monotonic()
    if now - last_toggle_check > TOGGLE_CHECK_TIME:
      new_zones = parse_zones(params.get("VisionBsmZones")) if params.get_bool("VisionBsm") else None
      if new_zones != zones:
        zones = new_zones
        detector = Detector(zones) if zones is not None else None
        left = right = False
      last_toggle_check = now

    if zones is None:
      # keep publishing so consumers see a live, all-clear signal
      if now - last_publish > 1.0 / PUBLISH_RATE:
        publish(pm, False, False)
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

    frame_count += 1
    if frame_count % FRAME_SKIP == 0:
      y_plane = np.frombuffer(buf.data[:buf.uv_offset], dtype=np.uint8).reshape((-1, buf.stride))[:buf.height, :buf.width]
      left, right = detector.process(y_plane, now)

    if now - last_publish > 1.0 / PUBLISH_RATE:
      publish(pm, left, right)
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
