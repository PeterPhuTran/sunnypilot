import numpy as np
import pyray as rl
from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.cereal.visionipc import VisionStreamType
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.selfdrive.ui.mici.onroad import SIDE_PANEL_WIDTH
from openpilot.selfdrive.ui.mici.onroad.alert_renderer import AlertRenderer
from openpilot.selfdrive.ui.mici.onroad.driver_state import DriverStateRenderer
from openpilot.selfdrive.ui.mici.onroad.hud_renderer import HudRenderer
from openpilot.selfdrive.ui.mici.onroad.model_renderer import ModelRenderer
from openpilot.selfdrive.ui.mici.onroad.confidence_ball import ConfidenceBall
from openpilot.selfdrive.ui.mici.onroad.cameraview import CameraView
from openpilot.system.ui.lib.application import FontWeight, gui_app, MousePos, MouseEvent
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets import Widget
from openpilot.common.filter_simple import BounceFilter
from openpilot.common.swaglog import cloudlog
import json
import os
import time
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.common.transformations.camera import DEVICE_CAMERAS, DeviceCameraConfig, view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler
from enum import IntEnum

if gui_app.sunnypilot_ui():
  from openpilot.selfdrive.ui.sunnypilot.mici.onroad.hud_renderer import HudRendererSP as HudRenderer
  from openpilot.selfdrive.ui.sunnypilot.ui_state import OnroadTimerStatus

OpState = log.SelfdriveState.OpenpilotState
CALIBRATED = log.ExtrinsicsCalibration.Status.calibrated
NARROW_ROAD_CAM = VisionStreamType.VISION_STREAM_NARROW_ROAD
WIDE_CAM = VisionStreamType.VISION_STREAM_WIDE_ROAD
DRIVER_CAM = VisionStreamType.VISION_STREAM_DRIVER
BSM_PARAM_INTERVAL = 2.0  # seconds between blind spot camera config checks
BSM_CONFIG_PATH = "/data/vision_bsm.json"
BSM_STATE_PATH = "/dev/shm/vision_bsm_state"
BSM_STATE_INTERVAL = 1.0  # seconds between night-flag checks
BSM_STATE_STALE = 3.0
BSM_NIGHT_TEXT = "low light - blind spot detection unreliable"
BSM_NIGHT_FONT_SIZE = 22
DEFAULT_DEVICE_CAMERA = DEVICE_CAMERAS["tici", "ar0231"]


class BookmarkState(IntEnum):
  HIDDEN = 0
  DRAGGING = 1
  TRIGGERED = 2

WIDE_CAM_MAX_SPEED = 5.0  # m/s (10 mph)
ROAD_CAM_MIN_SPEED = 10  # m/s (25 mph)

CAM_Y_OFFSET = 20


class BookmarkIcon(Widget):
  PEEK_THRESHOLD = 50  # If icon peeks out this much, snap it fully visible
  FULL_VISIBLE_OFFSET = 200  # How far onscreen when fully visible
  HIDDEN_OFFSET = -50  # How far offscreen when hidden

  def __init__(self, bookmark_callback):
    super().__init__()
    self._bookmark_callback = bookmark_callback
    self._icon = gui_app.texture("icons_mici/onroad/bookmark.png", 180, 180)
    self._offset_filter = BounceFilter(0.0, 0.1, 1 / gui_app.target_fps)

    # State
    self._interacting = False
    self._state = BookmarkState.HIDDEN
    self._swipe_start_x = 0.0
    self._swipe_current_x = 0.0
    self._is_swiping = False
    self._is_swiping_left: bool = False
    self._triggered_time: float = 0.0

  def is_swiping_left(self) -> bool:
    """Check if currently swiping left (for scroller to disable)."""
    return self._is_swiping_left

  def interacting(self):
    interacting, self._interacting = self._interacting, False
    return interacting

  def _update_state(self):
    if self._state == BookmarkState.DRAGGING:
      # Allow pulling past activated position with rubber band effect
      swipe_offset = self._swipe_start_x - self._swipe_current_x
      swipe_offset = min(swipe_offset, self.FULL_VISIBLE_OFFSET + 50)
      self._offset_filter.update(swipe_offset)

    elif self._state == BookmarkState.TRIGGERED:
      # Continue animating to fully visible
      self._offset_filter.update(self.FULL_VISIBLE_OFFSET)
      # Stay in TRIGGERED state for 1 second
      if rl.get_time() - self._triggered_time >= 1.5:
        self._state = BookmarkState.HIDDEN

    elif self._state == BookmarkState.HIDDEN:
      self._offset_filter.update(self.HIDDEN_OFFSET)

      if self._offset_filter.x < 1e-3:
        self._interacting = False

  def _handle_mouse_event(self, mouse_event: MouseEvent):
    if not ui_state.started:
      return

    if mouse_event.left_pressed:
      # Store relative position within widget
      self._swipe_start_x = mouse_event.pos.x
      self._swipe_current_x = mouse_event.pos.x
      self._is_swiping = True
      self._is_swiping_left = False
      self._state = BookmarkState.DRAGGING

    elif mouse_event.left_down and self._is_swiping:
      self._swipe_current_x = mouse_event.pos.x
      swipe_offset = self._swipe_start_x - self._swipe_current_x
      self._is_swiping_left = swipe_offset > 0
      if self._is_swiping_left:
        self._interacting = True

    elif mouse_event.left_released:
      if self._is_swiping:
        swipe_distance = self._swipe_start_x - self._swipe_current_x

        # If peeking past threshold, transition to animating to fully visible and bookmark
        if swipe_distance > self.PEEK_THRESHOLD:
          self._state = BookmarkState.TRIGGERED
          self._triggered_time = rl.get_time()
          self._bookmark_callback()
        else:
          # Otherwise, transition back to hidden
          self._state = BookmarkState.HIDDEN

        # Reset swipe state
        self._is_swiping = False
        self._is_swiping_left = False

  def _render(self, _):
    """Render the bookmark icon."""
    if self._offset_filter.x > 0:
      icon_x = self.rect.x + self.rect.width - round(self._offset_filter.x)
      icon_y = self.rect.y + (self.rect.height - self._icon.height) / 2  # Vertically centered
      rl.draw_texture_ex(self._icon, rl.Vector2(icon_x, icon_y), 0.0, 1.0, rl.WHITE)


class AugmentedRoadView(CameraView):
  def __init__(self, bookmark_callback=None, stream_type: VisionStreamType = VisionStreamType.VISION_STREAM_NARROW_ROAD):
    super().__init__("camerad", stream_type)
    self._bookmark_callback = bookmark_callback
    self._set_placeholder_color(rl.BLACK)

    self.device_camera: DeviceCameraConfig | None = None
    self.view_from_calib = view_frame_from_device_frame.copy()
    self.view_from_wide_calib = view_frame_from_device_frame.copy()

    self._matrix_cache_key: tuple | None = None
    self._cached_matrix: np.ndarray | None = None
    self._content_rect = rl.Rectangle()
    self._last_click_time = 0.0

    # Bookmark icon with swipe gesture
    self._bookmark_icon = BookmarkIcon(bookmark_callback)

    self._model_renderer = ModelRenderer()
    self._hud_renderer = HudRenderer()
    self._alert_renderer = AlertRenderer()
    self._driver_state_renderer = DriverStateRenderer()
    self._confidence_ball = ConfidenceBall()
    self._offroad_label = UnifiedLabel("start the car to\nuse sunnypilot", 54, FontWeight.DISPLAY,
                                       text_color=rl.Color(255, 255, 255, int(255 * 0.9)),
                                       alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER,
                                       alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE)

    self._fade_texture = gui_app.texture("icons_mici/onroad/onroad_fade.png")

    # blind spot camera view
    self._bsm_config_mtime = 0.0
    self._bsm_night = False
    self._bsm_state_checked = 0.0
    self._bsm_zone: tuple[float, float, float, float] | None = None
    self._bsm_left = True
    self._bsm_zones: dict = {}
    self._bsm_enabled = False
    self._bsm_params_checked = 0.0

  def _refresh_bsm_params(self):
    now = rl.get_time()
    if now - self._bsm_params_checked < BSM_PARAM_INTERVAL:
      return
    self._bsm_params_checked = now

    # only re-read when the file actually changed
    try:
      mtime = os.path.getmtime(BSM_CONFIG_PATH)
    except OSError:
      self._bsm_enabled = False
      self._bsm_zones = {}
      return
    if mtime == self._bsm_config_mtime:
      return
    self._bsm_config_mtime = mtime

    try:
      with open(BSM_CONFIG_PATH) as f:
        config = json.load(f)
      self._bsm_enabled = bool(config.get("enabled")) and bool(config.get("camera_view"))
      self._bsm_zones = config.get("zones", {}) if self._bsm_enabled else {}
    except (OSError, ValueError):
      self._bsm_enabled = False
      self._bsm_zones = {}

  def _bsm_bounds(self, side: str) -> tuple[float, float, float, float] | None:
    """Bounding box of a side's calibrated polygons, as (x, y, w, h) in 0..1."""
    xs, ys = [], []
    for polygon in self._bsm_zones.get(side, []):
      for point in polygon:
        if len(point) == 2:
          xs.append(float(point[0]))
          ys.append(float(point[1]))
    if not xs:
      return None
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if x1 <= x0 or y1 <= y0:
      return None
    return x0, y0, x1 - x0, y1 - y0

  def _refresh_bsm_night(self):
    now = rl.get_time()
    if now - self._bsm_state_checked < BSM_STATE_INTERVAL:
      return
    self._bsm_state_checked = now
    try:
      with open(BSM_STATE_PATH) as f:
        state = json.load(f)
      fresh = time.clock_gettime(time.CLOCK_BOOTTIME) - state.get("ts", -1e9) < BSM_STATE_STALE
      self._bsm_night = fresh and bool(state.get("night"))
    except (OSError, ValueError):
      self._bsm_night = False

  def _draw_bsm_night_notice(self, rect: rl.Rectangle):
    """Quiet caption: after dark the outside of the glass is barely lit."""
    font = gui_app.font(FontWeight.MEDIUM)
    size = measure_text_cached(font, BSM_NIGHT_TEXT, BSM_NIGHT_FONT_SIZE)
    x = rect.x + (rect.width - size.x) / 2
    y = rect.y + rect.height - size.y - 12
    rl.draw_rectangle_rounded(rl.Rectangle(x - 10, y - 5, size.x + 20, size.y + 10),
                              0.35, 8, rl.Color(0, 0, 0, 140))
    rl.draw_text_ex(font, BSM_NIGHT_TEXT, rl.Vector2(x, y), BSM_NIGHT_FONT_SIZE, 0,
                    rl.Color(255, 255, 255, 190))

  def _update_bsm(self) -> bool:
    """The signalled side's window takes the whole screen while that blinker is on."""
    self._refresh_bsm_params()
    if not self._bsm_enabled:
      self._bsm_zone = None
      return False

    # Only trust the blinker while carState is actually flowing. During the
    # memory-exhaustion drive the comms broke down, SubMaster kept returning the
    # last message it ever got, and a frozen leftBlinker=True held the driver
    # camera on screen long after the signal ended. When in doubt, show the
    # road, not the window.
    sm = ui_state.sm
    if not (sm.alive['carState'] and sm.valid['carState']):
      self._bsm_zone = None
      return False

    CS = sm['carState']
    left, right = CS.leftBlinker, CS.rightBlinker
    if left == right:  # neither, or hazards
      self._bsm_zone = None
      return False

    self._bsm_left = left
    self._bsm_zone = self._bsm_bounds("left" if left else "right")
    return self._bsm_zone is not None

  def is_swiping_left(self) -> bool:
    """Check if currently swiping left (for scroller to disable)."""
    return self._bookmark_icon.is_swiping_left()

  def _update_state(self):
    super()._update_state()

    # update offroad label
    if ui_state.panda_type == log.PandaState.PandaType.unknown:
      self._offroad_label.set_text("system booting")
    elif ui_state.ignition and not ui_state.started:
      self._offroad_label.set_text("openpilot can't start\ncheck alerts")
    else:
      self._offroad_label.set_text("start the car to\nuse sunnypilot")

  def _handle_mouse_release(self, mouse_pos: MousePos):
    # Don't trigger click callback if bookmark was triggered
    if not self._bookmark_icon.interacting():
      super()._handle_mouse_release(mouse_pos)

  def _render(self, _):
    # Draw text if not onroad
    if not ui_state.started:
      rl.draw_rectangle_rec(self.rect, rl.BLACK)
      self._offroad_label.render(self._rect)
      return

    blind_spot_view = self._update_bsm()
    self._switch_stream_if_needed(ui_state.sm)

    # Update calibration before rendering
    self._update_calibration()

    # Create inner content area with border padding
    self._content_rect = rl.Rectangle(
      self.rect.x,
      self.rect.y,
      self.rect.width - SIDE_PANEL_WIDTH,
      self.rect.height,
    )

    # Enable scissor mode to clip all rendering within content rectangle boundaries
    # This creates a rendering viewport that prevents graphics from drawing outside the border
    rl.begin_scissor_mode(
      int(self._content_rect.x),
      int(self._content_rect.y),
      int(self._content_rect.width),
      int(self._content_rect.height)
    )

    # Render the base camera view
    super()._render(self._content_rect)

    # Draw all UI overlays. The driving path makes no sense drawn over a view out
    # of a side window, but alerts still have to reach the driver.
    if blind_spot_view:
      if self._bsm_night:
        self._draw_bsm_night_notice(self._content_rect)
    else:
      self._model_renderer.render(self._content_rect)

    # Fade out bottom of overlays for looks
    rl.draw_texture_ex(self._fade_texture, rl.Vector2(self._content_rect.x, self._content_rect.y), 0.0, 1.0, rl.WHITE)

    alert_to_render, not_animating_out = self._alert_renderer.will_render()

    # Hide DMoji when disengaged unless AlwaysOnDM is enabled
    should_draw_dmoji = (not self._hud_renderer.drawing_top_icons() and
                         (ui_state.status != UIStatus.DISENGAGED or ui_state.always_on_dm))
    self._driver_state_renderer.set_should_draw(should_draw_dmoji)
    self._driver_state_renderer.set_position(self._rect.x + 16, self._rect.y + 10)
    self._driver_state_renderer.render()

    self._hud_renderer.set_can_draw_top_icons(alert_to_render is None)
    self._hud_renderer.set_wheel_critical_icon(alert_to_render is not None and not not_animating_out and
                                               alert_to_render.visual_alert == car.CarControl.HUDControl.VisualAlert.steerRequired)
    self._alert_renderer.render(self._content_rect)
    self._hud_renderer.render(self._content_rect)

    # Draw fake rounded border
    rl.draw_rectangle_rounded_lines_ex(self._content_rect, 0.2 * 1.02, 10, 50, rl.BLACK)

    # End clipping region
    rl.end_scissor_mode()

    # Custom UI extension point - add custom overlays here
    # Use self._content_rect for positioning within camera bounds
    self._confidence_ball.render(self.rect)

    self._bookmark_icon.render(self.rect)

  def _cancel_pending_switch(self):
    """Abandon an in-flight stream switch.

    The base class switch is asynchronous: it arms a target client and only
    completes when that stream delivers its first frame. Its switch_stream()
    also early-returns when asked for the stream already on screen, so a
    switch armed AWAY from the current stream cannot be redirected back - it
    has to be cancelled outright, or it completes after the reason for it has
    passed. That is exactly how the driver view kept appearing after a brief
    turn signal: the signal ended before the driver stream connected, nothing
    could retarget, and the stale switch landed seconds later.
    """
    if self._switching:
      cloudlog.info("vision_bsm ui: cancelled in-flight camera switch "
                    f"to {self._target_stream_type} (no longer wanted)")
      self._target_client = None
      self._target_stream_type = None
      self._switching = False

  def _switch_stream_if_needed(self, sm):
    # decide against what WILL be on screen once any in-flight switch lands,
    # not what is on screen now
    effective = self._target_stream_type if self._switching else self.stream_type

    if self._bsm_zone is not None and DRIVER_CAM in self.available_streams:
      target = DRIVER_CAM
    elif sm['selfdriveState'].experimentalMode and WIDE_CAM in self.available_streams:
      v_ego = sm['carState'].vEgo
      if v_ego < WIDE_CAM_MAX_SPEED:
        target = WIDE_CAM
      elif v_ego > ROAD_CAM_MIN_SPEED:
        target = NARROW_ROAD_CAM
      else:
        # Hysteresis zone - keep whatever we are showing or heading to, but
        # only ever a road-facing stream. Holding "current" unconditionally
        # latched the DRIVER view here: end a turn signal at 11-22 mph - the
        # normal speed coming out of a street turn, with experimental mode on -
        # and the blind spot preview stayed on screen until speed left the
        # band. Measured on the 2026-08-10 drive: experimental active 100% of
        # the time, so every turn ending in the band latched.
        target = effective if effective in (WIDE_CAM, ROAD_CAM) else ROAD_CAM
    else:
      target = NARROW_ROAD_CAM

    if effective == target:
      return
    if target == self.stream_type:
      # already showing the right stream; kill the switch away from it
      self._cancel_pending_switch()
    else:
      self.switch_stream(target)

  def _update_calibration(self):
    # Update device camera if not already set
    sm = ui_state.sm
    if not self.device_camera and sm.seen['narrowRoadCameraState'] and sm.seen['deviceState']:
      self.device_camera = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['narrowRoadCameraState'].sensor))]

    # Check if camera calibration data is available and valid
    if not (sm.updated["extrinsicsCalibration"] and sm.valid['extrinsicsCalibration']):
      return

    calib = sm['extrinsicsCalibration']
    if len(calib.rpyCalib) != 3 or calib.calStatus != CALIBRATED:
      return

    # Update view_from_calib matrix
    device_from_calib = rot_from_euler(calib.rpyCalib)
    self.view_from_calib = view_frame_from_device_frame @ device_from_calib

    # Update wide calibration if available
    if hasattr(calib, 'wideFromDeviceEuler') and len(calib.wideFromDeviceEuler) == 3:
      wide_from_device = rot_from_euler(calib.wideFromDeviceEuler)
      self.view_from_wide_calib = view_frame_from_device_frame @ wide_from_device @ device_from_calib

  def _calc_frame_matrix(self, rect: rl.Rectangle) -> np.ndarray:
    # blind spot view: magnify the driver frame until only the signalled window is left
    if self._bsm_zone is not None and self.stream_type == DRIVER_CAM and self.frame:
      x0, y0, w, h = self._bsm_zone
      zx, zy = 1.0 / w, 1.0 / h

      # keep the crop's real shape rather than stretching it to the screen
      zone_ratio = (w * self.frame.width) / (h * self.frame.height)
      rect_ratio = rect.width / rect.height
      if zone_ratio < rect_ratio:
        zx *= zone_ratio / rect_ratio
      else:
        zy *= rect_ratio / zone_ratio

      # the driver stream is drawn mirrored, so x is measured from the far side
      cx, cy = x0 + w / 2, y0 + h / 2
      return np.array([
        [zx, 0.0, zx * (2.0 * cx - 1.0)],
        [0.0, zy, zy * (1.0 - 2.0 * cy)],
        [0.0, 0.0, 1.0],
      ])

    cache_key = (
      ui_state.sm.recv_frame['extrinsicsCalibration'],
      int(self._content_rect.width),
      int(self._content_rect.height),
      self.stream_type,
      round(ui_state.sm['carState'].vEgo, 1),
    )

    if cache_key == self._matrix_cache_key and self._cached_matrix is not None:
      return self._cached_matrix

    # Get camera configuration
    device_camera = self.device_camera or DEFAULT_DEVICE_CAMERA
    is_wide_camera = self.stream_type == WIDE_CAM
    intrinsic = device_camera.wide_road.intrinsics if is_wide_camera else device_camera.narrow_road.intrinsics
    calibration = self.view_from_wide_calib if is_wide_camera else self.view_from_calib
    if is_wide_camera:
      zoom = 0.7 * 1.5
    else:
      zoom = np.interp(ui_state.sm['carState'].vEgo, [10, 30], [0.8, 1.0])

    # Calculate transforms for vanishing point
    inf_point = np.array([1000.0, 0.0, 0.0])
    calib_transform = intrinsic @ calibration
    kep = calib_transform @ inf_point

    # Calculate center points and dimensions
    w, h = self._content_rect.width, self._content_rect.height
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    # Ensure zoom views the whole area
    zoom = max(zoom, w / (2 * cx), h / (2 * cy))

    # Calculate max allowed offsets with margins
    margin = 5
    max_x_offset = max(0.0, cx * zoom - w / 2 - margin)
    max_y_offset = max(0.0, cy * zoom - h / 2 - margin)

    # Calculate and clamp offsets to prevent out-of-bounds issues
    try:
      if abs(kep[2]) > 1e-6:
        x_offset = np.clip((kep[0] / kep[2] - cx) * zoom, -max_x_offset, max_x_offset)
        y_offset = np.clip((kep[1] / kep[2] - cy) * zoom + CAM_Y_OFFSET, -max_y_offset, max_y_offset)
      else:
        x_offset, y_offset = 0, 0
    except (ZeroDivisionError, OverflowError):
      x_offset, y_offset = 0, 0

    # Cache the computed transformation matrix to avoid recalculations
    self._matrix_cache_key = cache_key
    self._cached_matrix = np.array([
      [zoom * 2 * cx / w, 0, -x_offset / w * 2],
      [0, zoom * 2 * cy / h, -y_offset / h * 2],
      [0, 0, 1.0]
    ])

    # built without rect.x/y so cache stays hot during scroll. ModelRenderer adds offset at draw time
    video_transform = np.array([
      [zoom, 0.0, (w / 2 - x_offset) - (cx * zoom)],
      [0.0, zoom, (h / 2 - y_offset) - (cy * zoom)],
      [0.0, 0.0, 1.0]
    ])
    self._model_renderer.set_transform(video_transform @ calib_transform)

    return self._cached_matrix

  def show_event(self):
    if gui_app.sunnypilot_ui():
      ui_state.reset_onroad_sleep_timer(OnroadTimerStatus.RESUME)

  def hide_event(self):
    if gui_app.sunnypilot_ui():
      ui_state.reset_onroad_sleep_timer(OnroadTimerStatus.PAUSE)


if __name__ == "__main__":
  gui_app.init_window("OnRoad Camera View")
  road_camera_view = AugmentedRoadView(lambda: None, stream_type=NARROW_ROAD_CAM)
  print("***press space to switch camera view***")
  try:
    for _ in gui_app.render():
      ui_state.update()
      if rl.is_key_released(rl.KeyboardKey.KEY_SPACE):
        if WIDE_CAM in road_camera_view.available_streams:
          stream = NARROW_ROAD_CAM if road_camera_view.stream_type == WIDE_CAM else WIDE_CAM
          road_camera_view.switch_stream(stream)
      road_camera_view.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
  finally:
    road_camera_view.close()
