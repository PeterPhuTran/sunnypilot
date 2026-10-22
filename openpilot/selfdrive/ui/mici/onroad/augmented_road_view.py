import numpy as np
import pyray as rl
from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.cereal.visionipc import VisionStreamType
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.selfdrive.ui.mici.onroad import SIDE_PANEL_WIDTH
from openpilot.selfdrive.ui.mici.onroad.alert_renderer import Alert, AlertRenderer
from openpilot.selfdrive.ui.mici.onroad.driver_state import DriverStateRenderer
from openpilot.selfdrive.ui.mici.onroad.hud_renderer import HudRenderer
from openpilot.selfdrive.ui.mici.onroad.long_indicator import LongIndicator
from openpilot.selfdrive.ui.mici.onroad.model_renderer import ModelRenderer
from openpilot.selfdrive.ui.mici.onroad.navigation import NavigationPanel, PANEL_WIDTH
from openpilot.selfdrive.ui.mici.onroad.confidence_ball import ConfidenceBall
from openpilot.selfdrive.ui.mici.onroad.cameraview import CameraView
from openpilot.system.ui.lib.application import FontWeight, gui_app, MousePos, MouseEvent, TextAlignment, TextAlignmentVertical
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets import Widget
from openpilot.common.filter_simple import BounceFilter, FirstOrderFilter
from openpilot.common.swaglog import cloudlog
import json
import os
import time
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
DRIVER_CAM = VisionStreamType.VISION_STREAM_CABIN
BSM_PARAM_INTERVAL = 2.0  # seconds between blind spot camera config checks
REVERSE_ALERT_S = 5.0  # VBSM_REVERSE_CAM: how long the full-size "Reverse" alert may cover the cabin view
HOLD_RESTART_S = 3.0  # VBSM_HOLD_RESTART: hold a finger still on the road view this long for the restart prompt
HOLD_RESTART_SLOP_PX = 20  # farther than this before the lock and the touch is a swipe, not a hold (the profile-chip tap tolerance)
HOLD_RESTART_LOCK_S = 0.5  # still this long: the touch locks as a hold (no scroll, no bookmark, no tap) and shows its progress ring
HOLD_RESTART_LOCKED_SLOP_PX = 60  # a locked hold may drift this far (a finger resting through 3 s of road bumps) before it cancels
HOLD_RING_R = 46  # progress ring radius, px
BSM_CONFIG_PATH = "/data/vision_bsm.json"
BSM_STATE_PATH = "/dev/shm/vision_bsm_state"  # written by vision_bsm; the night notice that read it was retired with the model (VBSM_HUD)
DEFAULT_DEVICE_CAMERA = DEVICE_CAMERAS["tici", "ar0231"]


class BookmarkState(IntEnum):
  HIDDEN = 0
  DRAGGING = 1
  TRIGGERED = 2

WIDE_CAM_MAX_SPEED = 5.0  # m/s (11 mph)
ROAD_CAM_MIN_SPEED = 10.0  # m/s (22 mph)

CAM_Y_OFFSET = 20


def open_restart_prompt() -> None:
  """VBSM_HOLD_RESTART: open the restart prompt of Settings > Device > reboot, unchanged ("slide to reboot" when
  disengaged, "Disengage to reboot" when engaged, with the same DoReboot action). Imported on use, so a rename
  upstream only disables the gesture (logged) and never stops the UI from starting."""
  try:
    from openpilot.selfdrive.ui.mici.layouts.settings.device.device_layout import _engaged_confirmation_click
    _engaged_confirmation_click(lambda: ui_state.params.put_bool("DoReboot", True, block=True), "reboot",
                                gui_app.texture("icons_mici/settings/device/reboot.png", 64, 70), exit_on_confirm=False)
    cloudlog.warning("VBSM_HOLD_RESTART: restart prompt opened from the road view")
  except Exception:
    cloudlog.exception("VBSM_HOLD_RESTART: could not open the restart prompt")


class BookmarkIcon(Widget):
  PEEK_THRESHOLD = 150 # If icon peeks out this much, snap it fully visible
  FULL_VISIBLE_OFFSET = 200  # How far onscreen when fully visible
  HIDDEN_OFFSET = -50  # How far offscreen when hidden

  def __init__(self, bookmark_callback):
    super().__init__()
    self._bookmark_callback = bookmark_callback
    self._icon = gui_app.texture("icons_mici/onroad/bookmark.png", 180, 180)
    self._filled_icon = gui_app.texture("icons_mici/onroad/bookmark_fill.png", 180, 180)
    self._offset_filter = BounceFilter(0.0, 0.1, 1 / gui_app.target_fps)
    self._active_alpha = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)

    # State
    self._interacting = False
    self._state = BookmarkState.HIDDEN
    self._swipe_start_x = 0.0
    self._swipe_current_x = 0.0
    self._is_swiping = False
    self._is_swiping_left: bool = False
    self._triggered_time: float = 0.0
    self._triggered_duration = 1.5
    self._swipe_expired = False
    # VBSM_HOLD_RESTART: when this touch started, whether it opened the restart prompt, and how far it has moved
    # from where it started
    self._press_time: float = 0.0
    self._hold_fired = False
    self._hold_start_y = 0.0
    self._hold_moved_px = 0.0
    self._last_update_t = 0.0
    # VBSM_HOLD_RESTART: the touch stayed still for HOLD_RESTART_LOCK_S (kept until the next press, like _hold_fired),
    # and whether this touch's end was logged
    self._hold_locked = False
    self._hold_end_logged = False
    self._hold_icon = None
    self._hold_icon_tried = False

  def is_swiping_left(self) -> bool:
    """Check if currently swiping left (for scroller to disable)."""
    return self._is_swiping_left

  def interacting(self):
    interacting, self._interacting = self._interacting, False
    return interacting

  @property
  def hold_fired(self) -> bool:
    """VBSM_HOLD_RESTART: the current touch opened the restart prompt; its release must not act as a tap."""
    return self._hold_fired

  @property
  def hold_locked(self) -> bool:
    """VBSM_HOLD_RESTART: the current touch is a hold (still for HOLD_RESTART_LOCK_S): the main scroller must not
    scroll it and its release is not a tap."""
    return self._hold_locked

  @property
  def hold_down(self) -> bool:
    """VBSM_HOLD_RESTART: a locked hold's finger is still on the screen."""
    return self._hold_locked and self._is_swiping

  def _hold_active(self) -> bool:
    """VBSM_HOLD_RESTART: a locked hold that is still counting toward the restart prompt."""
    return (self._hold_locked and self._is_swiping and not self._hold_fired and
            self._hold_moved_px <= HOLD_RESTART_LOCKED_SLOP_PX and self._parent_touch_ok())

  def _log_hold_end(self, why: str) -> None:
    """VBSM_HOLD_RESTART: say once why a locked hold ended without the prompt (data for tuning the tolerances)."""
    if self._hold_locked and not self._hold_fired and not self._hold_end_logged:
      self._hold_end_logged = True
      cloudlog.warning(f"VBSM_HOLD_RESTART: hold ended without the prompt after {rl.get_time() - self._press_time:.1f} s, "
                       f"moved {self._hold_moved_px:.0f} px: {why}")

  def _parent_touch_ok(self) -> bool:
    """VBSM_HOLD_RESTART / VBSM_HUD: a widget pushed over the road view (the restart prompt itself) turns the road
    view's touches off but not this icon's, so the gestures that act (prompt, profile chip) check the road view."""
    cb = getattr(self, "parent_touch_valid", None)
    return cb is None or bool(cb())

  def _update_state(self):
    # VBSM_HOLD_RESTART: this icon sees touches only while the road view draws it (onroad and on screen). A touch
    # whose release came while it was not drawn (car turned off, view scrolled away) is over, not a 3 s hold.
    now = rl.get_time()
    if now - self._last_update_t > 0.5 and self._press_time <= self._last_update_t:
      if self._is_swiping:
        self._log_hold_end("view not drawn for %.1f s" % (now - self._last_update_t))
      self._is_swiping = False
      self._is_swiping_left = False
      if self._state == BookmarkState.DRAGGING:
        self._state = BookmarkState.HIDDEN
    self._last_update_t = now
    if self._is_swiping and not self._parent_touch_ok():
      self._log_hold_end("covered or scrolled")
      self._hold_moved_px = HOLD_RESTART_LOCKED_SLOP_PX + 1  # a touch made over a pushed widget, or a scroll, is never a hold
    if self._is_swiping and self._hold_moved_px > HOLD_RESTART_LOCKED_SLOP_PX:
      self._log_hold_end("moved")

    # VBSM_HOLD_RESTART: a touch still (within HOLD_RESTART_SLOP_PX) for HOLD_RESTART_LOCK_S locks as a hold. From then
    # on the main scroller ignores it (AugmentedRoadView.is_swiping_left), it is no bookmark swipe and no tap, and it
    # may drift up to HOLD_RESTART_LOCKED_SLOP_PX. Before the lock, the scroller's 12 px drag threshold ended most holds.
    if (ui_state.started and self._is_swiping and not self._hold_locked and not self._hold_fired and self._parent_touch_ok() and
        self._hold_moved_px <= HOLD_RESTART_SLOP_PX and now - self._press_time >= HOLD_RESTART_LOCK_S):
      self._hold_locked = True
      self._is_swiping_left = False
      if self._state == BookmarkState.DRAGGING:
        self._state = BookmarkState.HIDDEN

    # VBSM_HOLD_RESTART: a locked hold that reaches HOLD_RESTART_S opens the restart prompt. The prompt takes over the
    # touch, so this view may never see the release: end the swipe here, and mark the touch so nothing else acts on it
    if ui_state.started and self._hold_active() and now - self._press_time >= HOLD_RESTART_S:
      self._hold_fired = True
      self._is_swiping = False
      self._is_swiping_left = False
      self._state = BookmarkState.HIDDEN
      open_restart_prompt()

    if self._state == BookmarkState.TRIGGERED and rl.get_time() - self._triggered_time >= self._triggered_duration:
      self._state = BookmarkState.HIDDEN
      self._swipe_expired = self._is_swiping

    swipe_offset = self._swipe_start_x - self._swipe_current_x
    armed = self._state == BookmarkState.DRAGGING and swipe_offset > self.PEEK_THRESHOLD
    if self._state == BookmarkState.DRAGGING:
      # Snap to the released position when armed, while allowing further dragging or cancellation.
      if armed:
        swipe_offset += self.FULL_VISIBLE_OFFSET - self.PEEK_THRESHOLD
      swipe_offset = min(swipe_offset, self.FULL_VISIBLE_OFFSET + 50)
      self._offset_filter.update(swipe_offset)

    elif self._state == BookmarkState.TRIGGERED:
      # Let another left swipe move the same bookmark.
      offset = self.FULL_VISIBLE_OFFSET
      if self._is_swiping and self._is_swiping_left:
        offset += min(swipe_offset, 50)
      self._offset_filter.update(offset)

    elif self._state == BookmarkState.HIDDEN:
      self._offset_filter.update(self.HIDDEN_OFFSET)

      if self._offset_filter.x < 1e-3:
        self._interacting = False

    self._active_alpha.update(float(armed or self._state == BookmarkState.TRIGGERED))

  def _handle_mouse_event(self, mouse_event: MouseEvent):
    if not ui_state.started:
      return

    if mouse_event.left_pressed:
      if self._state == BookmarkState.TRIGGERED:
        self._triggered_time = rl.get_time()
        self._triggered_duration = 0.5

      # Store relative position within widget
      self._swipe_start_x = mouse_event.pos.x
      self._swipe_current_x = mouse_event.pos.x
      self._is_swiping = True
      self._is_swiping_left = False
      self._swipe_expired = False
      self._press_time = rl.get_time()   # VBSM_HOLD_RESTART: every touch decides afresh
      self._hold_fired = False
      self._hold_start_y = mouse_event.pos.y
      self._hold_moved_px = 0.0
      self._hold_locked = False
      self._hold_end_logged = False

    elif mouse_event.left_down and self._is_swiping:
      self._swipe_current_x = mouse_event.pos.x
      self._hold_moved_px = max(self._hold_moved_px, abs(mouse_event.pos.x - self._swipe_start_x),
                                abs(mouse_event.pos.y - self._hold_start_y))   # VBSM_HOLD_RESTART
      swipe_offset = self._swipe_start_x - self._swipe_current_x
      self._is_swiping_left = swipe_offset > 0 and not self._hold_locked  # VBSM_HOLD_RESTART: a locked hold is no bookmark swipe
      if self._is_swiping_left:
        self._interacting = True
        if self._state == BookmarkState.HIDDEN and not self._swipe_expired:
          self._state = BookmarkState.DRAGGING

    elif mouse_event.left_released and (self._hold_fired or self._hold_locked):
      # VBSM_HOLD_RESTART: this touch already opened the restart prompt, or was a hold let go early (or moved off);
      # its release does nothing else. The flags stay set until the next press: the parent's _handle_mouse_release
      # reads them after this handler has run (Widget.render() runs _render(), which renders this icon, before the
      # parent's _process_mouse_events()).
      if self._is_swiping:
        self._log_hold_end("released")
      self._is_swiping = False
      self._is_swiping_left = False

    elif mouse_event.left_released:
      # VBSM_HUD: a tap (no meaningful swipe) on the profile chip cycles the
      # driving personality; same decrement direction as the wheel button.
      # selfdrived re-reads the param on its periodic config check.
      # (this handler is BookmarkIcon's -- it receives all road-view touches;
      # the parent view injects hud_renderer after construction. A bare
      # self._hud_renderer here crashed the whole UI on any tap: BookmarkIcon
      # never had that attribute.)
      hud = getattr(self, "hud_renderer", None)
      if hud is not None and self._is_swiping and abs(self._swipe_start_x - self._swipe_current_x) < 20 and self._parent_touch_ok():
        chip = hud.profile_chip_rect
        if chip.width > 0 and rl.check_collision_point_rec(mouse_event.pos, chip):
          cur = int(ui_state.sm['selfdriveState'].personality.raw)
          ui_state.params.put("LongitudinalPersonality", (cur - 1) % 3)
          self._interacting = True  # the parent's click (scroll to home) must not fire for a chip tap
      if self._is_swiping and self._state == BookmarkState.DRAGGING:
        swipe_distance = self._swipe_start_x - self._swipe_current_x

        # If peeking past threshold, transition to animating to fully visible and bookmark
        if swipe_distance > self.PEEK_THRESHOLD:
          self._state = BookmarkState.TRIGGERED
          self._triggered_time = rl.get_time()
          self._triggered_duration = 1.5
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
      for icon, alpha in ((self._icon, 1.0 - self._active_alpha.x), (self._filled_icon, self._active_alpha.x)):
        rl.draw_texture_ex(icon, rl.Vector2(icon_x, icon_y), 0.0, 1.0, rl.Color(255, 255, 255, round(255 * alpha)))
    self._render_hold_ring()

  def _render_hold_ring(self):
    """VBSM_HOLD_RESTART: a locked hold shows a ring that fills up to the restart prompt, on the other half of the
    view from the finger so the finger does not cover it. The ring vanishes the moment the hold cancels."""
    if not ui_state.started or not self._hold_active():
      return
    frac = float(np.clip((rl.get_time() - self._press_time - HOLD_RESTART_LOCK_S) / (HOLD_RESTART_S - HOLD_RESTART_LOCK_S), 0.0, 1.0))
    left_half = self._swipe_start_x < self.rect.x + self.rect.width / 2
    center = rl.Vector2(self.rect.x + self.rect.width * (0.75 if left_half else 0.25), self.rect.y + self.rect.height / 2)
    rl.draw_circle_v(center, HOLD_RING_R + 16, rl.Color(0, 0, 0, 160))
    rl.draw_ring(center, HOLD_RING_R - 5, HOLD_RING_R + 5, 0.0, 360.0, 48, rl.Color(255, 255, 255, 60))
    rl.draw_ring(center, HOLD_RING_R - 5, HOLD_RING_R + 5, -90.0, -90.0 + 360.0 * frac, 48, rl.WHITE)
    if not self._hold_icon_tried:
      self._hold_icon_tried = True
      try:
        self._hold_icon = gui_app.texture("icons_mici/settings/device/reboot.png", 42, 46)
      except Exception:
        cloudlog.exception("VBSM_HOLD_RESTART: no reboot icon; the ring alone shows the progress")
    if self._hold_icon is not None:
      rl.draw_texture_ex(self._hold_icon, rl.Vector2(center.x - self._hold_icon.width / 2, center.y - self._hold_icon.height / 2),
                         0.0, 1.0, rl.WHITE)


class ReverseAlertRenderer(AlertRenderer):
  """VBSM_REVERSE_CAM: the stock "Reverse" alert is full size and stays up for the whole time in reverse, covering
  the cabin view. Show it for `reverse_alert_s`, then stop drawing it (it fades out like any other alert). Only the
  drawing changes: selfdrived still raises reverseGear, so engagement stays blocked, and its no-entry alert (a
  different alert type) still shows. None keeps the stock behaviour."""

  def __init__(self):
    super().__init__()
    self.reverse_alert_s: float | None = None
    self._reverse_alert_since: float | None = None

  def get_alert(self, sm) -> Alert | None:
    prev = self._prev_alert
    alert = super().get_alert(sm)
    if alert is None or alert.alert_type != "reverseGear/permanent":
      self._reverse_alert_since = None
      return alert
    now = time.monotonic()
    if self._reverse_alert_since is None:
      self._reverse_alert_since = now
    if self.reverse_alert_s is None or now - self._reverse_alert_since < self.reverse_alert_s:
      return alert
    # super() just stored the hidden alert as the fade-out target; put back what was there, or it never fades away
    self._prev_alert = prev
    return None


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

    self._navigation = NavigationPanel()
    self._model_renderer = ModelRenderer()
    self._hud_renderer = HudRenderer()
    # VBSM_HUD: the profile-chip tap target lives in BookmarkIcon's touch
    # handler (the only receiver of road-view touches); give it the renderer
    self._bookmark_icon.hud_renderer = self._hud_renderer
    self._bookmark_icon.parent_touch_valid = lambda: self.enabled and self._touch_valid()
    self._long_indicator = LongIndicator()
    self._alert_renderer = ReverseAlertRenderer()
    self._driver_state_renderer = DriverStateRenderer()
    self._confidence_ball = ConfidenceBall()
    self._offroad_label = UnifiedLabel("start the car to\nuse sunnypilot", 54, FontWeight.DISPLAY,
                                       text_color=rl.Color(255, 255, 255, int(255 * 0.9)),
                                       alignment=TextAlignment.CENTER,
                                       alignment_vertical=TextAlignmentVertical.MIDDLE)

    self._fade_texture = gui_app.texture("icons_mici/onroad/onroad_fade.png")

    # blind spot camera view
    self._bsm_config_mtime = 0.0
    self._bsm_zone: tuple[float, float, float, float] | None = None
    self._bsm_left = True
    self._bsm_zones: dict = {}
    self._bsm_enabled = False
    self._bsm_params_checked = 0.0

    # VBSM_REVERSE_CAM: the cabin camera takes the screen in reverse (`reverse_view`, default on)
    self._reverse_view = True
    self._reversing = False

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
      self._reverse_view = True
      return
    if mtime == self._bsm_config_mtime:
      return
    self._bsm_config_mtime = mtime

    try:
      with open(BSM_CONFIG_PATH) as f:
        config = json.load(f)
      if not isinstance(config, dict):
        raise ValueError("vision_bsm.json is not an object")
      # VBSM_HUD: the window view needs only camera_view + zones. `enabled`
      # gates the detector (model, chevrons, chime, icons); with it off the
      # view is a plain mirror of the signalled side and no CPU goes to the model.
      self._bsm_enabled = bool(config.get("camera_view"))
      zones = config.get("zones", {}) if self._bsm_enabled else {}
      self._bsm_zones = zones if isinstance(zones, dict) else {}
      self._reverse_view = bool(config.get("reverse_view", True))
    except (OSError, ValueError):
      self._bsm_enabled = False
      self._bsm_zones = {}
      self._reverse_view = True

  def _bsm_bounds(self, side: str) -> tuple[float, float, float, float] | None:
    """Bounding box of a side's calibrated polygons, as (x, y, w, h) in 0..1."""
    xs, ys = [], []
    polygons = self._bsm_zones.get(side, [])
    for polygon in polygons if isinstance(polygons, list) else []:
      for point in polygon if isinstance(polygon, list) else []:
        if isinstance(point, list) and len(point) == 2 and all(isinstance(v, (int, float)) for v in point):
          xs.append(float(point[0]))
          ys.append(float(point[1]))
    if not xs:
      return None
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if x1 <= x0 or y1 <= y0:
      return None
    return x0, y0, x1 - x0, y1 - y0

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

  def _update_reverse(self) -> bool:
    """VBSM_REVERSE_CAM: the whole cabin frame takes the screen while the car is in reverse.

    Same trust rule as the window view: the gear only counts while carState is
    flowing, so a frozen carState cannot hold the cabin view on screen.
    """
    sm = ui_state.sm
    self._reversing = bool(self._reverse_view and sm.alive['carState'] and sm.valid['carState'] and
                           sm['carState'].gearShifter == car.CarState.GearShifter.reverse)
    return self._reversing

  def is_swiping_left(self) -> bool:
    """Check if currently swiping left (for scroller to disable). VBSM_HOLD_RESTART: also while a locked hold is
    down, so the finger's drift never scrolls the main layout and ends the hold."""
    return self._bookmark_icon.is_swiping_left() or self._bookmark_icon.hold_down

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
    # Don't trigger click callback if bookmark was triggered, or if this touch was a hold (VBSM_HOLD_RESTART)
    if not self._bookmark_icon.interacting() and not self._bookmark_icon.hold_fired and not self._bookmark_icon.hold_locked:
      super()._handle_mouse_release(mouse_pos)

  def _render(self, _):
    # Draw text if not onroad
    if not ui_state.started:
      rl.draw_rectangle_rec(self.rect, rl.BLACK)
      self._offroad_label.render(self._rect)
      return

    blind_spot_view = self._update_bsm()
    reverse_view = self._update_reverse()
    self._alert_renderer.reverse_alert_s = REVERSE_ALERT_S if self._reverse_view else None
    # VBSM_REVERSE_CAM: the stock "Reverse" alert kept the screen awake for the whole reverse; it is hidden after
    # REVERSE_ALERT_S now, so keep the onroad sleep timer reset here instead, as the alert did
    if reverse_view and gui_app.sunnypilot_ui():
      ui_state.reset_onroad_sleep_timer()
    self._switch_stream_if_needed(ui_state.sm)

    # Update calibration before rendering
    self._update_calibration()

    alert_to_render, not_animating_out = self._alert_renderer.will_render()
    show_navigation = ui_state.navigation_enabled and ui_state.nav_destination_set and alert_to_render is None

    # Create inner content area with border padding
    self._content_rect = rl.Rectangle(
      self.rect.x,
      self.rect.y,
      self.rect.width - (PANEL_WIDTH if show_navigation else SIDE_PANEL_WIDTH),
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
    # of a side window or the cabin in reverse, but alerts still have to reach the driver.
    if not (blind_spot_view or reverse_view):
      self._model_renderer.render(self._content_rect)

    # Fade out bottom of overlays for looks
    rl.draw_texture_ex(self._fade_texture, rl.Vector2(self._content_rect.x, self._content_rect.y), 0.0, 1.0, rl.WHITE)

    # VBSM_HUD: the dmoji lives in the bottom-right eGPU slot now -- it shows
    # once the chestnut status icon has faded (the persistent set-speed owns its
    # old top-left slot, which kept it hidden whenever engaged). Hidden when
    # disengaged unless AlwaysOnDM is enabled, as before.
    should_draw_dmoji = (not self._hud_renderer.chestnut_icon_visible() and
                         (ui_state.status != UIStatus.DISENGAGED or ui_state.always_on_dm))
    self._driver_state_renderer.set_should_draw(should_draw_dmoji)
    dm_size = self._driver_state_renderer.BASE_SIZE
    self._driver_state_renderer.set_position(
      self._content_rect.x + self._content_rect.width - 10 - dm_size,
      self._content_rect.y + self._content_rect.height - 14 - (50 + dm_size) / 2)  # 50 = wheel row height, same math as the eGPU icon
    self._driver_state_renderer.render()
    self._long_indicator.set_should_draw(not self._hud_renderer.drawing_top_icons())
    self._long_indicator.render(self._content_rect)

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
    if show_navigation:
      self._navigation.render(rl.Rectangle(self._content_rect.x + self._content_rect.width, self.rect.y, PANEL_WIDTH, self.rect.height))
    else:
      self._confidence_ball.render(self.rect)

    self._bookmark_icon.render(self.rect)

  def close(self):
    self._navigation.close()
    super().close()

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

    if self._reversing and DRIVER_CAM in self.available_streams:
      target = DRIVER_CAM  # VBSM_REVERSE_CAM
    elif self._bsm_zone is not None and DRIVER_CAM in self.available_streams:
      target = DRIVER_CAM
    elif sm['carParams'].openpilotLongitudinalControl and WIDE_CAM in self.available_streams:
      v_ego = sm['carState'].vEgo
      if v_ego < WIDE_CAM_MAX_SPEED:
        target = WIDE_CAM
      elif v_ego > ROAD_CAM_MIN_SPEED:
        target = NARROW_ROAD_CAM
      else:
        # Hysteresis zone - keep whatever we are showing or heading to, but
        # only ever a road-facing stream. Holding "current" unconditionally
        # latched the DRIVER view here: end a turn signal at 11-22 mph - the
        # normal speed coming out of a street turn, with the wide camera in
        # play (then experimental mode, now any openpilot-longitudinal car) -
        # and the blind spot preview stayed on screen until speed left the
        # band. Measured on the 2026-08-10 drive: experimental active 100% of
        # the time, so every turn ending in the band latched.
        target = effective if effective in (WIDE_CAM, NARROW_ROAD_CAM) else NARROW_ROAD_CAM
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
    # VBSM_REVERSE_CAM: the whole cabin frame, fitted and mirrored like a rear-view mirror
    if self._reversing and self.stream_type == DRIVER_CAM:
      return super()._calc_frame_matrix(rect)

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
