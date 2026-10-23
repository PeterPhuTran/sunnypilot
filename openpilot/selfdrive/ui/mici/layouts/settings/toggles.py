import json
import os

from openpilot.cereal import log

from openpilot.system.ui.widgets.scroller import NavScroller
from openpilot.selfdrive.ui.mici.widgets.button import BigToggle, BigParamControl, BigMultiParamToggle
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.layouts.settings.common import restart_needed_callback
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.mici.layouts.settings.initial_cruise_speed import InitialCruiseSpeedButton

PERSONALITY_TO_INT = log.LongitudinalPersonality.schema.enumerants

VBSM_CONFIG = "/data/vision_bsm.json"


class BigConfigControl(BigToggle):
  """A toggle backed by a JSON config file rather than a param.

  The camera blind spot monitor ships as a drop-in with no compiled param of its
  own, but it still belongs in the same settings list as everything else.
  """

  def __init__(self, text: str, key: str, path: str = VBSM_CONFIG, default: bool = False, toggle_callback=None):
    super().__init__(text, "", toggle_callback=toggle_callback)
    self.key = key
    self.path = path
    self.default = default
    self.set_checked(self._read())

  def _read(self) -> bool:
    # an absent key falls back to the consumer's own default, so a toggle never
    # reads OFF for a feature that is actually running
    try:
      with open(self.path) as f:
        config = json.load(f)
    except (OSError, ValueError):
      return self.default
    return bool(config.get(self.key, self.default)) if isinstance(config, dict) else self.default

  def _write(self, value: bool) -> None:
    try:
      with open(self.path) as f:
        config = json.load(f)
    except (OSError, ValueError):
      config = {}
    if not isinstance(config, dict):
      config = {}
    config[self.key] = value
    tmp = self.path + ".tmp"
    try:
      with open(tmp, "w") as f:
        json.dump(config, f)
      os.replace(tmp, self.path)
    except OSError:
      pass

  def _handle_mouse_release(self, mouse_pos):
    super()._handle_mouse_release(mouse_pos)
    self._write(self._checked)

  def refresh(self) -> None:
    self.set_checked(self._read())


class TogglesLayoutMici(NavScroller):
  def __init__(self):
    super().__init__()

    self._personality_toggle = BigMultiParamToggle("following distance", "LongitudinalPersonality", ["aggressive", "standard", "relaxed"],
                                                   description="Standard is recommended.\n" +
                                                               "Aggressive follows closer, with firmer gas and braking.\n" +
                                                               "Relaxed leaves more space.\n" +
                                                               "Use the steering wheel distance button on supported cars.")
    self._initial_cruise_speed = InitialCruiseSpeedButton()
    self._initial_cruise_speed.set_visible(lambda: ui_state.CP is not None and ui_state.has_longitudinal_control and not ui_state.CP.pcmCruise)
    is_metric_toggle = BigToggle("use metric units", initial_state=ui_state.params.get_bool("IsMetric"), toggle_callback=self._on_metric_toggle)
    ldw_toggle = BigParamControl("lane departure warnings", "IsLdwEnabled",
                                 description="Warn when you drift across a detected lane line.\n" +
                                             "Only above 31 mph (50 km/h), with no turn signal.")
    always_on_dm_toggle = BigParamControl("always-on driver monitor", "AlwaysOnDM", description="Monitor the driver even when sunnypilot is not engaged.")
    record_front = BigParamControl("record & upload cabin camera", "RecordFront",
                                   description_icon=gui_app.texture("icons_mici/settings/device/cameras.png", 64, 64),
                                   toggle_callback=restart_needed_callback, description="Upload cabin camera data to help improve driver monitoring.")
    record_mic = BigParamControl("record & upload mic audio", "RecordAudio", description_icon=gui_app.texture("icons_mici/microphone.png", 64, 64),
                                 toggle_callback=restart_needed_callback,
                                 description="Record microphone audio while driving.\n" +
                                             "Audio is included in dashcam videos in sunnylink.")
    enable_openpilot = BigParamControl("enable sunnypilot", "OpenpilotEnabledToggle", toggle_callback=restart_needed_callback,
                                       description="Enable to use sunnypilot driver assistance.\n" +
                                                   "Disable to use your car's stock driver assistance.")
    self._vision_bsm = BigConfigControl("camera blind spot monitor", "enabled")
    # the three blind spot outputs, independently switchable: what you see on
    # screen, what you hear, and whether the window view takes over
    self._vision_bsm_icons = BigParamControl("blind spot icons on screen", "BlindSpot")
    self._vision_bsm_chime = BigConfigControl("blind spot chime on signal", "chime", default=True)
    self._vision_bsm_chime_always = BigConfigControl("chime on every blind spot car", "chime_always")
    self._vision_bsm_view = BigConfigControl("blind spot window view on signal", "camera_view")
    self._reverse_view = BigConfigControl("driver camera in reverse", "reverse_view", default=True)  # VBSM_REVERSE_CAM

    self._scroller.add_widgets([
      self._personality_toggle,
      self._initial_cruise_speed,
      is_metric_toggle,
      ldw_toggle,
      always_on_dm_toggle,
      self._vision_bsm,
      self._vision_bsm_icons,
      self._vision_bsm_chime,
      self._vision_bsm_chime_always,
      self._vision_bsm_view,
      self._reverse_view,
      record_front,
      record_mic,
      enable_openpilot,
    ])

    # Toggle lists
    self._refresh_toggles = (
      ("IsMetric", is_metric_toggle),
      ("IsLdwEnabled", ldw_toggle),
      ("AlwaysOnDM", always_on_dm_toggle),
      ("BlindSpot", self._vision_bsm_icons),
      ("RecordFront", record_front),
      ("RecordAudio", record_mic),
      ("OpenpilotEnabledToggle", enable_openpilot),
    )

    enable_openpilot.set_enabled(lambda: not ui_state.engaged)
    record_front.set_enabled(False if ui_state.params.get_bool("RecordFrontLock") else (lambda: not ui_state.engaged))
    record_mic.set_enabled(lambda: not ui_state.engaged)

    if ui_state.params.get_bool("ShowDebugInfo"):
      gui_app.set_show_touches(True)
      gui_app.set_show_fps(True)

    ui_state.add_engaged_transition_callback(self._update_toggles)

  def _update_state(self):
    super()._update_state()

    if ui_state.sm.updated["selfdriveState"]:
      personality = PERSONALITY_TO_INT[ui_state.sm["selfdriveState"].personality]
      if personality != ui_state.personality and ui_state.started:
        self._personality_toggle.set_value(self._personality_toggle._options[personality])
      ui_state.personality = personality

  def show_event(self):
    super().show_event()
    self._update_toggles()

  def _on_metric_toggle(self, state: bool):
    ui_state.params.put_bool("IsMetric", state, block=True)
    self._update_toggles()

  def _update_toggles(self):
    ui_state.update_params()
    self._initial_cruise_speed.refresh()

    if ui_state.CP is not None:
      self._personality_toggle.set_visible(ui_state.has_longitudinal_control)

    # Refresh toggles from params to mirror external changes
    for key, item in self._refresh_toggles:
      item.set_checked(ui_state.params.get_bool(key))

    # config backed, so not in the params list above
    self._vision_bsm.refresh()
    self._vision_bsm_chime.refresh()
    self._vision_bsm_chime_always.refresh()
    self._vision_bsm_view.refresh()
    self._reverse_view.refresh()
