"""Step C of the paper - autonomous control.

`IntelligentAutopilot` = Flight Manager (paper Fig. 5, phases final approach -> flare -> landing)
+ the trained ANN cascades. It is simulator-agnostic: feed it one raw observation dict every
0.1 s (see ias/features.py) and it returns normalised control commands.
"""
from __future__ import annotations

import numpy as np

from .ann import ANN, load_models
from .config import BIKF, DEFAULT_RUNWAY, DT, INITIAL_CONDITION, MODELS_DIR, Runway
from .features import FeatureExtractor
from .networks import FINAL, FLAP_NOTCHES, FLARE, LANDING

TOUCHDOWN_CONFIRM_S = 0.5      # weight-on-wheels must persist this long before "landing" is latched


class IntelligentAutopilot:
    def __init__(self, models: dict[str, ANN] | None = None, params: dict | None = None,
                 runway: Runway | None = None, elevator_trim: float = INITIAL_CONDITION["elevator_trim"]):
        if models is None:
            models, params = load_models(MODELS_DIR / "ias_models.json")
        self.ann, self.p = models, params
        self.runway = runway or BIKF[DEFAULT_RUNWAY]
        self.fx = FeatureExtractor(self.runway, self.p["aim_x_m"])
        self.trim = elevator_trim
        self.reset()

    def reset(self):
        self.fx.reset()
        self.phase = FINAL
        self.flaps = 0.0
        self._wow_time = 0.0
        self.debug: dict = {}

    # ---- Flight Manager -----------------------------------------------------------------
    def _update_phase(self, f: dict, dt: float):
        # radar height triggers the flare; the barometric height must agree that the runway is near
        if self.phase == FINAL and f["agl"] <= self.p["flare_height_ft"] and f["h"] < 3 * self.p["flare_height_ft"]:
            self.phase = FLARE
        if self.phase != LANDING:
            self._wow_time = self._wow_time + dt if f["wow"] else 0.0
            if self._wow_time >= TOUCHDOWN_CONFIRM_S:
                self.phase = LANDING

    # ---- one control cycle --------------------------------------------------------------
    def step(self, obs: dict, dt: float = DT) -> dict:
        f, a = self.fx(obs, dt), self.ann
        self._update_phase(f, dt)
        dbg = dict(phase=self.phase)

        # lateral cascade: centreline angle -> desired track -> desired roll -> aileron
        if self.phase in (FINAL, FLARE):
            trk_des = a["track"](f["loc_angle"], f["loc_int"])
            roll_des = a["roll"](trk_des - f["track_err"])
            rudder = a["rudder"](f["hdg_err"])
            aileron = a["aileron"](roll_des - f["roll"], rudder)
            brake = 0.0
            dbg.update(track_des=trk_des, roll_des=roll_des)

        if self.phase == FINAL:
            vs_des = a["glideslope_roc"](f["gs_angle"] - self.p["glideslope_ref_deg"])
            elevator = a["glideslope_elevator"](vs_des - f["vs_fps"], f["ias"])
            throttle = a["throttle"](vs_des)
            notch = FLAP_NOTCHES[np.abs(FLAP_NOTCHES - a["flaps"](f["h"], f["ias"])).argmin()]
            self.flaps = max(self.flaps, float(notch))          # flaps are only ever extended on final
            dbg.update(vs_des=vs_des)
        elif self.phase == FLARE:
            vs_des = a["flare_roc"](f["agl"])
            elevator = a["flare_elevator"](vs_des - f["vs_fps"], f["ias"])
            throttle = a["flare_throttle"](f["agl"])
            dbg.update(vs_des=vs_des)
        else:                                                    # LANDING: roll-out on the centreline
            hdg_des = a["heading"](f["loc_angle"])
            rudder = a["rudder_ground"](hdg_des - f["hdg_err"], f["hdg_int"])
            aileron = a["aileron"](-f["roll"], rudder)
            elevator = a["landing_elevator"](f["gs_kt"])
            throttle = 0.0
            brake = a["brakes"](f["gs_kt"])
            dbg.update(hdg_des=hdg_des)

        clip = lambda v, lo=-1.0, hi=1.0: float(min(max(v, lo), hi))
        self.debug = {**f, **dbg}
        return dict(
            aileron=clip(aileron),
            elevator=clip(elevator - self.trim),                 # ANN output is yoke+trim; trim stays fixed
            elevator_trim=self.trim,
            rudder=clip(rudder),
            throttle=clip(throttle, 0.0),
            flaps=self.flaps,
            brake=clip(brake, 0.0),
        )
