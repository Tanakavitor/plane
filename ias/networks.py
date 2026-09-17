"""The task-dedicated ANNs (paper Table 1 / Fig. 3, adapted to a visual C172P final approach)
and how their training sets are cut out of the demonstrations.

Every control surface gets the paper's two-stage cascade of single-hidden-layer ANNs:

    outer ("rate of change") ANN :  error                      -> desired response
    inner ("command") ANN        :  desired - current response -> control command

Labels come from the demonstration itself (learning by imitation, no hand-made targets):
  * outer target  = the response the pilot actually produced a moment later, given the error;
  * inner input   = the change of response that followed, inner target = the command the pilot
                    was holding to produce it (what the paper calls "the difference between the
                    current and the desired rate of change" -> command).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .config import DT

STEPS = lambda seconds: int(round(seconds / DT))

# flight phases handled by the flight manager (paper Fig. 5, last three boxes)
FINAL, FLARE, LANDING = "final_approach", "flare", "landing"

FLAP_NOTCHES = np.array([0.0, 1 / 3, 2 / 3, 1.0])


@dataclass
class Spec:
    name: str
    phase: str                      # which part of the demonstrations it learns from
    inputs: list[str]               # human-readable, for the report
    output: str
    n_hidden: int
    build: Callable[[pd.DataFrame, dict], tuple[np.ndarray, np.ndarray]]
    description: str = ""
    odd: bool = False               # left/right symmetric task -> bias-free, exactly odd network
    input_pct: tuple = (0.5, 99.5)  # input range kept (percentiles); tighter where the tails are too sparse to trust


def _smooth(s: pd.Series, seconds: float = 1.0) -> pd.Series:
    return s.rolling(STEPS(seconds), center=True, min_periods=1).mean()


def _lead(s: pd.Series, seconds: float) -> pd.Series:
    return s.shift(-STEPS(seconds))


def phase_of(d: pd.DataFrame, p: dict) -> pd.Series:
    ph = pd.Series(LANDING, index=d.index)
    ph[(d.airborne == 1)] = FLARE
    ph[(d.airborne == 1) & (d.agl > p["flare_height_ft"])] = FINAL
    # once the flare has begun it stays begun (a balloon above flare height is still the flare)
    first_flare = ph.eq(FLARE).idxmax() if ph.eq(FLARE).any() else None
    if first_flare is not None:
        ph[(d.index > first_flare) & (d.airborne == 1)] = FLARE
    return ph


def _xy(d, p, phase, x, y, settle_s=0.0):
    """Single-input training pairs restricted to one flight phase (and after the initial
    engine spool-up transient when settle_s is given)."""
    ok = (phase_of(d, p) == phase) & (d.t >= settle_s)
    df = pd.DataFrame({"x": x, "y": y})[ok].dropna()
    return df[["x"]].to_numpy(float), df["y"].to_numpy(float)


def _xy2(d, p, phase, x1, x2, y, settle_s=0.0):
    ok = (phase_of(d, p) == phase) & (d.t >= settle_s)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})[ok].dropna()
    return df[["x1", "x2"]].to_numpy(float), df["y"].to_numpy(float)


# ---- response horizons [s] ------------------------------------------------------------------
GS_MIN_DIST_M = 500.0
T_VS, T_ELEV = 2.0, 1.0          # sink-rate response / elevator effect
T_TRACK, T_ROLL, T_AIL = 3.0, 1.5, 0.5
T_HDG = 1.0


def _mirror(xy):
    """Lateral dynamics are left/right symmetric, but every recorded crosswind blows from the
    left. Adding the mirrored demonstration teaches the right-hand case too."""
    x, y = xy
    return np.vstack([x, -x]), np.concatenate([y, -y])


def elevator_total(d):           # yoke + trim act on the same surface (c172p "Pitch Trim Sum")
    return d.elevator + d.elevator_trim


def _b_glideslope_roc(d, p):
    # the angle seen from the aim point blows up geometrically when close to it -> keep it out
    far = d.x < p["aim_x_m"] - GS_MIN_DIST_M
    x, y = _xy(d[far], p, FINAL, (d.gs_angle - p["glideslope_ref_deg"])[far], _lead(_smooth(d.vs_fps), T_VS)[far])
    return x, y


def _b_glideslope_elev(d, p):
    # second input = airspeed: the elevator that holds a given flight path depends on it
    vs = _smooth(d.vs_fps, 0.5)
    return _xy2(d, p, FINAL, _lead(vs, T_ELEV) - vs, _smooth(d.ias, 0.5), _smooth(elevator_total(d), 0.5), settle_s=p["settle_s"])


def _b_flare_roc(d, p):
    return _xy(d, p, FLARE, d.agl.rename("x"), _lead(_smooth(d.vs_fps, 0.5), 0.5).rename("y"))


def _b_flare_elev(d, p):
    vs = _smooth(d.vs_fps, 0.5)
    return _xy2(d, p, FLARE, _lead(vs, T_ELEV) - vs, _smooth(d.ias, 0.5), _smooth(elevator_total(d), 0.5))


def _b_track(d, p):
    # 2nd input = accumulated centreline error: pilots correct a long-standing offset more firmly
    return _mirror(_xy2(d, p, FINAL, d.loc_angle, d.loc_int, _lead(_smooth(d.track_err), T_TRACK)))


def _b_roll(d, p):
    tr = _smooth(d.track_err)
    return _mirror(_xy(d, p, FINAL, (_lead(tr, T_TRACK) - tr).rename("x"), _lead(_smooth(d.roll), T_ROLL).rename("y")))


def _b_aileron(d, p):
    # 2nd input = rudder being held: pilots cross-control (rudder to align, opposite aileron to hold the wing)
    r = _smooth(d.roll, 0.3)
    x1, x2, y = _lead(r, T_AIL) - r, _smooth(d.rudder, 0.5), _smooth(d.aileron, 0.3)
    a, b = _xy2(d, p, FINAL, x1, x2, y)
    c, e = _xy2(d, p, FLARE, x1, x2, y)
    return _mirror((np.vstack([a, c]), np.concatenate([b, e])))


def _b_rudder(d, p):
    x, y = d.hdg_err.rename("x"), _smooth(d.rudder, 0.5).rename("y")
    a, b = _xy(d, p, FINAL, x, y, settle_s=p["settle_s"])
    c, e = _xy(d, p, FLARE, x, y)
    return _mirror((np.vstack([a, c]), np.concatenate([b, e])))


def _b_throttle(d, p):
    # power setting the pilot was holding for the sink rate being flown
    return _xy(d, p, FINAL, _smooth(d.vs_fps, 2.0), _smooth(d.throttle, 1.0), settle_s=p["settle_s"])


def _b_flare_throttle(d, p):
    return _xy(d, p, FLARE, d.agl.rename("x"), d.throttle.rename("y"))


def _b_flaps(d, p):
    ok = d.airborne == 1
    return d.loc[ok, ["h", "ias"]].to_numpy(float), d.loc[ok, "flaps"].to_numpy(float)


def _rollout(d):
    return d[(d.gs_kt > 10) & (d.hdg_err.abs() < 25)]


def _b_heading(d, p):
    g = _rollout(d)
    return _mirror(_xy(g, p, LANDING, g.loc_angle, _lead(_smooth(d.hdg_err), 2.0)[g.index]))


def _b_rudder_ground(d, p):
    # 2nd input = accumulated heading error: the rudder a pilot holds against weathervaning
    h, g = _smooth(d.hdg_err, 0.5), _rollout(d)
    return _mirror(_xy2(g, p, LANDING, (_lead(h, T_HDG) - h)[g.index], g.hdg_int, _smooth(d.rudder, 0.5)[g.index]))


def _b_brakes(d, p):
    return _xy(d, p, LANDING, d.gs_kt.rename("x"), (0.5 * (d.brake_left + d.brake_right)).rename("y"))


def _b_landing_elev(d, p):
    return _xy(d, p, LANDING, d.gs_kt.rename("x"), _smooth(elevator_total(d), 0.5).rename("y"))


SPECS: list[Spec] = [
    Spec("glideslope_roc", FINAL, ["glideslope angle - desired glideslope [deg]"], "desired sink rate [ft/s]", 5, _b_glideslope_roc,
         "Paper: Glideslope rate of change ANN"),
    Spec("glideslope_elevator", FINAL, ["desired - current sink rate [ft/s]", "airspeed [kt]"], "elevator command", 8, _b_glideslope_elev,
         "Paper: Glideslope elevators(-trim) ANN"),
    Spec("flare_roc", FLARE, ["height above runway [ft]"], "desired sink rate [ft/s]", 5, _b_flare_roc),
    Spec("flare_elevator", FLARE, ["desired - current sink rate [ft/s]", "airspeed [kt]"], "elevator command", 8, _b_flare_elev),
    Spec("track", FINAL, ["centreline angle [deg]", "accumulated centreline angle [deg s]"], "desired track relative to runway [deg]", 8, _b_track,
         "Paper: Heading ANN idea applied in the air (intercept track)", odd=True),
    Spec("roll", FINAL, ["desired - current track [deg]"], "desired roll [deg]", 5, _b_roll, "Paper: Roll ANN", odd=True),
    Spec("aileron", FINAL, ["desired - current roll [deg]", "rudder command"], "aileron command", 8, _b_aileron, "Paper: Ailerons ANN", odd=True),
    Spec("rudder", FINAL, ["heading - runway heading [deg]"], "rudder command", 5, _b_rudder, "Paper: Rudder ANN", odd=True),
    Spec("throttle", FINAL, ["desired sink rate [ft/s]"], "throttle command", 5, _b_throttle,
         "Paper: Throttle ANN (fed by the glideslope ANN: these pilots hold the glidepath with power)"),
    Spec("flare_throttle", FLARE, ["height above runway [ft]"], "throttle command", 5, _b_flare_throttle),
    Spec("flaps", FINAL, ["height above runway [ft]", "airspeed [kt]"], "flaps command", 10, _b_flaps, "Paper: Flaps ANN"),
    Spec("heading", LANDING, ["centreline angle [deg]"], "desired heading relative to runway [deg]", 5, _b_heading, "Paper: Heading ANN", odd=True),
    Spec("rudder_ground", LANDING, ["desired - current heading [deg]", "accumulated heading error [deg s]"], "rudder command", 8, _b_rudder_ground, "Paper: Rudder ANN", odd=True, input_pct=(2.0, 98.0)),
    Spec("brakes", LANDING, ["ground speed [kt]"], "brake command", 5, _b_brakes),
    Spec("landing_elevator", LANDING, ["ground speed [kt]"], "elevator command", 5, _b_landing_elev),
]


def estimate_parameters(frames: dict[str, pd.DataFrame], meta: dict) -> dict:
    """Set-points the paper's pilot dialled in by hand (3 deg, 150 kt). Here they are read off
    the demonstrations: what the pilots actually held on a stable final."""
    gs, v, hf = [], [], []
    for d in frames.values():
        a = d[(d.airborne == 1) & (d.h > 80) & (d.h < 350)]
        gs.append(a.gs_angle.median())
        v.append(a.ias.median())
        vs_app = a.vs_fps.median()
        fl = d[d.airborne == 1].iloc[::-1]
        below = fl[_smooth(fl.vs_fps, 0.5) <= 0.8 * vs_app]          # last moment still at approach sink rate
        if len(below):
            hf.append(float(below.agl.iloc[0]))
    return dict(glideslope_ref_deg=float(np.median(gs)), v_ref_kt=float(np.median(v)),
                flare_height_ft=float(np.median(hf)), aim_x_m=meta["aim_x_m"], settle_s=12.0)
