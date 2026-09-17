"""Flight data -> ANN inputs.

The *same* causal extractor is used offline (on the recorded demonstrations) and online
(on live FlightGear data) so the networks always see identically-built inputs.

Raw observation dict (one per 0.1 s tick), FlightGear units:
    lat, lon [deg] · alt_ft (MSL) · agl_ft · roll, pitch, heading [deg]
    roll_rate, pitch_rate, yaw_rate [deg/s] · ias [kt] · gs_kt · vs_fps
    vn_fps, ve_fps (north/east ground velocity) · wow (0/1)
"""
from __future__ import annotations

import math

from .config import DT, Runway

CG_HEIGHT_FT = 3.5          # altitude-agl-ft reads ~3.5 ft with the wheels on the ground
RATE_TAU_S = 0.5            # low-pass time constant for numerically differentiated signals
LOC_MEMORY_S = 30.0         # "how far and how long have I been off the centreline" (leaky integral)
HDG_MEMORY_S = 10.0         # same idea for the heading error during the ground roll (weathervaning)


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class _Derivative:
    """Causal first-order-filtered derivative."""

    def __init__(self, tau: float = RATE_TAU_S, wrap: bool = False):
        self.alpha = DT / (tau + DT)
        self.prev = None
        self.out = 0.0
        self.wrap = wrap

    def __call__(self, v: float, dt: float = DT) -> float:
        if self.prev is not None and dt > 1e-6:
            dv = v - self.prev
            if self.wrap:
                dv = wrap180(dv)
            self.out += self.alpha * (dv / dt - self.out)
        self.prev = v
        return self.out


class FeatureExtractor:
    def __init__(self, runway: Runway, aim_x_m: float, cg_height_ft: float = CG_HEIGHT_FT):
        self.rwy = runway
        self.aim_x = aim_x_m
        self.cg_h = cg_height_ft
        self.reset()

    def reset(self):
        self._d_gs = _Derivative()
        self._d_ias = _Derivative(tau=1.0)
        self._gs_angle = 0.0
        self._loc_int = 0.0
        self._hdg_int = 0.0

    def __call__(self, o: dict, dt: float = DT) -> dict:
        r = self.rwy
        x, y = r.to_frame(o["lat"], o["lon"])
        h = o["alt_ft"] - r.elev_ft - self.cg_h          # wheel height above the touchdown zone [ft]
        agl = o["agl_ft"] - self.cg_h

        # Glideslope angle: elevation of the aircraft seen from the aim point (paper: "glideslope degree").
        dist_aim = self.aim_x - x
        if dist_aim > 60.0:
            self._gs_angle = math.degrees(math.atan2(h / 3.28084, dist_aim))
        gs_angle = self._gs_angle                          # frozen once (almost) over the aim point

        # Centreline angle (paper: "angle between the aircraft and the centreline"), measured
        # from the far end of the runway like a localizer. Positive = aircraft right of centreline.
        loc_angle = math.degrees(math.atan2(y, max(r.length_m - x, 300.0)))

        self._loc_int += dt * (loc_angle - self._loc_int / LOC_MEMORY_S)

        # Ground-track and heading relative to the runway direction.
        vn, ve = o.get("vn_fps", 0.0), o.get("ve_fps", 0.0)
        if math.hypot(vn, ve) > 5.0:
            track = math.degrees(math.atan2(ve, vn))
        else:
            track = o["heading"]
        track_err = wrap180(track - r.hdg)
        hdg_err = wrap180(o["heading"] - r.hdg)

        self._hdg_int = self._hdg_int + dt * (hdg_err - self._hdg_int / HDG_MEMORY_S) if o["wow"] else 0.0

        return dict(
            x=x, y=y, h=h, agl=agl,
            gs_angle=gs_angle, gs_rate=self._d_gs(gs_angle, dt),
            loc_angle=loc_angle, loc_int=self._loc_int, track_err=track_err, hdg_err=hdg_err, hdg_int=self._hdg_int,
            roll=o["roll"], pitch=o["pitch"],
            roll_rate=o["roll_rate"], pitch_rate=o["pitch_rate"], yaw_rate=o["yaw_rate"],
            ias=o["ias"], ias_rate=self._d_ias(o["ias"], dt),
            gs_kt=o["gs_kt"], vs_fps=o["vs_fps"], wow=o["wow"],
        )
