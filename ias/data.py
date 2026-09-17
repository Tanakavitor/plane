"""Step A of the paper (data collection) for already-recorded FlightGear CSVs:
load -> find every final-approach/landing demonstration -> resample to 10 Hz -> features.

A file may contain several demonstrations (e.g. circuito-completo.csv = approach, landing,
take-off, circuit, approach, landing on another runway), so segmentation is driven by
*touchdown events* and runway geometry, never by file boundaries.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .config import BIKF, DT, PROCESSED_DIR, RAW_DIR, Runway
from .features import CG_HEIGHT_FT, FeatureExtractor, wrap180

# raw FlightGear property -> short name
OBS_COLS = {
    "/position/latitude-deg": "lat", "/position/longitude-deg": "lon",
    "/position/altitude-ft": "alt_ft", "/position/altitude-agl-ft": "agl_ft",
    "/orientation/roll-deg": "roll", "/orientation/pitch-deg": "pitch",
    "/orientation/heading-deg": "heading",
    "/orientation/roll-rate-degps": "roll_rate", "/orientation/pitch-rate-degps": "pitch_rate",
    "/orientation/yaw-rate-degps": "yaw_rate",
    "/velocities/airspeed-kt": "ias", "/velocities/groundspeed-kt": "gs_kt",
    "/velocities/vertical-speed-fps": "vs_fps",
    "/velocities/speed-north-fps": "vn_fps", "/velocities/speed-east-fps": "ve_fps",
    "/environment/wind-speed-kt": "wind_kt", "/environment/wind-from-heading-deg": "wind_from",
    "/environment/turbulence/magnitude-norm": "turbulence",
}
ACT_COLS = {
    "/controls/flight/aileron": "aileron", "/controls/flight/elevator": "elevator",
    "/controls/flight/rudder": "rudder", "/controls/flight/elevator-trim": "elevator_trim",
    "/controls/engines/engine/throttle": "throttle", "/controls/flight/flaps": "flaps",
    "/controls/gear/brake-left": "brake_left", "/controls/gear/brake-right": "brake_right",
}
WOW_COLS = ["/gear/gear/wow", "/gear/gear[1]/wow", "/gear/gear[2]/wow"]

MIN_AIRBORNE_S = 20.0      # a touchdown only counts after this much continuous flight
MAX_TRACK_ERR_DEG = 35.0   # "on final": flying roughly along the runway direction ...
MAX_CROSS_TRACK_M = 400.0  # ... near the extended centreline ...
MAX_DIST_M = 3300.0        # ... within the distance covered by the standard demonstrations
GO_THROTTLE = 0.8          # throttle above this on the ground = touch-and-go, demo is over


@dataclass
class Segment:
    name: str
    source: str
    runway: str
    t_start: float
    t_touchdown: float
    t_end: float
    touchdown_x_m: float
    touchdown_y_m: float
    touchdown_vs_fps: float
    wind_kt: float
    wind_from: float
    turbulence: float
    n_samples: int = 0


def load_raw(path) -> pd.DataFrame:
    want = set(OBS_COLS) | set(ACT_COLS) | set(WOW_COLS) | {"sim_time"}
    strip = lambda c: c.replace("[0]", "")
    d = pd.read_csv(path, usecols=lambda c: strip(c) in want)
    d.columns = [strip(c) for c in d.columns]
    out = pd.DataFrame({"t": d["sim_time"]})
    for src, dst in {**OBS_COLS, **ACT_COLS}.items():
        out[dst] = d[src].astype(float)
    out["wow_main"] = d[WOW_COLS[1:]].max(axis=1).astype(int)
    out["wow"] = d[WOW_COLS].max(axis=1).astype(int)
    return out


def _touchdowns(d: pd.DataFrame) -> list[int]:
    """Indices of first main-gear contact after >= MIN_AIRBORNE_S of flight (bounces ignored)."""
    t, wow = d["t"].to_numpy(), d["wow"].to_numpy()
    out, last_ground_t = [], -1e9 if wow[0] == 0 else t[0]
    airborne_since = t[0] if wow[0] == 0 else None
    for i in range(1, len(d)):
        if wow[i] and not wow[i - 1]:
            if airborne_since is not None and t[i] - airborne_since >= MIN_AIRBORNE_S:
                out.append(i)
            airborne_since = None
        elif not wow[i] and wow[i - 1]:
            airborne_since = t[i]
    return out


def _pick_runway(d: pd.DataFrame, i: int) -> Runway | None:
    best, best_y = None, 1e9
    for r in BIKF.values():
        x, y = r.to_frame(d["lat"].iat[i], d["lon"].iat[i])
        if -200 < x < r.length_m and abs(wrap180(d["heading"].iat[i] - r.hdg)) < 30 and abs(y) < best_y:
            best, best_y = r, abs(y)
    return best if best_y < 60 else None


def find_segments(d: pd.DataFrame, source: str) -> list[Segment]:
    segs = []
    t = d["t"].to_numpy()
    for k, i_td in enumerate(_touchdowns(d)):
        rwy = _pick_runway(d, i_td)
        if rwy is None:
            continue
        xy = np.array([rwy.to_frame(a, b) for a, b in zip(d["lat"], d["lon"])])
        x, y = xy[:, 0], xy[:, 1]
        track = np.degrees(np.arctan2(d["ve_fps"], d["vn_fps"]))
        terr = np.abs((track - rwy.hdg + 180) % 360 - 180)
        on_final = (terr < MAX_TRACK_ERR_DEG) & (np.abs(y) < MAX_CROSS_TRACK_M) & (x > -MAX_DIST_M) & (d["wow"].to_numpy() == 0)
        i0 = i_td - 1
        while i0 > 0 and on_final[i0 - 1]:
            i0 -= 1
        # end of the demonstration: stopped, throttle-up for a touch-and-go, or end of file
        i1 = len(d) - 1
        thr, gs = d["throttle"].to_numpy(), d["gs_kt"].to_numpy()
        for j in range(i_td, len(d)):
            if thr[j] > GO_THROTTLE and d["wow"].iat[j]:
                i1 = max(i_td, j - int(1.0 / np.median(np.diff(t))))
                break
            if gs[j] < 3.0:
                i1 = j
                break
        name = source if k == 0 and len(_touchdowns(d)) == 1 else f"{source}#{k + 1}"
        w = slice(i0, i_td)
        segs.append(Segment(
            name=name, source=source, runway=rwy.name,
            t_start=float(t[i0]), t_touchdown=float(t[i_td]), t_end=float(t[i1]),
            touchdown_x_m=float(x[i_td]), touchdown_y_m=float(y[i_td]),
            touchdown_vs_fps=float(d["vs_fps"].iloc[max(i_td - 6, 0):i_td + 1].min()),
            wind_kt=float(d["wind_kt"].iloc[w].mean()), wind_from=float(d["wind_from"].iloc[w].median()),
            turbulence=float(d["turbulence"].iloc[w].max()),
        ))
    return segs


def resample(d: pd.DataFrame, t0: float, t1: float, dt: float = DT) -> pd.DataFrame:
    """Average the ~60 Hz recording into dt-wide bins (what a 10 Hz interface would see)."""
    s = d[(d["t"] >= t0) & (d["t"] <= t1)].copy()
    s["heading"] = np.degrees(np.unwrap(np.radians(s["heading"])))
    s["bin"] = np.floor((s["t"] - t0) / dt).astype(int)
    g = s.groupby("bin")
    out = g.mean(numeric_only=True)
    for c in ("flaps", "wow", "wow_main"):            # discrete signals: no averaging
        out[c] = g[c].last()
    out["heading"] = out["heading"] % 360.0
    out["t"] = out.index * dt
    return out.reset_index(drop=True)


def estimate_aim_point(frames: list[tuple[pd.DataFrame, Runway]]) -> tuple[float, float]:
    """Where, and at what angle, the demonstrated descent paths meet the runway."""
    xs, angs = [], []
    for s, rwy in frames:
        xy = np.array([rwy.to_frame(a, b) for a, b in zip(s["lat"], s["lon"])])
        h_m = (s["alt_ft"].to_numpy() - rwy.elev_ft - CG_HEIGHT_FT) / 3.28084
        m = (h_m > 30) & (h_m < 140) & (s["wow"].to_numpy() == 0)
        if m.sum() < 50:
            continue
        slope, icpt = np.polyfit(xy[m, 0], h_m[m], 1)
        xs.append(-icpt / slope)
        angs.append(np.degrees(np.arctan(-slope)))
    return float(np.median(xs)), float(np.median(angs))


def build_dataset(verbose: bool = True) -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    staged = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        raw = load_raw(path)
        for seg in find_segments(raw, path.stem):
            staged.append((seg, resample(raw, seg.t_start, seg.t_end)))
    aim_x, gs_ref = estimate_aim_point([(s, BIKF[seg.runway]) for seg, s in staged])

    segments = []
    for seg, s in staged:
        fx = FeatureExtractor(BIKF[seg.runway], aim_x)
        feats = pd.DataFrame([fx(row) for row in s.to_dict("records")])
        keep = [c for c in s.columns if c not in feats.columns]
        full = pd.concat([s[keep], feats], axis=1)
        full["airborne"] = (full["t"] < seg.t_touchdown - seg.t_start).astype(int)
        full.to_csv(PROCESSED_DIR / f"{seg.name.replace('#', '_')}.csv", index=False)
        seg.n_samples = len(full)
        segments.append(asdict(seg))
        if verbose:
            print(f"{seg.name:36s} rwy {seg.runway}  t=[{seg.t_start:6.1f},{seg.t_end:6.1f}]  "
                  f"touchdown t={seg.t_touchdown:6.1f} x={seg.touchdown_x_m:5.0f} m y={seg.touchdown_y_m:+5.1f} m "
                  f"vs={seg.touchdown_vs_fps:+5.1f} fps  wind {seg.wind_kt:4.1f} kt/{seg.wind_from:3.0f}  "
                  f"turb {seg.turbulence:.2f}  n={seg.n_samples}")
    meta = dict(aim_x_m=aim_x, glideslope_ref_deg=gs_ref, dt=DT, segments=segments)
    (PROCESSED_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
    if verbose:
        print(f"\naim point {aim_x:.0f} m past the threshold, demonstrated glideslope {gs_ref:.2f} deg")
    return meta


def load_dataset() -> tuple[dict, dict[str, pd.DataFrame]]:
    meta = json.loads((PROCESSED_DIR / "meta.json").read_text())
    frames = {s["name"]: pd.read_csv(PROCESSED_DIR / f"{s['name'].replace('#', '_')}.csv") for s in meta["segments"]}
    return meta, frames
