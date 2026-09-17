"""Step 4 - compare the autonomous landings with the human demonstrations (cf. paper Figs. 21-32).

    python 04_evaluate.py            # every FlightGear run found in reports/fg_runs/
    python 04_evaluate.py --bench    # additionally fly a weather sweep in the headless bench

Outputs reports/04_ias_vs_pilot.png and reports/04_touchdowns.md
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ias.config import REPORTS_DIR
from ias.data import load_dataset

PANELS = [("h", "height above runway [ft]"), ("y", "cross-track [m]  (+ = right of centreline)"), ("ias", "airspeed [kt]"),
          ("vs_fps", "vertical speed [ft/s]"), ("pitch", "pitch [deg]"), ("roll", "roll [deg]"),
          ("elevator", "elevator (yoke + trim)"), ("throttle", "throttle"), ("rudder", "rudder")]


def touchdown(d, name, kind):
    g = d[d.wow == 1]
    if not len(g):
        return dict(run=name, pilot=kind, landed=False)
    i = g.index[0]
    end = d.iloc[-1]
    return dict(run=name, pilot=kind, landed=True, x_m=d.x[i], y_m=d.y[i], sink_fpm=d.vs_fps[max(i - 3, 0):i + 1].min() * 60,
                ias_kt=d.ias[i], pitch_deg=d.pitch[i], roll_deg=d.roll[i], max_y_rollout_m=d.y[i:].abs().max(),
                stop_x_m=end.x, final_gs_kt=end.gs_kt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    _, demos = load_dataset()
    for d in demos.values():
        d["elevator"] = d.elevator + d.elevator_trim

    runs = {}
    for p in sorted((REPORTS_DIR / "fg_runs").glob("ias_*.csv")):
        d = pd.read_csv(p)
        if len(d) > 300:
            runs[f"FlightGear {p.stem[4:]}"] = d
    if args.bench:
        from ias.controller import IntelligentAutopilot
        from sim.jsb_bench import C172Bench
        from sim.run_bench import fly
        bench, pilot = C172Bench(), IntelligentAutopilot()
        for label, w in {"calm": {}, "xwind 270/8": dict(wind_kt=8, wind_from_deg=270), "xwind 270/16": dict(wind_kt=16, wind_from_deg=270),
                         "xwind 090/12": dict(wind_kt=12, wind_from_deg=90), "gust 270/10+8": dict(wind_kt=10, wind_from_deg=270, gust_kt=8),
                         "headwind 360/15": dict(wind_kt=15, wind_from_deg=0), "turbulence 0.5": dict(turbulence=0.5, wind_kt=4, wind_from_deg=150)}.items():
            runs[f"bench {label}"] = fly(bench, pilot, **w)
    for d in runs.values():
        d["elevator"] = d.cmd_elevator + d.cmd_elevator_trim
        d["throttle"], d["rudder"] = d.cmd_throttle, d.cmd_rudder

    fig, axes = plt.subplots(3, 3, figsize=(21, 14))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs), 1)))
    for ax, (col, title) in zip(axes.ravel(), PANELS):
        for k, d in enumerate(demos.values()):
            ax.plot(d.x, d[col], color="0.6", lw=0.7, alpha=0.8, label="human demonstrations" if k == 0 else None)
        for c, (name, d) in zip(colors, runs.items()):
            ax.plot(d.x, d[col], color=c, lw=1.8, label=f"IAS - {name}")
        ax.set_title(title); ax.set_xlabel("distance past runway threshold [m]"); ax.grid(True); ax.set_xlim(-3100, 1500)
        if col == "y":
            ax.set_ylim(-120, 120); ax.axhspan(-30, 30, color="tab:green", alpha=0.08)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Intelligent Autopilot System (ANNs trained by imitation) vs. the human demonstrations - final approach and landing, BIKF RWY 02")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "04_ias_vs_pilot.png", dpi=70)

    table = pd.DataFrame([touchdown(d, n, "human") for n, d in demos.items()] + [touchdown(d, n, "IAS") for n, d in runs.items()])
    (REPORTS_DIR / "04_touchdowns.md").write_text(table.round(1).to_markdown(index=False))
    pd.set_option("display.width", 250)
    print(table.round(1).to_string(index=False))
    h = table[table.pilot == "human"]
    print(f"\nhuman envelope: touchdown x {h.x_m.min():.0f}..{h.x_m.max():.0f} m, |y| <= {h.y_m.abs().max():.1f} m, "
          f"sink {h.sink_fpm.min():.0f}..{h.sink_fpm.max():.0f} fpm, ias {h.ias_kt.min():.0f}..{h.ias_kt.max():.0f} kt")


if __name__ == "__main__":
    main()
