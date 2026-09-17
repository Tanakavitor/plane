"""How far from the demonstrated initial condition can the IAS start?  python -m sim.ic_sweep"""
import pandas as pd
from ias.config import start_condition
from ias.controller import IntelligentAutopilot
from sim.jsb_bench import C172Bench
from sim.run_bench import fly, summarize

CASES = {
    "as recorded": {}, "100 m right": dict(offset_m=100), "200 m left": dict(offset_m=-200), "400 m right": dict(offset_m=400),
    "heading 15 deg off": dict(heading=15), "heading 30 deg off": dict(heading=-30),
    "high 1000 ft": dict(alt_ft=1000), "very high 1300 ft": dict(alt_ft=1300), "low 600 ft": dict(alt_ft=600), "low 450 ft": dict(alt_ft=450),
    "slow 65 kt": dict(ias_kt=65), "fast 105 kt": dict(ias_kt=105),
    "close 2000 m / 600 ft": dict(dist_m=2000, alt_ft=600), "close 1200 m / 450 ft": dict(dist_m=1200, alt_ft=450),
    "far 4500 m / 1000 ft": dict(dist_m=4500, alt_ft=1000), "far 6000 m / 1200 ft": dict(dist_m=6000, alt_ft=1200),
    "combined: 150 m left, 950 ft, hdg 10, 95 kt": dict(offset_m=-150, alt_ft=950, heading=10, ias_kt=95),
}

if __name__ == "__main__":
    bench, pilot = C172Bench(), IntelligentAutopilot()
    rows = []
    for name, kw in CASES.items():
        s = summarize(fly(bench, pilot, max_s=300, ic=start_condition(**kw)))
        ok = s.get("landed") and 0 < s["td_x"] < 2500 and abs(s["td_y"]) < 30 and s["td_vs"] > -8 and s["max_abs_y_ground"] < 30
        rows.append(dict(case=name, ok="YES" if ok else "no", **{k: s.get(k) for k in ("td_x", "td_y", "td_vs", "td_ias", "max_abs_y_ground", "max_abs_y_air")}))
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(1).to_string(index=False))
