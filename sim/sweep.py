"""Weather sweep of the trained IAS in the headless bench:  python -m sim.sweep"""
import pandas as pd
from ias.controller import IntelligentAutopilot
from sim.jsb_bench import C172Bench
from sim.run_bench import fly, summarize

CASES = {
    "calm": {}, "fair 150/4": dict(wind_kt=3.7, wind_from_deg=150),
    "xwind 270/8": dict(wind_kt=8, wind_from_deg=270), "xwind 270/12": dict(wind_kt=12, wind_from_deg=270),
    "xwind 270/16": dict(wind_kt=16, wind_from_deg=270), "xwind 090/12": dict(wind_kt=12, wind_from_deg=90),
    "gust 270/10+8": dict(wind_kt=10, wind_from_deg=270, gust_kt=8), "headwind 360/15": dict(wind_kt=15, wind_from_deg=0),
    "turbulence 0.3": dict(turbulence=0.3, wind_kt=4, wind_from_deg=150), "turbulence 0.6": dict(turbulence=0.6, wind_kt=4, wind_from_deg=150),
}

if __name__ == "__main__":
    bench, pilot = C172Bench(), IntelligentAutopilot()
    rows = [dict(case=name, **summarize(fly(bench, pilot, **w))) for name, w in CASES.items()]
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).drop(columns=["td_t"]).round(1).to_string(index=False))
