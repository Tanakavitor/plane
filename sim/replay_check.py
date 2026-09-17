"""Bench fidelity check: replay a pilot's recorded controls open-loop and compare with FlightGear."""
import sys
import numpy as np, pandas as pd
from ias.data import load_dataset
from sim.jsb_bench import C172Bench

meta, fr = load_dataset()
b = C172Bench()
for name in sys.argv[1:] or ["voo-normal2", "voo2-com-8-nos"]:
    d = fr[name]
    b.reset(wind_kt=float(d.wind_kt.iloc[0]), wind_from_deg=float(d.wind_from.iloc[0]),
            ic=dict(heading=float(d.heading.iloc[0]), ias_kt=float(d.ias.iloc[0])))
    rows = [b.step(dict(aileron=r.aileron, elevator=r.elevator, rudder=r.rudder, throttle=r.throttle, flaps=r.flaps,
                        brake=0, elevator_trim=r.elevator_trim)) for r in d.iloc[:400].itertuples()]
    s = pd.DataFrame(rows)
    k = slice(0, 400, 50)
    print(name)
    print(pd.DataFrame({"t": d.t[k].values, "ias_fg": d.ias.shift(-1)[k].values, "ias_jsb": s.ias[k].values,
                        "alt_fg": d.alt_ft.shift(-1)[k].values, "alt_jsb": s.alt_ft[k].values,
                        "pitch_fg": d.pitch.shift(-1)[k].values, "pitch_jsb": s.pitch[k].values,
                        "vs_fg": d.vs_fps.shift(-1)[k].values, "vs_jsb": s.vs_fps[k].values,
                        "roll_fg": d.roll.shift(-1)[k].values, "roll_jsb": s.roll[k].values, "rpm": s.rpm[k].values}).round(1).to_string(index=False))
