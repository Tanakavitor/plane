"""Fly the trained IAS in the headless bench. Usage: python -m sim.run_bench [wind_kt wind_from turbulence gust]"""
import sys
import pandas as pd
from ias.controller import IntelligentAutopilot
from sim.jsb_bench import C172Bench


def fly(bench, ap, max_s=200.0, **weather):
    obs = bench.reset(**weather)
    ap.reset()
    log = []
    while bench.t < max_s:
        cmd = ap.step(obs)
        log.append({**obs, **{f"cmd_{k}": v for k, v in cmd.items()}, **ap.debug})
        obs = bench.step(cmd)
        if ap.phase == "landing" and obs["gs_kt"] < 3:
            break
        if obs["agl_ft"] < 1.0 or abs(obs["roll"]) > 80:
            break
    return pd.DataFrame(log)


def summarize(log):
    air = log[log.phase != "landing"]
    td = log[log.wow == 1]
    out = dict(landed=len(td) > 0)
    if len(td):
        i = td.index[0]
        out.update(td_t=log.t[i], td_x=log.x[i], td_y=log.y[i], td_vs=log.vs_fps[max(i - 2, 0):i + 1].min(), td_pitch=log.pitch[i],
                   td_ias=log.ias[i], td_roll=log.roll[i], td_hdg_err=log.hdg_err[i], end_x=log.x.iloc[-1], end_y=log.y.iloc[-1],
                   end_gs=log.gs_kt.iloc[-1], max_abs_y_ground=log.y[i:].abs().max())
    out.update(ias_mean=air.ias[air.t > 20].mean(), max_abs_y_air=air.y.abs().max())
    return out


if __name__ == "__main__":
    a = [float(v) for v in sys.argv[1:]] + [0, 0, 0, 0]
    log = fly(C172Bench(), IntelligentAutopilot(), wind_kt=a[0], wind_from_deg=a[1], turbulence=a[2], gust_kt=a[3])
    cols = ["t", "phase", "x", "y", "h", "ias", "vs_fps", "pitch", "roll", "hdg_err", "gs_angle", "vs_des", "cmd_elevator", "cmd_throttle", "cmd_aileron", "cmd_rudder", "cmd_flaps", "cmd_brake"]
    print(log[cols].iloc[::50].round(2).to_string(index=False))
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in summarize(log).items()})
    log.to_csv("reports/bench_last_run.csv", index=False)
