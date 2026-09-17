"""Step 3 - autonomous final approach and landing in FlightGear (SIMULATION ONLY).

Start this first, then FlightGear with ./launch_flightgear.sh. The IAS takes control as soon as
flight data arrives and logs the whole attempt to reports/fg_runs/.

    python 03_fly_flightgear.py [--runway 02] [--hz 10]
"""
import argparse
import json
import socket
import time
from datetime import datetime

import pandas as pd

from ias.config import BIKF, CONTROL_HZ, DEFAULT_RUNWAY, REPORTS_DIR
from ias.controller import IntelligentAutopilot
from ias.fg_interface import FlightGearLink
from ias.networks import LANDING


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runway", default=DEFAULT_RUNWAY, choices=list(BIKF))
    ap.add_argument("--hz", type=float, default=CONTROL_HZ, help="control rate (paper: 10 Hz)")
    args = ap.parse_args()

    link = FlightGearLink()
    monitor = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)       # telemetry for 05_monitor.py (fire and forget)
    pilot = IntelligentAutopilot(runway=BIKF[args.runway])
    print(f"IAS ready - runway {args.runway}. Waiting for FlightGear flight data on UDP 5501 ...")

    log, last_t, last_print, stopped_since = [], None, 0.0, None
    try:
        while True:
            obs = link.receive(timeout=5.0)
            if obs is None:
                if log:
                    print("no data from FlightGear for 5 s - stopping.")
                    break
                continue
            if last_t is None:
                # FlightGear streams while it is still loading scenery: wait for a real, flying aircraft
                if obs["ias"] < 30.0 or obs["agl_ft"] < 100.0 or obs["t"] <= 0.0:
                    continue
                print("FlightGear connected - IAS has control.")
                pilot.trim = obs["elevator_trim"]                 # leave the trim wheel where it is
            elif obs["t"] - last_t < 0.5 / args.hz:
                continue                                          # sim paused or duplicate frame
            dt = 1.0 / args.hz if last_t is None else min(max(obs["t"] - last_t, 0.02), 0.5)
            last_t = obs["t"]

            cmd = pilot.step(obs, dt)
            link.send(cmd)
            f = pilot.debug
            row = {**obs, **{f"cmd_{k}": v for k, v in cmd.items()}, **{k: v for k, v in f.items() if k not in obs}}
            log.append(row)
            monitor.sendto(json.dumps({**row, "cmd_elevator_total": cmd["elevator"] + cmd["elevator_trim"]}).encode(), ("127.0.0.1", 5503))

            if time.time() - last_print > 1.0:
                last_print = time.time()
                print(f"t={obs['t']:6.1f}s {pilot.phase:14s} dist={-f['x']:6.0f} m  y={f['y']:+6.1f} m  h={f['h']:5.0f} ft  "
                      f"ias={obs['ias']:4.0f} kt  vs={obs['vs_fps'] * 60:+5.0f} fpm | elev={cmd['elevator']:+.2f} ail={cmd['aileron']:+.2f} "
                      f"rud={cmd['rudder']:+.2f} thr={cmd['throttle']:.2f} flaps={cmd['flaps']:.2f} brk={cmd['brake']:.2f}")

            if obs["crashed"]:
                print("FlightGear reports a crash.")
                break
            if pilot.phase == LANDING and obs["gs_kt"] < 2.0:
                stopped_since = stopped_since or time.time()
                if time.time() - stopped_since > 3.0:
                    print("Aircraft stopped on the ground - landing complete.")
                    break
    except KeyboardInterrupt:
        print("interrupted.")
    finally:
        link.close()

    if log:
        out = REPORTS_DIR / "fg_runs"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"ias_{datetime.now():%Y%m%d_%H%M%S}.csv"
        d = pd.DataFrame(log)
        d.to_csv(path, index=False)
        td = d[d.wow == 1]
        if len(td):
            i = td.index[0]
            print(f"touchdown: {d.x[i]:.0f} m past the threshold, {d.y[i]:+.1f} m from the centreline, "
                  f"{d.vs_fps[max(i - 3, 0):i + 1].min() * 60:+.0f} fpm, {d.ias[i]:.0f} kt, pitch {d.pitch[i]:+.1f} deg")
        print("log saved to", path)


if __name__ == "__main__":
    main()
