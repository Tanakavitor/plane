"""Live view of what the Intelligent Autopilot System is commanding.

    python 05_monitor.py                                   # live: run next to 03_fly_flightgear.py
    python 05_monitor.py --replay reports/fg_runs/ias_XXXX.csv [--speed 4]
    python 05_monitor.py --replay <run.csv> --snapshot out.png   # render the last frame to a file

Left: the controls exactly as sent to FlightGear (yoke, rudder, throttle, flaps, brakes).
Middle: what the ANN cascades want vs. what the aircraft is doing.  Right: the flight path
over the human demonstrations (grey).
"""
import argparse
import json
import socket
import time
from collections import deque

import matplotlib
import numpy as np
import pandas as pd

MONITOR_PORT = 5503
HISTORY_S = 60.0


class Source:
    """Telemetry either from the running IAS (UDP) or from a saved run (CSV)."""

    def __init__(self, replay=None, speed=1.0):
        self.rows = None
        if replay:
            d = pd.read_csv(replay)
            d["cmd_elevator_total"] = d.cmd_elevator + d.cmd_elevator_trim
            self.rows, self.i, self.t0, self.speed = d.to_dict("records"), 0, time.time(), speed
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(("127.0.0.1", MONITOR_PORT))
            self.sock.setblocking(False)

    def poll(self, everything=False):
        out = []
        if self.rows is not None:
            limit = len(self.rows) if everything else None
            t_play = self.rows[0]["t"] + (time.time() - self.t0) * self.speed
            while self.i < len(self.rows) and (limit or self.rows[self.i]["t"] <= t_play):
                out.append(self.rows[self.i]); self.i += 1
        else:
            try:
                while True:
                    out.append(json.loads(self.sock.recv(65536).decode()))
            except BlockingIOError:
                pass
        return out


def build(plt, demos):
    fig = plt.figure(figsize=(15, 8.5))
    fig.canvas.manager.set_window_title("Intelligent Autopilot System - live monitor") if fig.canvas.manager else None
    gs = fig.add_gridspec(4, 3, width_ratios=[1.0, 1.6, 1.6], hspace=0.55, wspace=0.28, left=0.05, right=0.98, top=0.93, bottom=0.07)
    a = {}
    a["yoke"] = fig.add_subplot(gs[0:2, 0]); a["yoke"].set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal", title="yoke   (push = up, pull = down)")
    a["yoke"].axhline(0, color="0.8"); a["yoke"].axvline(0, color="0.8")
    a["bars"] = fig.add_subplot(gs[2:4, 0]); a["bars"].set(xlim=(-1, 1), ylim=(-0.5, 3.5), title="rudder / throttle / flaps / brakes")
    a["bars"].set_yticks([0, 1, 2, 3], ["brakes", "flaps", "throttle", "rudder"]); a["bars"].axvline(0, color="0.8")
    for k, (key, title) in enumerate([("pitch", "elevator command  &  sink rate: wanted vs actual [ft/s]"), ("roll", "aileron command  &  roll: wanted vs actual [deg]"),
                                      ("yaw", "rudder command  &  heading - runway heading [deg]"), ("power", "throttle command  &  airspeed [kt]")]):
        a[key] = fig.add_subplot(gs[k, 1]); a[key].set_title(title, fontsize=9); a[key].grid(True, alpha=0.4)
        a[key + "2"] = a[key].twinx()
    a["side"] = fig.add_subplot(gs[0:2, 2]); a["side"].set(title="side view: height above runway [ft]", xlim=(-3100, 1500), ylim=(-20, 800)); a["side"].grid(True, alpha=0.4)
    a["top"] = fig.add_subplot(gs[2:4, 2]); a["top"].set(title="top view: cross-track [m]", xlim=(-3100, 1500), ylim=(-120, 120), xlabel="distance past threshold [m]"); a["top"].grid(True, alpha=0.4)
    a["top"].axhspan(-30, 30, xmin=(3100) / 4600, color="0.85")
    for d in demos:
        a["side"].plot(d.x, d.h, color="0.75", lw=0.6); a["top"].plot(d.x, d.y, color="0.75", lw=0.6)

    art = dict(
        yoke=a["yoke"].plot([0], [0], "o", ms=16, color="tab:red")[0],
        bars=a["bars"].barh([3, 2, 1, 0], [0, 0, 0, 0], color=["tab:purple", "tab:green", "tab:blue", "tab:red"]),
        side=a["side"].plot([], [], color="tab:red", lw=2)[0], top=a["top"].plot([], [], color="tab:red", lw=2)[0],
        side_pt=a["side"].plot([], [], "o", color="tab:red")[0], top_pt=a["top"].plot([], [], "o", color="tab:red")[0],
        title=fig.suptitle("waiting for the IAS ...", fontsize=13, fontweight="bold"),
    )
    lines = {}
    for key, cmd_color in (("pitch", "tab:red"), ("roll", "tab:red"), ("yaw", "tab:red"), ("power", "tab:red")):
        lines[key + "_cmd"] = a[key].plot([], [], color=cmd_color, lw=1.6, label="command")[0]
        lines[key + "_want"] = a[key + "2"].plot([], [], color="tab:blue", lw=1.2, ls="--", label="ANN wants")[0]
        lines[key + "_is"] = a[key + "2"].plot([], [], color="tab:blue", lw=1.2, label="actual")[0]
        a[key].set_ylim(-1, 1)
    a["power"].set_ylim(0, 1)
    a["pitch"].legend(loc="upper left", fontsize=7); a["pitch2"].legend(loc="upper right", fontsize=7)
    return fig, a, art, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay"); ap.add_argument("--speed", type=float, default=1.0); ap.add_argument("--snapshot")
    args = ap.parse_args()
    if args.snapshot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from ias.data import load_dataset

    fig, a, art, lines = build(plt, list(load_dataset()[1].values()))
    src = Source(args.replay, args.speed)
    hist = deque()
    path = dict(x=[], h=[], y=[])
    nan = float("nan")

    def update(_=None, everything=False):
        rows = src.poll(everything)
        if not rows:
            return
        for r in rows:
            if hist and r["t"] < hist[-1]["t"] - 1.0:          # new flight: start over
                hist.clear(); [v.clear() for v in path.values()]
            hist.append(r); path["x"].append(r["x"]); path["h"].append(r["h"]); path["y"].append(r["y"])
        while hist and hist[-1]["t"] - hist[0]["t"] > HISTORY_S:
            hist.popleft()
        r = hist[-1]
        g = lambda k, row=None: (row or r).get(k) if (row or r).get(k) is not None else nan
        t = np.array([h["t"] for h in hist]) - r["t"]
        col = lambda k: np.array([g(k, h) for h in hist], float)

        art["yoke"].set_data([g("cmd_aileron")], [g("cmd_elevator_total")])
        for bar, v in zip(art["bars"], (g("cmd_rudder"), g("cmd_throttle"), g("cmd_flaps"), g("cmd_brake"))):
            bar.set_width(v)
        for key, cmd, want, actual in (("pitch", "cmd_elevator_total", "vs_des", "vs_fps"), ("roll", "cmd_aileron", "roll_des", "roll"),
                                       ("yaw", "cmd_rudder", "hdg_des", "hdg_err"), ("power", "cmd_throttle", None, "ias")):
            lines[key + "_cmd"].set_data(t, col(cmd))
            lines[key + "_is"].set_data(t, col(actual))
            lines[key + "_want"].set_data(t, col(want) if want else np.full_like(t, nan))
            a[key].set_xlim(-HISTORY_S, 0)
            vals = np.concatenate([col(actual), col(want) if want else []]); vals = vals[np.isfinite(vals)]
            if len(vals):
                a[key + "2"].set_ylim(vals.min() - 1, vals.max() + 1)
        art["side"].set_data(path["x"], path["h"]); art["top"].set_data(path["x"], path["y"])
        art["side_pt"].set_data([r["x"]], [r["h"]]); art["top_pt"].set_data([r["x"]], [r["y"]])
        art["title"].set_text(f"{str(g('phase')).replace('_', ' ').upper()}   |   {-r['x']:.0f} m to threshold   h {r['h']:.0f} ft   "
                              f"{g('ias'):.0f} kt   {g('vs_fps') * 60:+.0f} fpm   cross-track {r['y']:+.1f} m")

    if args.snapshot:
        update(everything=True)
        fig.savefig(args.snapshot, dpi=80)
        print("saved", args.snapshot)
        return
    anim = FuncAnimation(fig, update, interval=150, cache_frame_data=False)   # noqa: F841 (keep a reference)
    plt.show()


if __name__ == "__main__":
    main()
