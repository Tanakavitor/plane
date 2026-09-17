"""The "Interface" box of the paper (Fig. 4): moves flight data and control commands between
FlightGear and the IAS over UDP, using FlightGear's generic protocol (fg/ias_out.xml, fg/ias_in.xml).

    FlightGear --(flight data, UDP 5501)--> IAS --(control commands, UDP 5502)--> FlightGear
"""
from __future__ import annotations

import socket

DATA_PORT = 5501        # FlightGear -> IAS
COMMAND_PORT = 5502     # IAS -> FlightGear
HOST = "127.0.0.1"

# must match the chunk order in fg/ias_out.xml and fg/ias_in.xml
OUT_FIELDS = ["sim_time", "lat", "lon", "alt_ft", "agl_ft", "roll", "pitch", "heading", "roll_rate", "pitch_rate",
              "yaw_rate", "ias", "gs_kt", "vs_fps", "vn_fps", "ve_fps", "wow0", "wow1", "wow2", "elevator_trim",
              "rpm", "crashed", "wind_kt", "wind_from"]
IN_FIELDS = ["aileron", "elevator", "rudder", "throttle", "throttle0", "flaps", "brake_left", "brake_right"]


class FlightGearLink:
    def __init__(self, host: str = HOST, data_port: int = DATA_PORT, command_port: int = COMMAND_PORT):
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.rx.bind((host, data_port))
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target = (host, command_port)

    def receive(self, timeout: float | None = None) -> dict | None:
        """Newest flight-data packet (older queued packets are dropped). None on timeout."""
        self.rx.settimeout(timeout)
        try:
            data, _ = self.rx.recvfrom(4096)
        except socket.timeout:
            return None
        self.rx.setblocking(False)
        try:
            while True:
                data, _ = self.rx.recvfrom(4096)
        except BlockingIOError:
            pass
        line = data.decode("ascii", "ignore").strip().splitlines()[-1]
        values = line.split(",")
        if len(values) != len(OUT_FIELDS):
            return None
        o = {k: float(v) for k, v in zip(OUT_FIELDS, values)}
        o["wow"] = int(o["wow0"] or o["wow1"] or o["wow2"])
        o["t"] = o["sim_time"]
        return o

    def send(self, c: dict):
        v = dict(aileron=c["aileron"], elevator=c["elevator"], rudder=c["rudder"], throttle=c["throttle"],
                 throttle0=c["throttle"], flaps=c["flaps"], brake_left=c["brake"], brake_right=c["brake"])
        self.tx.sendto((",".join(f"{v[k]:.5f}" for k in IN_FIELDS) + "\n").encode("ascii"), self.target)

    def close(self):
        self.rx.close()
        self.tx.close()
