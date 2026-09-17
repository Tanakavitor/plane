"""Static configuration: runway geometry, file locations, loop timing.

Simulation-only project (FlightGear + C172P). Nothing here is meant for a real aircraft.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "voos-piloto"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# The paper exchanges data with the simulator every 0.1 s (Section 3.A.1).
CONTROL_HZ = 10.0
DT = 1.0 / CONTROL_HZ

FT_PER_M = 3.28084
KT_PER_MPS = 1.943844


@dataclass(frozen=True)
class Runway:
    """One landing direction of a runway. Coordinates come from FlightGear's
    Scenery/Airports/B/I/K/BIKF.threshold.xml so they match what the sim draws."""

    name: str
    lat: float          # threshold latitude  [deg]
    lon: float          # threshold longitude [deg]
    hdg: float          # true landing heading [deg]
    length_m: float
    elev_ft: float      # touchdown-zone elevation (median of recorded on-ground samples)

    # WGS84 metres per degree at the threshold latitude
    @property
    def m_per_deg_lat(self) -> float:
        p = math.radians(self.lat)
        return 111132.92 - 559.82 * math.cos(2 * p) + 1.175 * math.cos(4 * p)

    @property
    def m_per_deg_lon(self) -> float:
        p = math.radians(self.lat)
        return 111412.84 * math.cos(p) - 93.5 * math.cos(3 * p)

    def to_frame(self, lat, lon):
        """(lat, lon) -> (x, y) metres. x: along the landing direction, 0 at the
        threshold, negative before it. y: cross-track, positive = right of centreline."""
        dn = (lat - self.lat) * self.m_per_deg_lat
        de = (lon - self.lon) * self.m_per_deg_lon
        h = math.radians(self.hdg)
        x = dn * math.cos(h) + de * math.sin(h)
        y = -dn * math.sin(h) + de * math.cos(h)
        return x, y


BIKF = {
    "02": Runway("02", 63.96448319, -22.60544552, 0.02, 3054.0, 143.0),
    "20": Runway("20", 63.99188019, -22.60542552, 180.02, 3054.0, 143.0),
    "11": Runway("11", 63.98504094, -22.65499831, 89.97, 3065.0, 147.0),
    "29": Runway("29", 63.98504419, -22.59238152, 270.02, 3065.0, 147.0),
}
DEFAULT_RUNWAY = "02"

# Initial condition shared by the recorded demonstrations (first row of the CSVs).
INITIAL_CONDITION = dict(
    lat=63.93762, lon=-22.60547, alt_ft=801.0, heading=0.0, ias_kt=82.0,
    elevator_trim=-0.03,
)


def start_condition(dist_m: float = 2995.0, offset_m: float = 0.0, alt_ft: float = 801.0, heading: float = 0.0,
                    ias_kt: float = 82.0, runway: str = DEFAULT_RUNWAY) -> dict:
    """An initial condition described relative to the runway: `dist_m` before the threshold on the
    extended centreline, `offset_m` to the right of it (negative = left). Defaults = the recordings."""
    r = BIKF[runway]
    h = math.radians(r.hdg)
    dn = -dist_m * math.cos(h) - offset_m * math.sin(h)
    de = -dist_m * math.sin(h) + offset_m * math.cos(h)
    return dict(lat=r.lat + dn / r.m_per_deg_lat, lon=r.lon + de / r.m_per_deg_lon, alt_ft=alt_ft,
                heading=heading, ias_kt=ias_kt, elevator_trim=INITIAL_CONDITION["elevator_trim"])


if __name__ == "__main__":      # used by launch_flightgear.sh:  python -m ias.config DIST OFFSET  ->  "lat lon"
    import sys
    c = start_condition(float(sys.argv[1]), float(sys.argv[2]))
    print(f"{c['lat']:.6f} {c['lon']:.6f}")
