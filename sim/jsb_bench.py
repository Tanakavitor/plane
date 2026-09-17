"""Headless test bench: FlightGear's own C172P flight-dynamics model (JSBSim) run standalone.

sim/fdm/c172p_bench is a copy of fgdata/Aircraft/c172p/c172p.xml with the FlightGear-only
systems (fuel selector, damage, heating, icing, mooring ...) removed; aerodynamics, flight
controls, propulsion and landing gear are untouched. It exists so the controller can be
exercised thousands of times faster than real time before it is let loose in FlightGear.
Terrain is flat at runway elevation and the engine-start sequence is simplified.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import jsbsim
import numpy as np

from ias.config import BIKF, DEFAULT_RUNWAY, DT, INITIAL_CONDITION

HERE = Path(__file__).resolve().parent
FDM_ROOT = HERE / "fdm"
SIM_HZ = 120
SUBSTEPS = int(round(SIM_HZ * DT))

STUB_VALUES = {                      # FlightGear-side properties the FDM reads, with their normal values
    "/systems/electrical/outputs/flaps": 28.0,
    "/controls/engines/current-engine/mixture": 0.81,
    "aero/coefficient/spiral-propwash-coeff": 0.25,
    "/controls/flight/adv-yaw-fact1": 0.2,
    "/controls/flight/adv-yaw-fact2": 0.2,
}


class C172Bench:
    def __init__(self, runway=None):
        self.runway = runway or BIKF[DEFAULT_RUNWAY]
        jsbsim.FGJSBBase().debug_lvl = 0
        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.set_debug_level(0)
        pm = self.fdm.get_property_manager()
        stubs = json.loads((HERE / "fg_stub_bench.json").read_text())
        for k in stubs:
            v = STUB_VALUES.get(k, 0.0)
            if k.startswith("/"):
                pm.get_node(k, True).set_double_value(v)
            else:
                self.fdm.set_property_value(k, v)
        self.fdm.set_aircraft_path(str(FDM_ROOT))
        self.fdm.set_engine_path(str(FDM_ROOT / "c172p_bench" / "Engines"))
        self.fdm.set_systems_path(str(FDM_ROOT / "c172p_bench" / "Systems"))
        devnull, saved = os.open(os.devnull, os.O_WRONLY), os.dup(1)
        os.dup2(devnull, 1)                                   # JSBSim prints its banner to stdout
        try:
            assert self.fdm.load_model("c172p_bench")
        finally:
            os.dup2(saved, 1); os.close(devnull); os.close(saved)
        self.fdm.set_dt(1.0 / SIM_HZ)
        self._pm = pm

    def _set(self, k, v):
        if k.startswith("/"):
            self._pm.get_node(k, True).set_double_value(float(v))
        else:
            self.fdm[k] = float(v)

    def reset(self, wind_kt=0.0, wind_from_deg=0.0, turbulence=0.0, gust_kt=0.0, engine_delay_s=10.0,
              ic: dict | None = None, seed: int = 0):
        ic = {**INITIAL_CONDITION, **(ic or {})}
        f = self.fdm
        f["ic/lat-geod-deg"], f["ic/long-gc-deg"] = ic["lat"], ic["lon"]
        f["ic/terrain-elevation-ft"] = self.runway.elev_ft
        f["ic/h-sl-ft"] = ic["alt_ft"]
        f["ic/vc-kts"] = ic["ias_kt"]
        f["ic/psi-true-deg"] = ic["heading"]
        f["ic/gamma-deg"] = 0.0
        f["ic/alpha-deg"] = 1.0
        # the wind must exist before initialisation, otherwise the aircraft is hit by a wind step at t=0
        self.wind = (wind_kt, wind_from_deg, gust_kt)
        v = wind_kt * 1.6878
        f["ic/vw-north-fps"] = -v * math.cos(math.radians(wind_from_deg))
        f["ic/vw-east-fps"] = -v * math.sin(math.radians(wind_from_deg))
        f["ic/vw-down-fps"] = 0.0
        f["ic/vc-kts"] = ic["ias_kt"]
        f["ic/beta-deg"] = 0.0
        f.run_ic()
        self._apply_wind(0.0)
        if turbulence > 0:
            f["atmosphere/turb-type"] = 3                      # Milspec, as FlightGear uses
            f["atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps"] = 15 + 60 * turbulence
            f["atmosphere/turbulence/milspec/severity"] = int(round(1 + 5 * turbulence))
            f["atmosphere/turb-rand-seed"] = seed if "atmosphere/turb-rand-seed" in f.get_property_catalog() else 0
        f["inertia/pointmass-weight-lbs[0]"] = 170.0           # pilot, as recorded
        for tank in (0, 1):
            f[f"propulsion/tank[{tank}]/contents-lbs"] = 32.2  # 5.37 US gal per wing tank, as recorded
        # FlightGear's in-air start: the propeller windmills for ~10 s before the engine fires.
        self.engine_delay = engine_delay_s
        f["propulsion/set-running"] = 0                         # engine[0] = the 160 hp O-320 of the recordings
        self._engine_on = engine_delay_s <= 0
        f["propulsion/magneto_cmd"] = 3 if self._engine_on else 0
        self.t = 0.0
        self._rng = np.random.default_rng(seed)
        self.apply(dict(aileron=0, elevator=0, rudder=0, throttle=0.6, flaps=0, brake=0, elevator_trim=ic["elevator_trim"]))
        return self.observe()

    def _start_engine(self):
        self.fdm["propulsion/magneto_cmd"] = 3
        self._engine_on = True

    def _apply_wind(self, t):
        kt, frm, gust = self.wind
        # slow sinusoidal gusting on top of the mean wind (FlightGear's gust model is similar in effect)
        speed = kt + gust * 0.5 * (1 + math.sin(2 * math.pi * t / 17.0)) * (math.sin(2 * math.pi * t / 5.3) > -0.3)
        v = speed * 1.6878
        self.fdm["atmosphere/wind-north-fps"] = -v * math.cos(math.radians(frm))
        self.fdm["atmosphere/wind-east-fps"] = -v * math.sin(math.radians(frm))

    def apply(self, c: dict):
        s = self._set
        s("fcs/aileron-cmd-norm", c["aileron"]); s("fcs/elevator-cmd-norm", c["elevator"])
        s("fcs/rudder-cmd-norm", c["rudder"]); s("/controls/flight/rudder", c["rudder"])
        s("fcs/pitch-trim-cmd-norm", c.get("elevator_trim", INITIAL_CONDITION["elevator_trim"]))
        s("fcs/throttle-cmd-norm", c["throttle"]); s("/controls/engines/current-engine/throttle", c["throttle"])
        s("fcs/roll-trim-cmd-norm", 0.022); s("fcs/yaw-trim-cmd-norm", 0.02)   # c172p defaults, constant in every recording
        s("fcs/mixture-cmd-norm", 0.81)
        s("fcs/flap-cmd-norm", c["flaps"]); s("/controls/flight/flaps", c["flaps"])
        s("/controls/gear/brake-left", c.get("brake", 0.0)); s("/controls/gear/brake-right", c.get("brake", 0.0))
        s("fcs/left-brake-cmd-norm", c.get("brake", 0.0)); s("fcs/right-brake-cmd-norm", c.get("brake", 0.0))

    def step(self, commands: dict) -> dict:
        self.apply(commands)
        for _ in range(SUBSTEPS):
            self._apply_wind(self.t)
            if not self._engine_on and self.t >= self.engine_delay:
                self._start_engine()
            self.fdm["propulsion/tank[4]/contents-lbs"] = 0.1      # float chamber, normally refilled by FlightGear's fuel system
            self._mirror_fg_properties()
            self.fdm.run()
            self.t += 1.0 / SIM_HZ
        return self.observe()

    def _mirror_fg_properties(self):
        """FlightGear publishes these FDM outputs under its own names and the c172p FCS reads
        them back (e.g. nose-wheel steering only engages with /gear/gear[0]/compression-norm > 0)."""
        f, s = self.fdm, self._set
        for i in range(3):
            s(f"/gear/gear[{i}]/wow", f[f"gear/unit[{i}]/WOW"])
            s(f"/gear/gear[{i}]/rollspeed-ms", f["velocities/vg-fps"] * 0.3048 * f[f"gear/unit[{i}]/WOW"])
        s("/gear/gear[0]/compression-norm", f["gear/unit[0]/compression-ft"])
        s("/velocities/groundspeed-kt", f["velocities/vg-fps"] / 1.6878)
        s("/position/altitude-agl-m", f["position/h-agl-ft"] * 0.3048)

    def observe(self) -> dict:
        f = self.fdm
        deg = math.degrees
        wow = int(f["gear/unit[0]/WOW"] or f["gear/unit[1]/WOW"] or f["gear/unit[2]/WOW"])
        return dict(
            t=self.t, lat=f["position/lat-geod-deg"], lon=f["position/long-gc-deg"],
            alt_ft=f["position/h-sl-ft"], agl_ft=f["position/h-agl-ft"],
            roll=f["attitude/phi-deg"], pitch=f["attitude/theta-deg"], heading=f["attitude/psi-deg"],
            roll_rate=deg(f["velocities/p-rad_sec"]), pitch_rate=deg(f["velocities/q-rad_sec"]),
            yaw_rate=deg(f["velocities/r-rad_sec"]),
            ias=f["velocities/vc-kts"], gs_kt=f["velocities/vg-fps"] / 1.6878, vs_fps=f["velocities/h-dot-fps"],
            vn_fps=f["velocities/v-north-fps"], ve_fps=f["velocities/v-east-fps"], wow=wow,
            rpm=f["propulsion/engine/propeller-rpm"],
        )
