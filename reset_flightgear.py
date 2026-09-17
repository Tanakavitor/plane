"""Put the running FlightGear back at the start of the final approach (same state as the CSVs),
so another autonomous landing can be flown without restarting the simulator.

    python reset_flightgear.py [wind_from_deg wind_kt [gust_kt]]      e.g.  270 12 20  ->  METAR 27012G20KT
    python reset_flightgear.py --dist 4500 --alt 1000                 a different start (see --help)
"""
import argparse
import socket
import time

from ias.config import start_condition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wind", nargs="*", help="wind_from_deg wind_kt [gust_kt]")
    ap.add_argument("--dist", type=float, default=2995.0, help="metres before the runway 02 threshold")
    ap.add_argument("--offset", type=float, default=0.0, help="metres right(+)/left(-) of the centreline")
    ap.add_argument("--alt", type=float, default=801.0, help="altitude ft MSL (runway = 143 ft)")
    ap.add_argument("--heading", type=float, default=0.0)
    ap.add_argument("--speed", type=float, default=82.0, help="airspeed kt")
    args = ap.parse_args()
    a = args.wind
    IC = start_condition(args.dist, args.offset, args.alt, args.heading, args.speed)
    conn = {}

    def connect():
        for _ in range(20):                                       # FlightGear restarts its telnet server on reposition
            try:
                conn["s"] = socket.create_connection(("127.0.0.1", 5401), timeout=5)
                conn["s"].sendall(b"data\r\n")
                return
            except OSError:
                time.sleep(0.5)
        raise SystemExit("cannot reach FlightGear on telnet port 5401 - was it started with launch_flightgear.sh?")

    def send(line):
        conn["s"].sendall((line + "\r\n").encode())
        time.sleep(0.05)

    connect()
    for prop, val in {
        "/sim/presets/airport-id": "", "/sim/presets/runway": "", "/sim/presets/parkpos": "",
        "/sim/presets/latitude-deg": IC["lat"], "/sim/presets/longitude-deg": IC["lon"],
        "/sim/presets/altitude-ft": IC["alt_ft"], "/sim/presets/heading-deg": IC["heading"],
        "/sim/presets/airspeed-kt": IC["ias_kt"], "/sim/presets/offset-distance-nm": 0,
        "/sim/presets/glideslope-deg": 0, "/sim/presets/onground": "false",
    }.items():
        send(f"set {prop} {val}")
    def set_weather():
        # FlightGear's weather follows a METAR string ("Fair weather" = 15003KT, the calm recordings)
        gust = f"G{int(float(a[2])):02d}" if len(a) >= 3 else ""
        send("set /environment/weather-scenario Manual input")    # otherwise the scenario re-asserts its own METAR
        send(f"set /environment/metar/data XXXX 012345Z {int(float(a[0])):03d}{int(float(a[1])):02d}{gust}KT 12SM SCT041 FEW200 20/08 Q1015 NOSIG")

    if len(a) >= 2:
        set_weather()
        print("new wind set - FlightGear blends it in over ~30 s ...")
        time.sleep(30)
    for prop, val in {"/controls/flight/flaps": 0, "/controls/gear/brake-left": 0, "/controls/gear/brake-right": 0,
                      "/controls/flight/elevator-trim": IC["elevator_trim"], "/controls/flight/elevator": 0,
                      "/controls/flight/aileron": 0, "/controls/flight/rudder": 0}.items():
        send(f"set {prop} {val}")
    send("run presets-commit")
    conn["s"].close()
    if len(a) >= 2:                                               # repositioning reloads the weather scenario
        time.sleep(2.0)
        connect()
        set_weather()
        time.sleep(0.3)
        conn["s"].close()
    print("FlightGear repositioned on final approach.")


if __name__ == "__main__":
    main()
