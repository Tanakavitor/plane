# Neural Landing — an imitation-learning autopilot for the final approach (FlightGear, C172P)

> **Simulation only.** This project controls a simulated Cessna 172P inside FlightGear. It is not
> designed, tested or suitable for any real aircraft.

Artificial neural networks learn to fly the final approach, flare, touchdown and roll-out at
Keflavík (BIKF) runway 02 by imitating a human pilot's recorded flights, then fly the aircraft
autonomously in FlightGear. The method follows

> H. Baomar & P. J. Bentley, *Autonomous flight cycles and extreme landings of airliners beyond the
> current limits and capabilities using artificial neural networks*, Applied Intelligence 51 (2021).

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # once

.venv/bin/python 01_prepare_data.py      # CSVs  -> 16 segmented demonstrations (data/processed/)
.venv/bin/python 02_train.py             # train the ANNs -> models/ias_models.json, reports/02_*
./run_demo.sh                            # monitor + autopilot + FlightGear (calm)
./run_demo.sh 270 10                     # same, 10 kt crosswind from the left
```

Or step by step, in three terminals: `python 05_monitor.py`, `python 03_fly_flightgear.py`, then
`./launch_flightgear.sh [wind_from wind_kt [turbulence]]`. To fly again without restarting
FlightGear: `python reset_flightgear.py [wind_from wind_kt [gust_kt]]` and start `03_fly_flightgear.py` again.
`python 04_evaluate.py [--bench]` compares every logged autonomous landing with the human ones.

`launch_flightgear.sh` starts the C172P in the air exactly where the recordings start (63.93762 N,
22.60547 W, 801 ft, heading 0°, 82 kt ≈ 3 km before the threshold), copies `fg/ias_*.xml` into
`$FG_ROOT/Protocol/`, and opens two UDP generic-protocol sockets plus a telnet port. It finds
FlightGear in /Applications or on the mounted DMG (mounting `~/Downloads/flightgear-2024.1.7-…dmg`
if needed); override with `FG_APP=/path/FlightGear.app`. A connected joystick will fight the
autopilot for the same `/controls` properties — unplug it or leave it centred.

### Other initial conditions
Defaults reproduce the recordings; override them relative to runway 02 with environment variables
(`launch_flightgear.sh`, `run_demo.sh`) or flags (`reset_flightgear.py --dist --offset --alt --heading --speed`):

```bash
DIST_M=4500 ALT_FT=1000 ./run_demo.sh                       # further out, higher
OFFSET_M=-150 HEADING=10 SPEED_KT=95 ./run_demo.sh 270 10   # left of the centreline, fast, crosswind
```

`python -m sim.ic_sweep` tries a range of starts in the headless bench. There the IAS still lands on the
runway from 1.2–6 km out, up to ~200 m beside the centreline, 30° off heading, 65–105 kt and 600–1300 ft;
it fails when started 400 m to the side (touches down beside the runway) or too low to reach it (450 ft at 3 km).
Outside the demonstrated range the ANN inputs are clipped, so behaviour degrades gracefully rather than wildly,
but only the recorded start has been flown in FlightGear itself.

### Seeing what the model does
* `05_monitor.py` — live window: yoke / rudder / throttle / flaps / brakes as sent, "what each ANN
  wants vs. what the aircraft does", and the flight path over the human demonstrations.
  `--replay reports/fg_runs/<run>.csv` replays a saved flight.
* In FlightGear: the cockpit yoke, pedals, throttle and flap lever move; `h` shows the HUD, `v`
  cycles to outside views where the control surfaces move; *Debug → Browse Internal Properties →
  /controls/flight* shows the raw commands.
* Every run is logged at 10 Hz (observations, features, ANN set-points, commands) in `reports/fg_runs/`.

## The data (step A of the paper)

`voos-piloto/*.csv` are 60 Hz FlightGear recordings. `ias/data.py` never trusts file boundaries:
it finds every **touchdown preceded by ≥ 20 s of flight**, identifies the runway from the touchdown
position/heading, then walks back in time while the aircraft is "on final" (track within 35° of the
runway, < 400 m from the extended centreline, < 3.3 km out). The demonstration ends when the aircraft
stops or the throttle is opened for a touch-and-go.

* 14 files → one demonstration each.
* `circuito-completo.csv` → **two**: the first landing on RWY 02 (t = 8–115 s) and, after the
  take-off and circuit, a second landing on the crossing **RWY 29** (t = 309–437 s). All features are
  runway-relative, so the RWY 29 landing is usable training data.
* `voo-inicio-teste.csv` → the 138 s of circling before the real approach are discarded.

Each demonstration is averaged down to 10 Hz — the paper's 0.1 s interface rate.

## The model (step B)

As in the paper, there is no big network. Each task has its own **single-hidden-layer tanh ANN**
(5–10 neurons, NumPy, `ias/ann.py`), trained by back-propagation with learning rate 0.1 and
momentum 0.9 (paper Eq. 1–3), and the ANNs are chained in the paper's two-stage cascade:

```
outer ANN:  error                       -> desired response      (paper: "rate of change" ANN)
inner ANN:  desired − current response  -> control command       (paper: command ANN)
```

Labels come from the demonstrations by hindsight: the outer target is the response the pilot
actually produced a moment later; the inner pair is (the change of response that followed → the
command the pilot was holding). `reports/02_ann_fits.png` shows every learned curve.

| Phase | Chain |
|---|---|
| final approach | glideslope angle error → **glideslope_roc** → desired sink rate → **glideslope_elevator** (+ airspeed) → elevator |
| | desired sink rate → **throttle** |
| | height, airspeed → **flaps** |
| | centreline angle (+ its accumulated value) → **track** → desired track → **roll** → desired roll → **aileron** (+ rudder) → aileron |
| | heading − runway heading → **rudder** |
| flare (< 44 ft) | height → **flare_roc** → desired sink rate → **flare_elevator** (+ airspeed) → elevator; height → **flare_throttle** |
| landing (wheels down) | centreline angle → **heading** → desired heading → **rudder_ground** (+ accumulated heading error) → rudder; ground speed → **brakes**, **landing_elevator** |

Set-points are also read from the demonstrations instead of being typed in: demonstrated glidepath
5.4°, aim point 349 m past the threshold, flare height 44 ft.

### Where this deliberately differs from the paper (and why)
* **Throttle follows the glidepath, not the airspeed.** The paper's airline pilot holds speed with
  thrust. In these C172 demonstrations throttle correlates +0.70 with sink rate and 0.00 with airspeed
  error: the pilots fly "power for glidepath". The Throttle ANN is therefore fed by the glideslope ANN.
* **Elevator instead of elevator trim**, with airspeed as a second input, because the elevator needed
  to hold a flight path changes as the speed decays in the flare.
* **Memory inputs.** A memoryless error→command chain is proportional-only and leaves a steady offset
  under any persistent disturbance (propeller effects, crosswind). Human pilots integrate that away.
  Two ANNs get the *accumulated* error as a second input (centreline angle for `track`, heading error for
  `rudder_ground`); the demonstrations support it (R² 0.22 → 0.28) and the response to it is learned,
  not tuned.
* **Left/right mirroring.** Every recorded crosswind blows from the left. Lateral ANNs are trained on
  the demonstrations plus their mirror image and are bias-free, hence exactly odd: zero error → zero command.
* **Flare phase** between final approach and landing (the paper refers to earlier work for it).
* Not learned — flight-manager rules, as in the paper's Fig. 5: phase switching (flare height,
  weight on wheels ≥ 0.5 s), throttle closed after touchdown, flaps never retracted, trim wheel untouched.

## Autonomous control (step C) and results

`ias/controller.py` (flight manager + cascades) is simulator-agnostic; `ias/fg_interface.py` is the
paper's "interface" (UDP, 10 Hz). `sim/jsb_bench.py` runs FlightGear's own C172P JSBSim flight model
headless for fast closed-loop testing (`python -m sim.sweep`).

FlightGear 2024.1.7, fully autonomous from the recorded initial condition to a full stop
(`reports/04_touchdowns.md`, `reports/04_ias_vs_pilot.png`):

| | touchdown past threshold | from centreline | sink rate | airspeed | roll-out |
|---|---|---|---|---|---|
| 16 human demonstrations | 318 – 840 m | ≤ 12.7 m | +15 … −297 fpm | 46 – 67 kt | on runway |
| IAS, calm (150°/4 kt), 4 runs¹ | 679 – 682 m | 11.5 m | −7 … −9 fpm | 57 kt | on runway, ends on centreline |
| IAS, crosswind 270°/10 kt (final models) | 558 m | 8.8 m | −6 fpm | 56 kt | on runway |

¹ flown with the model version before the last lateral refinements (aileron cross-control input, ground
rudder memory input); the final models were verified in FlightGear in the crosswind case and in the bench
for calm (1.5 m from the centreline).

### Known limits
* The headless bench is pessimistic about crosswind roll-outs (it leaves the runway there, FlightGear
  did not) — its ground/initialisation model is simplified. Treat FlightGear as the reference.
* Strong turbulence (≥ 0.5) gives firm touchdowns (~ −500 fpm in the bench); 16 demonstrations contain
  only two turbulent flights.
* Valid only for the demonstrated situation: C172P, this initial condition, BIKF RWY 02 (RWY 29 geometry
  is defined too: `--runway 29`, untested). There is no go-around logic.

## Layout
```
01_prepare_data.py 02_train.py 03_fly_flightgear.py 04_evaluate.py 05_monitor.py
launch_flightgear.sh  reset_flightgear.py  run_demo.sh
ias/   config.py features.py data.py ann.py networks.py controller.py fg_interface.py
fg/    ias_out.xml ias_in.xml            FlightGear generic-protocol definitions
sim/   jsb_bench.py run_bench.py sweep.py replay_check.py fdm/   headless JSBSim test bench
models/ias_models.json                   all weights, scalers and set-points (human-readable)
reports/                                 training table, ANN fits, comparisons, FlightGear run logs
```
