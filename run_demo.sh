#!/bin/bash
# One command: live monitor + Intelligent Autopilot + FlightGear.   SIMULATION ONLY.
#   ./run_demo.sh              calm (FlightGear "fair weather": 150 deg / 3.7 kt, as in voo-normal*.csv)
#   ./run_demo.sh 270 10       10 kt crosswind from the left (as in voo-com-8-nos.csv)
# Quit FlightGear when done; the autopilot saves its log to reports/fg_runs/ and exits by itself.
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY 05_monitor.py &            MON=$!
$PY -u 03_fly_flightgear.py &  IAS=$!
trap 'kill $MON $IAS 2>/dev/null' EXIT
sleep 1
./launch_flightgear.sh "$@" > /dev/null 2>&1
wait $IAS
