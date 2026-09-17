#!/bin/bash
# Launch FlightGear (C172P) at the same position/conditions as the recorded demonstrations and
# connect it to the Intelligent Autopilot System.   SIMULATION ONLY.
#
#   ./launch_flightgear.sh                    # calm  (wind 150 deg / 3.7 kt, like voo-normal*)
#   ./launch_flightgear.sh 270 10             # crosswind from 270 deg at 10 kt
#   ./launch_flightgear.sh 270 12 0.3         # ... plus turbulence 0.3 (0..1)
#
# Start  `python 03_fly_flightgear.py`  in another terminal BEFORE running this script.
set -e
cd "$(dirname "$0")"

WIND_FROM=${1:-150}
WIND_KT=${2:-3.7}
TURBULENCE=${3:-0.0}

# --- locate FlightGear ---------------------------------------------------------------------
DMG="$HOME/Downloads/flightgear-2024.1.7-macos-universal.dmg"
find_fg() { ls -d /Applications/FlightGear*.app "$HOME"/Applications/FlightGear*.app /Volumes/FlightGear*/FlightGear.app 2>/dev/null | head -1; }
FG_APP="${FG_APP:-$(find_fg)}"
if [ -z "$FG_APP" ] && [ -f "$DMG" ]; then
  echo "FlightGear.app not found - mounting $DMG"
  hdiutil attach -nobrowse -readonly "$DMG" >/dev/null
  FG_APP="$(find_fg)"
fi
[ -n "$FG_APP" ] || { echo "FlightGear.app not found. Set FG_APP=/path/to/FlightGear.app"; exit 1; }
FGFS="$FG_APP/Contents/MacOS/FlightGear"
[ -x "$FGFS" ] || FGFS="$FG_APP/Contents/MacOS/fgfs"

# --- install the IAS <-> FlightGear protocol definitions ---------------------------------------
FG_ROOT_DIR="${FG_ROOT:-$HOME/Library/Application Support/FlightGear/fgdata_2024_1}"
mkdir -p "$FG_ROOT_DIR/Protocol"
cp fg/ias_out.xml fg/ias_in.xml "$FG_ROOT_DIR/Protocol/"

# --- same initial state as the CSV demonstrations (first row of voo-com-*.csv) ---------------
exec "$FGFS" \
  --aircraft=c172p \
  --lat=63.93762 --lon=-22.60547 --altitude=801 --heading=0 --vc=82 \
  --timeofday=noon --disable-real-weather-fetch --disable-ai-traffic \
  --wind="${WIND_FROM}@${WIND_KT}" --turbulence="${TURBULENCE}" \
  --prop:/controls/flight/elevator-trim=-0.03 \
  --prop:/sim/rendering/shaders/quality-level=1 \
  --generic=socket,out,10,127.0.0.1,5501,udp,ias_out \
  --generic=socket,in,30,127.0.0.1,5502,udp,ias_in \
  --telnet=5401 \
  "${@:4}"
