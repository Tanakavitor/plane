#!/bin/bash
# Launch FlightGear (C172P) at the same position/conditions as the recorded demonstrations and
# connect it to the Intelligent Autopilot System.   SIMULATION ONLY.
#
#   ./launch_flightgear.sh                    # calm  (wind 150 deg / 3.7 kt, like voo-normal*)
#   ./launch_flightgear.sh 270 10             # crosswind from 270 deg at 10 kt
#   ./launch_flightgear.sh 270 12 0.3         # ... plus turbulence 0.3 (0..1)
#
# A different initial condition (defaults = the recordings), relative to runway 02:
#   DIST_M=4500 ALT_FT=1000 ./launch_flightgear.sh            # further out and higher
#   OFFSET_M=-150 HEADING=10 SPEED_KT=95 ./launch_flightgear.sh 270 10
#   DIST_M   metres before the threshold (2995)    OFFSET_M  metres right(+)/left(-) of the centreline (0)
#   ALT_FT   altitude MSL, runway is at 143 ft (801)   HEADING deg true (0)   SPEED_KT (82)
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

# --- initial state: by default the one of the CSV demonstrations (first row of voo-com-*.csv) ---
read LAT LON < <(.venv/bin/python -m ias.config "${DIST_M:-2995}" "${OFFSET_M:-0}")
echo "start: lat $LAT lon $LON  alt ${ALT_FT:-801} ft  heading ${HEADING:-0}  ${SPEED_KT:-82} kt  wind ${WIND_FROM}@${WIND_KT}"
exec "$FGFS" \
  --aircraft=c172p \
  --lat="$LAT" --lon="$LON" --altitude="${ALT_FT:-801}" --heading="${HEADING:-0}" --vc="${SPEED_KT:-82}" \
  --timeofday=noon --disable-real-weather-fetch --disable-ai-traffic \
  --wind="${WIND_FROM}@${WIND_KT}" --turbulence="${TURBULENCE}" \
  --prop:/controls/flight/elevator-trim=-0.03 \
  --prop:/sim/rendering/shaders/quality-level=1 \
  --generic=socket,out,10,127.0.0.1,5501,udp,ias_out \
  --generic=socket,in,30,127.0.0.1,5502,udp,ias_in \
  --telnet=5401 \
  "${@:4}"
