| ANN | phase | input(s) | output | samples | MSE (all demos) | MSE (held-out demos) |
|---|---|---|---|---|---|---|
| glideslope_roc | final_approach | glideslope angle - desired glideslope [deg] | desired sink rate [ft/s] | 12223 | 0.0585 | 0.0730 |
| glideslope_elevator | final_approach | desired - current sink rate [ft/s] , airspeed [kt] | elevator command | 12112 | 0.0276 | 0.0148 |
| flare_roc | flare | height above runway [ft] | desired sink rate [ft/s] | 1805 | 0.0514 | 0.0540 |
| flare_elevator | flare | desired - current sink rate [ft/s] , airspeed [kt] | elevator command | 1805 | 0.0296 | 0.0222 |
| track | final_approach | centreline angle [deg] , accumulated centreline angle [deg s] | desired track relative to runway [deg] | 28008 | 0.0381 | 0.0122 |
| roll | final_approach | desired - current track [deg] | desired roll [deg] | 28008 | 0.0082 | 0.0057 |
| aileron | final_approach | desired - current roll [deg] , rudder command | aileron command | 31618 | 0.0182 | 0.0155 |
| rudder | final_approach | heading - runway heading [deg] | rudder command | 27834 | 0.0491 | 0.0405 |
| throttle | final_approach | desired sink rate [ft/s] | throttle command | 12112 | 0.0760 | 0.0423 |
| flare_throttle | flare | height above runway [ft] | throttle command | 1805 | 0.3814 | 0.3206 |
| flaps | final_approach | height above runway [ft] , airspeed [kt] | flaps command | 15809 | 0.1606 | 0.1001 |
| heading | landing | centreline angle [deg] | desired heading relative to runway [deg] | 6420 | 0.0157 | 0.0011 |
| rudder_ground | landing | desired - current heading [deg] , accumulated heading error [deg s] | rudder command | 6612 | 0.0525 | 0.0379 |
| brakes | landing | ground speed [kt] | brake command | 3659 | 0.3198 | 0.4502 |
| landing_elevator | landing | ground speed [kt] | elevator command | 3659 | 0.0361 | 0.0091 |

Held-out demonstrations: voo-normal3, voo-com-rajada, turbulencia05

Set-points: {'glideslope_ref_deg': 5.363226368886366, 'v_ref_kt': 69.88238271077475, 'flare_height_ft': 44.02051891766243, 'aim_x_m': 349.3831995418932, 'settle_s': 12.0}
