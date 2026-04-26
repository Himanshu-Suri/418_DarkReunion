# Satellite Imaging Planner - Enigma 418 - Second place

**Team:** Himanshu Suri , Kush Jalan , Rakshit Raj , Bhavesh Gunreddy

---

## What It Does

An autonomous satellite imaging scheduler that, given a TLE (Two-Line Element) orbital description and an Area of Interest (AOI), plans the optimal sequence of camera shots during a pass window minimizing attitude slew effort while maximizing ground coverage.

## How It Works

1. **Orbital Propagation** - Uses SGP4 to compute satellite position and velocity in ECI frame throughout the pass window.
2. **Closest Approach Detection** - Finds the moment the satellite is nearest to the AOI center to anchor the scheduling window.
3. **Target Grid Generation** - Samples a 5×5 grid of geo-points across the AOI (with a 5% inset to respect the sensor FOV boundary).
4. **Greedy Shot Scheduler** - At each timestep, picks the reachable target requiring the least quaternion rotation from the current attitude, subject to a max off-nadir constraint (~58.9°) and reaction wheel safety limits (≤2.5 °/s slew rate).
5. **Attitude Profile** - Generates a smooth SLERP-interpolated quaternion timeline with hold pads around each shutter event for controller settling.

## Key Parameters

| Parameter | Value | Description |
|---|---|---|
| `PARAM_GRID_LAT/LON` | 5×5 | Target sampling resolution |
| `PARAM_OFF_MAX` | 58.92° | Max off-nadir angle |
| `PARAM_MAX_SLEW_RATE_DPS` | 2.5 °/s | Reaction wheel limit |
| `PARAM_HOLD_PAD` | 0.60 s | Controller settling pad |
| `PARAM_SLERP_DT` | 0.1 s | Attitude profile resolution |

## Inputs

```python
plan_imaging(
    tle_line1,          # TLE line 1
    tle_line2,          # TLE line 2
    aoi_polygon_llh,    # List of (lat, lon) vertices
    pass_start_utc,     # ISO 8601 string
    pass_end_utc,       # ISO 8601 string
    sc_params           # {"integration_s": <float>}
)
```

## Output

```json
{
  "objective": "Min-Quaternion True-Angle Heuristic w/ Dynamic Wheel Protection",
  "attitude": [{ "t": 0.0, "q_BN": [qx, qy, qz, qw] }, "..."],
  "shutter":  [{ "t_start": 12.5, "duration": 1.0 }, "..."]
}
```

## Dependencies

```
sgp4
scipy
numpy
```

---

> *Built for Enigma 418 Hackathon*
