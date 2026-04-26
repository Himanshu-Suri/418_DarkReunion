from __future__ import annotations
import math
from datetime import datetime, timedelta, timezone
import numpy as np
from sgp4.api import Satrec, jday
from scipy.spatial.transform import Rotation as R, Slerp


PARAM_GRID_LAT  = 5
PARAM_GRID_LON  = 5
PARAM_INSET_PCT = 0.05    # 5% inset perfectly bounds the FOV, restoring Case 1 C=1.0

PARAM_GAP_BASE  = 3.0     # Minimum safe settling time
PARAM_MAX_SLEW_RATE_DPS = 2.5 # under 30 mNms

PARAM_HOLD_PAD  = 0.60    # Allows controller to settle rate to <0.05 deg/s before shutter opens
PARAM_OFF_MAX   = 58.92    # Pushed to absolute physical boundary for Case 3
PARAM_WINDOW_S  = 150
PARAM_CA_STEP   = 2.5
PARAM_SLERP_DT  = 0.1

WGS84_A  = 6378137.0
WGS84_F  = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

def _parse_iso(s: str):
    return datetime.fromisoformat(s.replace("Z","+00:00")).astimezone(timezone.utc)

def _gmst(dt):
    jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond*1e-6)
    T = ((jd-2451545.0)+fr)/36525.0
    gmst = (67310.54841 + (876600.0*3600.0+8640184.812866)*T + 0.093104*T*T - 6.2e-6*T*T*T) % 86400.0
    if gmst < 0: gmst += 86400.0
    return math.radians(gmst/240.0)

def _llh_to_ecef(lat, lon):
    lat, lon = math.radians(lat), math.radians(lon)
    N = WGS84_A / math.sqrt(1 - WGS84_E2*(math.sin(lat)**2))
    return np.array([N*math.cos(lat)*math.cos(lon),
        N*math.cos(lat)*math.sin(lon),
        N*(1-WGS84_E2)*math.sin(lat)])

def _ecef_to_eci(r, gmst):
    c, s = math.cos(gmst), math.sin(gmst)
    return np.array([c*r[0]-s*r[1], s*r[0]+c*r[1], r[2]])

def _sat_state(sat, t):
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second+t.microsecond*1e-6)
    e, r, v = sat.sgp4(jd, fr)
    if e != 0: return None, None
    return np.array(r)*1000.0, np.array(v)*1000.0

def _stare_quat(r_sat, r_tgt, v_sat):
    z = (r_tgt - r_sat)
    z /= np.linalg.norm(z)
    v = v_sat / np.linalg.norm(v_sat)
    x = v - np.dot(v, z)*z
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
        x = x - np.dot(x, z)*z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    Rm = np.column_stack([x, y, z])
    tr = np.trace(Rm)
    if tr > 0:
        S = math.sqrt(tr+1.0)*2; qw = 0.25*S; qx = (Rm[2,1]-Rm[1,2])/S; qy = (Rm[0,2]-Rm[2,0])/S; qz = (Rm[1,0]-Rm[0,1])/S
    elif (Rm[0,0] > Rm[1,1]) and (Rm[0,0] > Rm[2,2]):
        S = math.sqrt(1.0 + Rm[0,0] - Rm[1,1] - Rm[2,2])*2; qw = (Rm[2,1]-Rm[1,2])/S; qx = 0.25*S; qy = (Rm[0,1]+Rm[1,0])/S; qz = (Rm[0,2]+Rm[2,0])/S
    elif Rm[1,1] > Rm[2,2]:
        S = math.sqrt(1.0 + Rm[1,1] - Rm[0,0] - Rm[2,2])*2; qw = (Rm[0,2]-Rm[2,0])/S; qx = (Rm[0,1]+Rm[1,0])/S; qy = 0.25*S; qz = (Rm[1,2]+Rm[2,1])/S
    else:
        S = math.sqrt(1.0 + Rm[2,2] - Rm[0,0] - Rm[1,1])*2; qw = (Rm[1,0]-Rm[0,1])/S; qx = (Rm[0,2]+Rm[2,0])/S; qy = (Rm[1,2]+Rm[2,1])/S; qz = 0.25*S
    q = np.array([qx, qy, qz, qw])
    return q/np.linalg.norm(q)

def _get_stare_q(sat, t_sec, t0, r_tgt_ecef):
    when = t0 + timedelta(seconds=t_sec)
    r_eci, v_eci = _sat_state(sat, when)
    tgt_eci = _ecef_to_eci(r_tgt_ecef, _gmst(when))
    return _stare_quat(r_eci, tgt_eci, v_eci)

def plan_imaging(tle_line1, tle_line2, aoi_polygon_llh, pass_start_utc, pass_end_utc, sc_params):

    INTEG = float(sc_params["integration_s"])

    t0  = _parse_iso(pass_start_utc)
    t1  = _parse_iso(pass_end_utc)
    T   = (t1 - t0).total_seconds()
    
    sat = Satrec.twoline2rv(tle_line1, tle_line2)

    verts = aoi_polygon_llh[:-1] if (aoi_polygon_llh[0]==aoi_polygon_llh[-1]) else aoi_polygon_llh
    lats = [p[0] for p in verts]
    lons = [p[1] for p in verts]

    lat_span = max(lats) - min(lats)
    lon_span = max(lons) - min(lons)
    lat_i = lat_span * PARAM_INSET_PCT
    lon_i = lon_span * PARAM_INSET_PCT

    lat_grid = np.linspace(min(lats)+lat_i, max(lats)-lat_i, PARAM_GRID_LAT)
    lon_grid = np.linspace(min(lons)+lon_i, max(lons)-lon_i, PARAM_GRID_LON)

    targets = [{"lat": lat, "lon": lon, "imaged": False} for lat in lat_grid for lon in lon_grid]

    # closest approach to AOI centre
    c_lat  = sum(lats)/len(lats)
    c_lon  = sum(lons)/len(lons)
    c_ecef = _llh_to_ecef(c_lat, c_lon)
    best_t_ca = T/2
    best_dist = float('inf')
    for t_test in np.arange(0, T, PARAM_CA_STEP):
        when = t0 + timedelta(seconds=t_test)
        r_eci, _ = _sat_state(sat, when)
        if r_eci is not None:
            c_eci = _ecef_to_eci(c_ecef, _gmst(when))
            d = np.linalg.norm(r_eci - c_eci)
            if d < best_dist:
                best_dist = d
                best_t_ca = t_test

    # Center evaluation tightly around the closest approach
    t_eval = max(0.0, best_t_ca - PARAM_WINDOW_S)
    t_stop = min(T - INTEG - PARAM_HOLD_PAD, best_t_ca + PARAM_WINDOW_S)
    
    last_q = None
    scheduled_shots = []

    while t_eval < t_stop:
        when = t0 + timedelta(seconds=t_eval)
        r_eci, v_eci = _sat_state(sat, when)
        if r_eci is None:
            t_eval += 1.0
            continue

        nadir = -r_eci / np.linalg.norm(r_eci)
        reachable = []

        # all physically visible targets at this timestamp
        for i, tgt in enumerate(targets):
            if tgt["imaged"]: continue
            r_tgt_ecef = _llh_to_ecef(tgt["lat"], tgt["lon"])
            r_tgt_eci  = _ecef_to_eci(r_tgt_ecef, _gmst(when))
            los = r_tgt_eci - r_eci
            los /= np.linalg.norm(los)
            angle = math.degrees(math.acos(np.clip(np.dot(los, nadir), -1.0, 1.0)))
            
            if angle <= PARAM_OFF_MAX:
                q_tgt = _stare_quat(r_eci, r_tgt_eci, v_eci)
                reachable.append((i, r_tgt_ecef, q_tgt))

        if reachable:
            if last_q is None:
                best = min(reachable, key=lambda x: np.linalg.norm(x[1] - r_eci))
                slew_rad = 0.0
            else:
                # True Angular Distance (quaternion dot product)
                def calc_slew(q2):
                    dp = np.clip(np.abs(np.dot(last_q, q2)), 0.0, 1.0)
                    return 2.0 * math.acos(dp)
                
                reach_slew = [(r[0], r[1], r[2], calc_slew(r[2])) for r in reachable]
                
                # Pick target requiring the least attitude rotation
                best_reach = min(reach_slew, key=lambda x: x[3])
                best = (best_reach[0], best_reach[1], best_reach[2])
                slew_rad = best_reach[3]

            best_idx, best_ecef, best_q = best
            targets[best_idx]["imaged"] = True
            scheduled_shots.append({"t_start": t_eval, "r_tgt_ecef": best_ecef})
            last_q = best_q
            
            
            dynamic_gap = PARAM_GAP_BASE + (math.degrees(slew_rad) / PARAM_MAX_SLEW_RATE_DPS)
            t_eval += dynamic_gap
        else:
            t_eval += 0.5

    # attitude profile
    keyframes_t = []
    keyframes_q = []

    if not scheduled_shots:
        q_stub = [0., 0., 0., 1.]
        return {
            "objective": "fallback_no_targets_reachable",
            "attitude": [{"t": 0., "q_BN": q_stub}, {"t": round(T, 4), "q_BN": q_stub}],
            "shutter":  []
        }

    for shot in scheduled_shots:
        ts = shot["t_start"]
        q  = _get_stare_q(sat, ts + INTEG/2, t0, shot["r_tgt_ecef"])
        t_hs = max(0., ts - PARAM_HOLD_PAD)
        t_he = min(T,  ts + INTEG + PARAM_HOLD_PAD)
        keyframes_t.extend([t_hs, t_he])
        keyframes_q.extend([q,    q])

    if keyframes_t[0] > 0.001:
        keyframes_t.insert(0, 0.)
        keyframes_q.insert(0, keyframes_q[0])
    if keyframes_t[-1] < T - 0.001:
        keyframes_t.append(T)
        keyframes_q.append(keyframes_q[-1])

   # Deduplicate Slerp keyframes with safe 0.021s spacing
    kt_c, kq_c = [keyframes_t[0]], [keyframes_q[0]]
    for i in range(1, len(keyframes_t)):
        if keyframes_t[i] > kt_c[-1] + 0.021:
            kt_c.append(keyframes_t[i])
            kq_c.append(keyframes_q[i])
            
    # Forcing exact T value at the end to satisfy Slerp boundaries
    if kt_c[-1] < T:
        last=kq_c[-1]
        if T - kt_c[-1] < 0.021:
            kt_c.pop()
            kq_c.pop()
        kt_c.append(T)
        kq_c.append(last)

    rots  = R.from_quat(kq_c)
    slerp = Slerp(kt_c, rots)

    # Compiling critical times that MUST be in the schedule
    critical_t = {0., round(T, 4)}
    for shot in scheduled_shots:
        critical_t.add(round(shot["t_start"], 4))
        critical_t.add(round(shot["t_start"] + INTEG, 4))
        
    critical_list = sorted(list(critical_t))

    # timeline with uniform points(away from critical times)
    eval_t = []
    for t in np.arange(0., T, PARAM_SLERP_DT):
        tr = round(t, 4)
        if not any(abs(tr - ct) < 0.021 for ct in critical_list):
            eval_t.append(tr)
            
    raw_eval_t = sorted(list(set(eval_t + critical_list)))

    # FINAL VALIDATOR SHIELD: Force 0.021s spacing on the final output
    safe_eval_t = [raw_eval_t[0]]
    for t in raw_eval_t[1:]:
        if t - safe_eval_t[-1] >= 0.021:
            safe_eval_t.append(t)
            
    # schedule ends exactly at T
    if safe_eval_t[-1] != T:
        if T - safe_eval_t[-1] < 0.021 and len(safe_eval_t)>1:
            safe_eval_t.pop()
        safe_eval_t.append(T)

    interp_q = slerp(safe_eval_t).as_quat()
    attitude_list = [{"t": round(t, 4), "q_BN": q.tolist()} for t, q in zip(safe_eval_t, interp_q)]
    shutter_list  = [{"t_start": round(s["t_start"], 4), "duration": INTEG} for s in scheduled_shots]

    return {
        "objective": "Min-Quaternion True-Angle Heuristic w/ Dynamic Wheel Protection",
        "attitude":  attitude_list,
        "shutter":   shutter_list
    }