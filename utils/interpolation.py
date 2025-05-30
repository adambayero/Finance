import numpy as np
from scipy.interpolate import CubicSpline

def linear_interpolation(curve: dict, t: float) -> float:
    known_ts = sorted(curve.keys())
    if t in curve:
        return curve[t]
    for i in range(len(known_ts) - 1):
        t1, t2 = known_ts[i], known_ts[i+1]
        if t1 < t < t2:
            D1, D2 = curve[t1], curve[t2]
            return np.interp(t, [t1, t2], [D1, D2])
    raise ValueError(f"t={t} not in range")

def log_interpolation(curve: dict, t: float) -> float:
    known_ts = sorted(curve.keys())
    if t in curve:
        return curve[t]
    for i in range(len(known_ts) - 1):
        t1, t2 = known_ts[i], known_ts[i+1]
        if t1 < t < t2:
            D1, D2 = np.log(curve[t1]), np.log(curve[t2])
            return np.exp(np.interp(t, [t1, t2], [D1, D2]))
    raise ValueError(f"t={t} not in range")

def spline_interpolation(curve: dict, t: float) -> float:
    known_ts = sorted(curve.keys())

    try:
        discounts = np.array([curve[ti] for ti in known_ts])
        spline = CubicSpline(known_ts, discounts, bc_type='natural')
        return spline(t)
    except Exception as e:
        raise ValueError(f"Erreur lors de l'interpolation en t={t:.2f} : {e}")