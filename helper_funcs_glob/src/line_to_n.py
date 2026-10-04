import numpy as np


def line_to_n(line_path: str,
              reftrack: np.ndarray,
              normvectors: np.ndarray,
              w_veh: float) -> np.ndarray:
    """
    Documentation:
    Lateral offset n of a closed line along the normal vectors of the reference line, for use as initial guess of the
    minimum time optimization. The line is intersected with the normal through every reference point; where a normal
    crosses the line several times, the crossing closest to the reference point is used. n is clipped to the corridor
    the optimizer allows (track width minus half the vehicle width on each side).

    Inputs:
    line_path:      CSV of the closed line, columns x_m, y_m first (',' or ';' separated, '#' comments)
    reftrack:       reference track [x_m, y_m, w_tr_right_m, w_tr_left_m]
    normvectors:    normalized normal vectors of the reference line (pointing to the right, as in TPH)
    w_veh:          vehicle width used in the optimization (width_opt)

    Outputs:
    n:              lateral offset in m for every reference point, positive to the left (n = -alpha)
    """

    with open(line_path) as fh:
        sep = ";" if ";" in next(l for l in fh if not l.startswith("#")) else ","
    line = np.loadtxt(line_path, comments="#", delimiter=sep, ndmin=2)[:, :2]
    if np.hypot(*(line[0] - line[-1])) > 1e-6:
        line = np.vstack((line, line[0]))

    a = line[:-1]
    b = line[1:] - line[:-1]
    n = np.zeros(reftrack.shape[0])

    for i in range(reftrack.shape[0]):
        p = reftrack[i, :2]
        d = -normvectors[i]                                 # left-pointing normal -> n positive to the left

        # solve p + t * d = a + u * b for every segment
        den = d[0] * b[:, 1] - d[1] * b[:, 0]
        ok = np.abs(den) > 1e-12
        diff = a - p
        t = np.full(den.shape, np.inf)
        u = np.full(den.shape, -1.0)
        t[ok] = (diff[ok, 0] * b[ok, 1] - diff[ok, 1] * b[ok, 0]) / den[ok]
        u[ok] = (diff[ok, 0] * d[1] - diff[ok, 1] * d[0]) / den[ok]
        hit = ok & (u >= 0.0) & (u <= 1.0)

        if np.any(hit):
            n[i] = t[hit][np.argmin(np.abs(t[hit]))]
        else:
            # no crossing (should not happen for a closed line inside the track): signed distance to nearest point
            j = np.argmin(np.hypot(*(line - p).T))
            n[i] = np.dot(line[j] - p, d)

    return np.clip(n, -reftrack[:, 2] + w_veh / 2, reftrack[:, 3] - w_veh / 2)
