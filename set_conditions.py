from search import forward_search, backward_search
from ks import ks
import numpy as np

def sufficient(P, X):
    """Determine sufficient subsets of convex and concave sets.

    Args:
        P (array-like): Convex or concave sets.
        X (array-like): 1-D dataset.

    Returns:
        tuple: (P_, success)
            - P_: Sufficient set of points.
            - success: True if P_ is sufficient, False otherwise.
    """
    P = np.asarray(P)
    X = np.asarray(X)
    xx = X[(X >= P[0]) & (X <= P[-1])]
    if ks(xx)[0] == 1:
        return np.array([P[0], P[-1]]), True
    P_ = np.array([P[0]])
    while np.max(P_) != np.max(P):
        eR = P[P > np.max(P_)][0]
        xx = X[(X >= np.max(P_)) & (X <= eR)]
        if ks(xx)[0] == 1:
            P_ = np.append(P_, eR)
        else:
            PF, success = forward_search(P, np.max(P_), X)
            if success:
                P_ = np.append(P_, PF)
            else:
                PB, success = backward_search(P_, eR, X)
                if not success:
                    return np.array([]), False
                P_ = PB
    return P_, True

def compute_consistent_sets(gcm, lcm):
    """Compute consistent subsets by adjusting gcm and lcm points.

    Args:
        gcm (array-like): Greatest convex minorant points.
        lcm (array-like): Least concave majorant points.

    Returns:
        tuple: (C, ind, sz)
            - C: List of consistent sets.
            - ind: List of indicators (0 for gcm, 1 for lcm).
            - sz: Number of consistent sets.
    """
    allsort = np.sort(np.concatenate([gcm, lcm]))
    gl_ind = np.isin(allsort, lcm).astype(int)
    pos = np.where((gl_ind[:-1] == 0) & (gl_ind[1:] == 1))[0]
    sz = len(pos)
    C = []
    ind = []
    for i in pos:
        c = []
        idx = []
        for j in range(i + 1):
            if gl_ind[j] == 0:
                c.append(allsort[j])
                idx.append(0)
        for j in range(i + 1, len(allsort)):
            if gl_ind[j] == 1:
                c.append(allsort[j])
                idx.append(1)
        C.append(np.array(c))
        ind.append(np.array(idx))
    return C, ind, sz

def consistent(gcm, lcm):
    """Check if gcm and lcm sets are consistent and compute consistent subsets.

    Args:
        gcm (array-like): Greatest convex minorant points.
        lcm (array-like): Least concave majorant points.

    Returns:
        tuple: (C, ind, sz)
            - C: List of consistent sets.
            - ind: List of indicators (0 for gcm, 1 for lcm).
            - sz: Number of consistent sets.
    """
    gcm = np.asarray(gcm)
    lcm = np.asarray(lcm)
    if len(gcm) == 0:
        return [lcm], [np.ones(len(lcm))], 1
    elif len(lcm) == 0:
        return [gcm], [np.zeros(len(gcm))], 1
    elif np.max(gcm) < np.min(lcm):
        return [np.concatenate([gcm, lcm])], [np.concatenate([np.zeros(len(gcm)), np.ones(len(lcm))])], 1
    else:
        return compute_consistent_sets(gcm, lcm)
