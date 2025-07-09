from set_conditions import sufficient, consistent
from gcmlcm import compute_gcmlcm
from ks import ks
import numpy as np

def UU(SG, PI, SL, X):
    """Recursively compute convex, intermediate, and concave parts of the CDF.

    Args:
        SG (array-like): Convex part.
        PI (array-like): Intermediate part [start, end].
        SL (array-like): Concave part.
        X (array-like): 1-D dataset.

    Returns:
        tuple: (SG_, PI_, SL_, success)
            - SG_: Updated convex part.
            - PI_: Updated intermediate part.
            - SL_: Updated concave part.
            - success: True if unimodal, False otherwise.
    """
    SG_ = np.asarray(SG)
    SL_ = np.asarray(SL)
    X = np.asarray(X)
    xx = X[(X >= PI[0]) & (X <= PI[1])]
    if ks(xx)[0] == 1:
        return SG_, PI, SL_, True
    gcm, lcm, _, _, _ = compute_gcmlcm(xx)
    C, ind, sz = consistent(gcm, lcm)
    for i in range(sz):
        pos = np.where(ind[i] == 1)[0]
        if len(pos) > 0 and pos[0] != 0:
            c = pos[0] - 1
            PG = C[i][:c + 1]
            PI_ = np.array([C[i][c], C[i][c + 1]])
            PL = C[i][c + 1:]
            PG_, success = sufficient(PG, X)
            if not success:
                continue
            PL_, success = sufficient(PL, X)
            if not success:
                continue
            SG_ = np.concatenate([SG_, PG_])
            SL_ = np.concatenate([SL_, PL_])
        elif len(pos) > 0 and pos[0] == 0:
            PI_ = np.array([])
            PL = C[i]
            PL_, success = sufficient(PL, X)
            if not success:
                continue
            SL_ = np.concatenate([SL_, PL_])
        else:
            PI_ = np.array([])
            PG = C[i]
            PG_, success = sufficient(PG, X)
            if not success:
                continue
            SG_ = np.concatenate([SG_, PG_])
        if success and len(PI_) > 0:
            SG_, PI_, SL_, success = UU(SG_, PI_, SL_, X)
        if success:
            return SG_, PI_, SL_, True
    return np.array([]), np.array([]), np.array([]), False

def UUtest(X):
    """Perform UU-test to determine if dataset is unimodal.

    Args:
        X (array-like): 1-D dataset.

    Returns:
        array: Subset S where CDF is unimodal and sufficient, empty if multimodal.
    """
    X = np.sort(X)
    SG, SL = np.array([]), np.array([])
    PI = np.array([np.min(X), np.max(X)])
    SG_, PI_, SL_, success = UU(SG, PI, SL, X)
    if not success:
        return np.array([])
    return np.unique(np.concatenate([SG_, PI_, SL_]))

def fitUU_1d(X):
    """Fit UU model to 1-D data.

    Args:
        X (array-like): 1-D dataset.

    Returns:
        tuple: (S, p)
            - S: Subset of X where CDF is unimodal and sufficient.
            - p: Percentages of data points in each interval.
    """
    X = np.sort(X)
    N = len(X)
    S = UUtest(X)
    if len(S) == 0:
        return S, np.array([])
    S = np.vstack((S[:-1], S[1:])).T
    if len(S) == 1:
        return S, np.array([1])
    p = np.zeros(len(S))
    for i in range(len(S)):
        if i < len(S) - 1:
            p[i] = np.sum((X >= S[i, 0]) & (X < S[i, 1])) / N
        else:
            p[i] = np.sum((X >= S[i, 0]) & (X < S[i, 1])) / N

    return S, p
