from ks import ks
import numpy as np

def backward_search(PB, eR, X):
    """Search backward for uniform sets in dataset X.

    Args:
        PB (array-like): Set of points forming uniform sets.
        eR (float): Data point causing non-uniformity.
        X (array-like): 1-D dataset.

    Returns:
        tuple: (PB, success)
            - PB: Updated set of points forming uniform sets.
            - success: True if insufficiency fixed, False otherwise.
    """
    PB = np.asarray(PB[:-1])
    X = np.asarray(X)
    while len(PB) >= 1:
        xx = X[(X >= np.max(PB)) & (X <= eR)]
        if ks(xx)[0] == 1:
            return np.append(PB, eR), True
        PB = PB[:-1]
    return PB, False

def forward_search(PF, eL, X):
    """Search forward for uniform sets in dataset X.

    Args:
        PF (array-like): Set of points forming uniform sets.
        eL (float): Data point causing non-uniformity.
        X (array-like): 1-D dataset.

    Returns:
        tuple: (PF_, success)
            - PF_: Updated set of points forming uniform sets.
            - success: True if insufficiency fixed, False otherwise.
    """
    PF = np.asarray(PF)
    X = np.asarray(X)
    idx = np.where(PF == eL)[0]
    if len(idx) == 0 or idx[0] + 1 >= len(PF):
        return np.array([]), False
    eR_ind = idx[0] + 2
    while eR_ind < len(PF):
        eR = PF[eR_ind]
        xx = X[(X >= eL) & (X <= eR)]
        if ks(xx)[0] == 1:
            return np.array([eR]), True
        eR_ind += 1
    return np.array([]), False
