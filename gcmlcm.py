from scipy.spatial import ConvexHull
import numpy as np

def compute_gcmlcm(X):
    """Compute greatest convex minorant and least concave majorant of dataset X.

    Args:
        X (array-like): 1-D dataset.

    Returns:
        tuple: (gcm, lcm, gcmf, lcmf, F)
            - gcm: Greatest convex minorant points.
            - lcm: Least concave majorant points.
            - gcmf: CDF values at gcm points.
            - lcmf: CDF values at lcm points.
            - F: Empirical CDF values.
    """
    X = np.sort(X)
    F, x = np.histogram(X, bins=len(X), density=True)
    F = np.cumsum(F) / np.sum(F)
    x = x[1:]
    if len(x) < 3:
        return np.array([x[0]]), np.array([x[1]]), np.array([F[0]]), np.array([F[1]]), F
    points = np.vstack((x, F)).T
    hull = points[ConvexHull(points).vertices]
    hull = hull[np.argsort(hull[:, 0])]
    gcm, lcm = [], []
    gcmf, lcmf = [], []
    split_idx = np.argmax(hull[:, 0])
    gcm = hull[:split_idx + 1, 0]
    lcm = hull[split_idx:, 0]
    gcmf = hull[:split_idx + 1, 1]
    lcmf = hull[split_idx:, 1]
    return gcm, lcm, gcmf, lcmf, F
