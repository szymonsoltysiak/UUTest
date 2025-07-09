import numpy as np
from scipy import stats

def ks(x):
    """Perform Kolmogorov-Smirnov test to check if dataset is uniform.

    Args:
        x (array-like): 1-D dataset.

    Returns:
        tuple: (h, p, stat, cv)
            - h: 1 if p > alpha (consistent with uniform), 0 otherwise
            - p: p-value of the hypothesis test
            - stat: test statistic value
            - cv: approximate critical value of the test
    """
    x = np.asarray(x)
    if len(x) <= 1:
        return 1, None, None, None
    
    dist = stats.uniform(loc=np.min(x), scale=np.max(x) - np.min(x))
    stat, p = stats.kstest(x, dist.cdf)
    alpha = 0.01
    h = 1 if p > alpha else 0
    
    n = len(x)
    cv = np.sqrt(-0.5 * np.log(alpha / 2)) / np.sqrt(n)
    
    return h, p, stat, cv
