import numpy as np

def cdfUU(X, S, p):
    """Compute CDF values of the Unimodal Uniform Model (UUM).

    Args:
        X (array-like): 1-D dataset.
        S (array-like): Subset of X (intervals as [start, end]).
        p (array-like): Percentages of data points in each interval.

    Returns:
        tuple: (y, X)
            - y: CDF values.
            - X: Sorted dataset.
    """
    X = np.sort(X)
    y = np.zeros(len(X))
    for i, xi in enumerate(X):
        if xi < S[0, 0]:
            y[i] = 0
        elif xi >= S[0, 0] and xi <= S[-1, 1]:
            for j in range(len(S)):
                if xi >= S[j, 0] and xi <= S[j, 1]:
                    y[i] = np.sum(p[:j]) + p[j] * (xi - S[j, 0]) / (S[j, 1] - S[j, 0])
                    break
        else:
            y[i] = 1
    return y, X

def pdfUU(X, S, p):
    """Compute PDF values of the Unimodal Uniform Model (UUM).

    Args:
        X (array-like): 1-D dataset.
        S (array-like): Subset of X (intervals as [start, end]).
        p (array-like): Percentages of data points in each interval.

    Returns:
        tuple: (y, X)
            - y: PDF values.
            - X: Sorted dataset.
    """
    X = np.sort(X)
    y = np.zeros(len(X))
    for i, xi in enumerate(X):
        for j in range(len(S)):
            if xi >= S[j, 0] and (xi < S[j, 1] or (j == len(S) - 1 and xi == S[j, 1])):
                y[i] = p[j] / (S[j, 1] - S[j, 0])
                break
    return y, X
