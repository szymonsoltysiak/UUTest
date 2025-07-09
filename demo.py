import numpy as np
import matplotlib.pyplot as plt
from uu_main import UUtest, fitUU_1d
from uu_distr import cdfUU, pdfUU
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UUTest demo script')
    parser.add_argument('--case', type=str, default='unimodal', 
                        choices=['unimodal', 'multimodal'],
                        help='Data distribution type (unimodal or multimodal)')
    args = parser.parse_args()
    
    case = args.case
    print(f"Running demo for case: {case}")
    N = 1000

    if case == "unimodal":
        X = np.random.normal(0, 1, N)
    elif case == "multimodal":
        X1 = np.random.normal(0, 1, N)
        X2 = np.random.normal(7, 1, N)
        X = np.concatenate([X1, X2])
    else:
        raise ValueError("Unknown case. Use 'unimodal' or 'multimodal'.")

    S = UUtest(X)
    unimodal = len(S) > 0
    if unimodal:
        print("The dataset is unimodal.")    
        S, p = fitUU_1d(X)
        y_cdf, x = cdfUU(X, S, p)
        y_pdf, x = pdfUU(X, S, p)
    else:
        print("The dataset is multimodal.")

    plt.figure()
    plt.hist(X, bins=N//20, cumulative=True, density=True, histtype='step', label='ECDF')
    if unimodal:
        plt.plot(x, y_cdf, 'k--', linewidth=2, label='CDF UU')
    plt.legend(loc='upper left', frameon=False)
    plt.show()

    plt.figure()
    plt.hist(X, bins=N//20, density=True, label='Histogram')
    if unimodal:
        plt.plot(x, y_pdf, 'k--', linewidth=2, label='PDF UU')
    plt.legend(loc='upper left', frameon=False)
    plt.show()