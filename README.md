# UU-test for Statistical Modeling of Unimodal Data

This repository provides a Python implementation of the UU-test for statistical modeling of unimodal data, based on the paper: [The UU-test for Statistical Modeling of Unimodal Data](https://arxiv.org/abs/2008.12537) by Paraskevi Chasani and Aristidis Likas.

The implementation is translated from the original MATLAB code: [pchasani/UUtest](https://github.com/pchasani/UUtest).
## Structure

### Experiments
The `experiments.ipynb` notebook contains tests for various distributions. It prints the test decision and displays the estimated CDF, PDF, ECDF, and histogram.

### Running the Demo

To run the demo, use the following command:

```bash
python demo.py --case <unimodal|bimodal>
```

Replace `<unimodal|bimodal>` with either `unimodal` or `bimodal`, depending on the case you want to test.
