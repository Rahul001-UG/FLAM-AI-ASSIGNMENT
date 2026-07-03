# Robust Parametric Curve Fitting via L1 Optimization

This project implements a robust curve-fitting pipeline to find three unknown parameters ($\theta$, $M$, $X$) in a parametric system. Instead of standard least-squares (L2), this model uses L1 loss minimization (least absolute deviations) to fit the parameters, which makes it highly robust to noise and outliers.

## The Problem
We have a dataset (`xy_data (1).csv`) with 1,500 $(x, y)$ coordinate points. These points lie on a curve defined by:
$$x(t) = t \cos(\theta) - e^{M|t|} \sin(0.3t)\sin(\theta) + X$$
$$y(t) = 42 + t \sin(\theta) + e^{M|t|} \sin(0.3t)\cos(\theta)$$

We need to recover the parameters within these bounds:
* $0^\circ < \theta < 50^\circ$ (or $0 < \theta < 0.8727$ rad)
* $-0.05 < M < 0.05$
* $0 < X < 100$
* $6 < t < 60$

Since we do not have the parameter $t$ for each point, standard regression doesn't work directly. We need a coordinate projection trick to estimate $t$ dynamically.

---

## Detailed Step-by-Step Guide

Here is exactly how the mathematical model and optimization are built:

### Step 1: Rotate and Translate
We simplify the parametric equations by treating the curve as a rotated and translated version of a simpler curve in a local coordinate system $(u, v)$, where:
* $u(t) = t$
* $v(t) = e^{M|t|} \sin(0.3t)$

The global coordinates $(x, y)$ are obtained by translating by $(X, 42)$ and rotating by $\theta$:
$$x - X = u \cos(\theta) - v \sin(\theta)$$
$$y - 42 = u \sin(\theta) + v \cos(\theta)$$

### Step 2: Project to Find $t$
For any data point $(x_i, y_i)$, we project the shifted point $(x_i - X, y_i - 42)$ onto the rotated axis to estimate the parameter $t_i$:
$$t_i = (x_i - X)\cos(\theta) + (y_i - 42)\sin(\theta)$$

This projection assumes that the perpendicular deviation is small, which holds true for a good fit.

### Step 3: Compute Perpendicular Deviations
The observed perpendicular component in the rotated frame is:
$$v_{\text{observed}, i} = -(x_i - X)\sin(\theta) + (y_i - 42)\cos(\theta)$$

We want this to match the theoretical model value $v(t_i) = e^{M|t_i|} \sin(0.3t_i)$.

### Step 4: Formulate the L1 Objective
The objective is to minimize the sum of absolute differences between the observed and theoretical perpendicular components:
$$\text{Loss}(\theta, M, X) = \sum_{i=1}^{N} \left| v_{\text{observed}, i} - e^{M|t_i|} \sin(0.3t_i) \right|$$

Using L1 loss (absolute error) makes the optimization much more robust to outliers compared to standard L2 (squared error) loss.

### Step 5: Solve via Nelder-Mead
The Python script uses SciPy's Nelder-Mead simplex algorithm to find the optimal parameters. 
* **Initial Guess:** $\theta = 0.4$ rad, $M = 0.0$, $X = 50.0$.
* **Bounds:** Handled by Nelder-Mead with parameter constraints to ensure $\theta$, $M$, and $X$ stay within physical bounds.
* **Tolerances:** Set to $10^{-8}$ for both parameter change (`xatol`) and function value change (`fatol`) to guarantee precise convergence.

---

## Code Walkthrough & How to Run

### Setup
Make sure you have Python 3 installed. Install the dependencies:
```bash
pip install numpy pandas scipy matplotlib
```

### Running the Script
Run the main script (`FLAM.py`) to fit the curve and generate the plot:
```bash
python FLAM.py
```

### What happens when you run it:
1. **Loads the Data:** Reads coordinates from `xy_data (1).csv`.
2. **Performs Optimization:** Runs the Nelder-Mead solver using the L1 loss function described above.
3. **Prints Output:** Displays the estimated values for $\theta$ (in both degrees and radians), $M$, $X$, and the final L1 loss metric.
4. **Saves Visualization:** Exports a plot named `fit_plot_new.png` showing the raw data points alongside the fitted parametric curve.

---

## Results

### Estimated Parameters

The solver successfully converged to the following values:

| Parameter | Optimized Value | Rounded (4 Decimals) | Notes |
| :--- | :--- | :--- | :--- |
| **$\theta$ (theta)** | $0.52359830$ | **$0.5236$** | Equivalent to $30.0^\circ$ |
| **$M$** | $0.03000000$ | **$0.0300$** | Exponential decay factor |
| **$X$** | $54.99999835$ | **$55.0000$** | Horizontal offset |

### Metrics
* **Total L1 Loss:** `0.00383789`
* **Average L1 Loss per point:** `2.56e-6`

The extremely low loss confirms that the fit is a near-perfect reconstruction of the underlying curve.

---

## References

* **Nelder-Mead Algorithm:** Nelder, J. A., & Mead, R. (1965). "A simplex method for function minimization." *The Computer Journal*, 7(4), 308–313.
* **L1 Loss & Robust Estimation:** Koenker, R., & Bassett, G. (1978). "Regression Quantiles." *Econometrica*, 46(1), 33-50.
* **SciPy Optimization:** Virtanen, P., et al. (2020). "SciPy 1.0: Fundamental algorithms for scientific computing in Python." *Nature Methods*, 17, 261–272.
