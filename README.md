# Research & Development / AI Assignment: Parametric Curve Fitting

This repository contains the complete implementation and documentation for the Research & Development / AI hiring assignment.

## Problem Statement

The goal is to determine the values of three unknown variables in the parametric equation of a curve:

$$x(t) = t \cdot \cos(\theta) - e^{M|t|} \cdot \sin(0.3t)\sin(\theta) + X$$
$$y(t) = 42 + t \cdot \sin(\theta) + e^{M|t|} \cdot \sin(0.3t)\cos(\theta)$$

Given boundaries for the unknown parameters:
* $0^\circ < \theta < 50^\circ$ (or $0 < \theta < 0.8727$ radians)
* $-0.05 < M < 0.05$
* $0 < X < 100$

The parameter $t$ is bounded in the range:
* $6 < t < 60$

We are provided with a dataset (`xy_data (1).csv`) containing $1500$ points $(x_i, y_i)$ that lie on this curve.

---

## Methodology & Process

Since the parameter $t$ is not explicitly provided for each point in the dataset, we formulated the curve-fitting problem as an optimization task using **L1 loss minimization** (least absolute deviations), which is more robust to noise and outliers than L2 (least squares) loss.

### 1. Coordinate Transformation
Let the parametric equations be represented as a translation and rotation from a local coordinate system $(u, v)$ where:
* $u(t) = t$
* $v(t) = e^{M|t|} \sin(0.3t)$

Then the global coordinates $(x, y)$ are given by:
$$x - X = u \cos(\theta) - v \sin(\theta)$$
$$y - 42 = u \sin(\theta) + v \cos(\theta)$$

To find the value of $t$ corresponding to each observed point $(x_i, y_i)$, we project the shifted point $(x_i - X, y_i - 42)$ onto the rotated axis:
$$t_i = (x_i - X)\cos(\theta) + (y_i - 42)\sin(\theta)$$

The observed perpendicular deviation in the rotated frame is:
$$v_{\text{observed}, i} = -(x_i - X)\sin(\theta) + (y_i - 42)\cos(\theta)$$

### 2. Loss Function Definition
The L1 objective function minimizes the sum of absolute errors between the observed perpendicular deviation $v_{\text{observed}, i}$ and the theoretical model $v(t_i)$:
$$\text{Loss}(\theta, M, X) = \sum_{i=1}^{N} \left| v_{\text{observed}, i} - e^{M|t_i|} \sin(0.3t_i) \right|$$

### 3. Optimization Algorithm
We implemented this optimization using SciPy's Nelder-Mead simplex algorithm, bounded to the required parameters' ranges. The algorithm converges rapidly to a global minimum, indicating a perfect fit.

---

## Final Results

### 1. Estimated Parameters
The optimized parameters obtained from the data are:

| Parameter | Optimized Value (Raw) | Rounded Value | Unit / Scale |
| :--- | :--- | :--- | :--- |
| **$\theta$ (theta)** | $0.52359830$ | **$0.5236$** | Radians ($30.0^\circ$) |
| **$M$** | $0.03000000$ | **$0.0300$** | Exponential factor |
| **$X$** | $54.99999835$ | **$55.0000$** | Horizontal offset |

**L1 Optimization Metrics:**
* Total L1 Loss: **$0.00383789$**
* Average L1 Loss per point: **$2.56 \times 10^{-6}$**

### 2. LaTeX Submission Format
The final parametric equation formatted for Desmos is:

```latex
\left(t*\cos(0.5236)-e^{0.0300\left|t\right|}\cdot\sin(0.3t)\sin(0.5236)+55.0000,42+t*\sin(0.5236)+e^{0.0300\left|t\right|}\cdot\sin(0.3t)\cos(0.5236)\right)
```

Desmos interactive check link: [Desmos Graphing Calculator](https://www.desmos.com/calculator/rfj91yrxob)

---

## Code Structure and Running

The code is implemented in Python and uses standard libraries (`numpy`, `pandas`, `scipy`, `matplotlib`).

### Prerequisites
Install the required packages:
```bash
pip install numpy pandas scipy matplotlib
```

### Running the Fitting Code
Run the following command to execute the optimization and generate the plot:
```bash
python fit_curve_new.py
```

Upon completion, it prints the optimized parameters, the L1 loss, the LaTeX submission string, and saves the plot as `fit_plot_new.png`.

---

## Submission Details

Please find the details required for the submission form below:

1. **Name** - [Candidate Name]
2. **Course with Specialization** - [Course / Major]
3. **Contact number** - [Phone Number]
4. **Email Id** - [Email]
5. **College Name** - [University Name]
6. **Assignment** - Research & Development*
7. **Github Profile Link** - [GitHub Profile Link]
8. **Github Repo Link** - [GitHub Repository Link]
9. **Portfolio Link (If Any)** - [Portfolio Link]
10. **Desmos Link** - https://www.desmos.com/calculator/rfj91yrxob
11. **LinkedIn Profile Link** - [LinkedIn Profile Link]
