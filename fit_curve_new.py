"""
Parametric Curve Fitting from Scratch
Assignment: Research & Development / AI

Problem Description:
Find the values of unknown variables in the given parametric equation of a curve:
    x = t * cos(theta) - e^(M*|t|) * sin(0.3*t) * sin(theta) + X
    y = 42 + t * sin(theta) + e^(M*|t|) * sin(0.3*t) * cos(theta)

Where:
    - 0 deg < theta < 50 deg
    - -0.05 < M < 0.05
    - 0 < X < 100
    - 6 < t < 60
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def load_data(file_path):
    """Loads the x, y coordinates from a CSV file."""
    print(f"Reading data from {file_path}...")
    df = pd.read_csv(file_path)
    return df['x'].values, df['y'].values

def l1_loss(params, x_data, y_data):
    """
    Computes the L1 distance between the observed data points and the curve.
    
    Given a candidate set of parameters (theta, M, X):
    We rotate the data points back by theta and shift by X, 42.
    In the local coordinate system of the curve:
        u(t) = t
        v(t) = e^(M*|t|) * sin(0.3*t)
        
    By inverting the rotation and shift:
        t = (x - X) * cos(theta) + (y - 42) * sin(theta)
        v_observed = -(x - X) * sin(theta) + (y - 42) * cos(theta)
        
    We then minimize the L1 loss:
        Loss = Sum |v_observed - e^(M*|t|) * sin(0.3*t)|
    """
    theta, M, X = params
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    dx = x_data - X
    dy = y_data - 42.0
    
    # Project data to get the parameter t for each point
    t_est = dx * cos_t + dy * sin_t
    
    # Calculate the observed vertical displacement in the rotated frame
    v_obs = -dx * sin_t + dy * cos_t
    
    # Evaluate the theoretical model
    v_model = np.exp(M * np.abs(t_est)) * np.sin(0.3 * t_est)
    
    # Compute L1 distance
    loss = np.sum(np.abs(v_obs - v_model))
    return loss

def fit_curve(x_data, y_data):
    """Fits the parametric curve to the data using L1 loss optimization."""
    # Bounds for the unknown variables:
    # 0 deg < theta < 50 deg  -> in radians: 0 < theta < 50 * pi / 180
    # -0.05 < M < 0.05
    # 0 < X < 100
    bounds = [
        (0.0, np.radians(50.0)),
        (-0.05, 0.05),
        (0.0, 100.0)
    ]
    
    # Initial guess
    initial_guess = [np.radians(25.0), 0.0, 50.0]
    
    print("Running optimization using Nelder-Mead method...")
    result = minimize(
        l1_loss,
        initial_guess,
        args=(x_data, y_data),
        bounds=bounds,
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}
    )
    
    if not result.success:
        print("Warning: Optimization did not converge successfully.")
        print(result.message)
        
    return result.x, result.fun

def generate_latex_format(theta, M, X):
    """Formats the equations in the requested Desmos LaTeX format."""
    # Convert theta to radians (rounded)
    theta_rad = round(theta, 4)
    M_val = round(M, 4)
    X_val = round(X, 4)
    
    latex_str = (
        f"\"\\left(t*\\cos({theta_rad:.4f})-e^{{{M_val:.4f}\\left|t\\right|}}"
        f"\\cdot\\sin(0.3t)\\sin({theta_rad:.4f})+{X_val:.4f},"
        f"42+t*\\sin({theta_rad:.4f})+e^{{{M_val:.4f}\\left|t\\right|}}"
        f"\\cdot\\sin(0.3t)\\cos({theta_rad:.4f})\\right)\""
    )
    return latex_str

def plot_fitting_results(x_data, y_data, theta, M, X, output_path="fit_plot_new.png"):
    """Plots the original data and the fitted curve, saving it to output_path."""
    print("Generating plot...")
    plt.figure(figsize=(10, 8))
    
    # Scatter plot of the input data
    plt.scatter(x_data, y_data, color='#3498db', s=8, alpha=0.6, label='Observed Data')
    
    # Generate points on the fitted curve for t in [6, 60]
    t_vals = np.linspace(6.0, 60.0, 2000)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    x_fitted = t_vals * cos_t - np.exp(M * np.abs(t_vals)) * np.sin(0.3 * t_vals) * sin_t + X
    y_fitted = 42.0 + t_vals * sin_t + np.exp(M * np.abs(t_vals)) * np.sin(0.3 * t_vals) * cos_t
    
    plt.plot(x_fitted, y_fitted, color='#e74c3c', linewidth=2.5, label='Fitted Parametric Curve')
    
    # Visual styling
    plt.title('Parametric Curve Fitting using L1 Optimization', fontsize=14, fontweight='bold')
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    
    theta_deg = np.degrees(theta)
    info_box = (
        f"Optimized Parameters:\n"
        f"$\\theta$ = {theta_deg:.4f}° ({theta:.6f} rad)\n"
        f"$M$ = {M:.6f}\n"
        f"$X$ = {X:.6f}"
    )
    plt.gca().text(
        0.05, 0.95, info_box,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fafafa', alpha=0.9, edgecolor='#bdc3c7')
    )
    
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved successfully as: {output_path}")

def main():
    csv_file = "xy_data (1).csv"
    x_data, y_data = load_data(csv_file)
    
    best_params, min_loss = fit_curve(x_data, y_data)
    theta_opt, M_opt, X_opt = best_params
    
    theta_deg = np.degrees(theta_opt)
    
    print("\n" + "="*50)
    print("                OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Theta (radians) : {theta_opt:.8f}")
    print(f"Theta (degrees) : {theta_deg:.6f}°")
    print(f"M (exp factor)  : {M_opt:.8f}")
    print(f"X (offset)      : {X_opt:.8f}")
    print(f"Total L1 Loss   : {min_loss:.8f}")
    print(f"Average L1 Loss : {min_loss / len(x_data):.8f} per point")
    print("="*50)
    
    latex_submission = generate_latex_format(theta_opt, M_opt, X_opt)
    print("\nLaTeX Submission Format:")
    print(latex_submission)
    print("="*50)
    
    plot_fitting_results(x_data, y_data, theta_opt, M_opt, X_opt)

if __name__ == '__main__':
    main()
