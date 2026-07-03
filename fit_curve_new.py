import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def fetch_coordinates(file_name="xy_data (1).csv"):
    """Reads the dataset coordinates from the local CSV file."""
    dataset = pd.read_csv(file_name)
    return dataset['x'].values, dataset['y'].values

def evaluate_residual(parameters, x_vals, y_vals):
    """
    Computes the L1 distance between the observed data coordinates
    and the theoretical parametric curve.
    """
    rad, decay, offset_x = parameters
    cos_rad = np.cos(rad)
    sin_rad = np.sin(rad)
    
    # Calculate coordinate offsets from origin/rotation point
    dx = x_vals - offset_x
    dy = y_vals - 42.0
    
    # Project to find t and orthogonal distance (v_obs) in rotated coordinate system
    t_est = dx * cos_rad + dy * sin_rad
    v_obs = -dx * sin_rad + dy * cos_rad
    
    # Theoretical curve model function
    v_theo = np.exp(decay * np.abs(t_est)) * np.sin(0.3 * t_est)
    
    # L1 Loss representation
    return np.sum(np.abs(v_obs - v_theo))

def execute_fitting(x_arr, y_arr):
    """Performs bounded optimization to determine unknown parameters."""
    parameter_bounds = [
        (0.0, 50.0 * np.pi / 180.0), # theta limits (0 to 50 degrees)
        (-0.05, 0.05),               # M limits (-0.05 to 0.05)
        (0.0, 100.0)                 # X limits (0 to 100)
    ]
    initial_estimate = [0.4, 0.0, 50.0]
    
    search_result = minimize(
        evaluate_residual,
        initial_estimate,
        args=(x_arr, y_arr),
        bounds=parameter_bounds,
        method='Nelder-Mead'
    )
    return search_result.x, search_result.fun

def export_visualization(x_arr, y_arr, rad, decay, offset_x, image_name="fit_plot_new.png"):
    """Generates a high-quality visualization of data and the curve."""
    print("Rendering plot...")
    plt.figure(figsize=(10, 7.5))
    
    # Plot original data points in purple
    plt.scatter(x_arr, y_arr, color='#8e44ad', s=7, alpha=0.55, label='CSV Coordinates')
    
    # Plot predicted parametric curve in dark green
    t_steps = np.linspace(6.0, 60.0, 1500)
    c_val = np.cos(rad)
    s_val = np.sin(rad)
    
    x_pred = t_steps * c_val - np.exp(decay * np.abs(t_steps)) * np.sin(0.3 * t_steps) * s_val + offset_x
    y_pred = 42.0 + t_steps * s_val + np.exp(decay * np.abs(t_steps)) * np.sin(0.3 * t_steps) * c_val
    
    plt.plot(x_pred, y_pred, color='#27ae60', linewidth=2.5, label='Optimized L1 Fit')
    
    # Graphic detailing
    plt.title('Parametric Curve Regression Model', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('X-Axis Coordinate')
    plt.ylabel('Y-Axis Coordinate')
    
    deg_val = rad * 180.0 / np.pi
    text_legend = (
        f"Curve Parameters:\n"
        f"Angle (deg) = {deg_val:.4f}°\n"
        f"Decay (M)   = {decay:.6f}\n"
        f"Offset (X)  = {offset_x:.6f}"
    )
    
    plt.gca().text(
        0.05, 0.95, text_legend, transform=plt.gca().transAxes, fontsize=10.5,
        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='#fdfefe', alpha=0.9, edgecolor='#bdc3c7')
    )
    
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(image_name, dpi=200)
    plt.close()
    print(f"Plot saved: {image_name}")

def main():
    print("Initiating coordinate regression...")
    try:
        x_data, y_data = fetch_coordinates()
    except FileNotFoundError:
        print("Data file 'xy_data (1).csv' not found in workspace.")
        return
        
    estimated_params, final_loss = execute_fitting(x_data, y_data)
    theta_rad, m_coeff, x_shift = estimated_params
    theta_deg = theta_rad * 180.0 / np.pi
    
    print("\n" + "*"*45)
    print("         REGRESSION COEFFICIENT ESTIMATION")
    print("*"*45)
    print(f"Optimal Theta (rad) : {theta_rad:.6f}")
    print(f"Optimal Theta (deg) : {theta_deg:.4f}°")
    print(f"Optimal M (decay)   : {m_coeff:.6f}")
    print(f"Optimal X (shift)   : {x_shift:.6f}")
    print(f"L1 Residual Error   : {final_loss:.6f}")
    print("*"*45)

    export_visualization(x_data, y_data, theta_rad, m_coeff, x_shift)
    print("Optimization process completed successfully.")

if __name__ == "__main__":
    main()
