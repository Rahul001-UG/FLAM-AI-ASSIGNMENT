import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def import_xy_dataset(csv_filepath):
    """Fetches the x and y coordinates from the CSV file."""
    data_frame = pd.read_csv(csv_filepath)
    return data_frame['x'].values, data_frame['y'].values

def compute_l1_discrepancy(parameters, x_points, y_points):
    """Calculates the absolute residual distance between data and model."""
    rad_angle, decay_factor, shift_x = parameters
    c_th, s_th = np.cos(rad_angle), np.sin(rad_angle)
    
    # Calculate coordinate shifts
    x_shifted = x_points - shift_x
    y_shifted = y_points - 42.0
    
    # Projected parameter (t) along the rotated main axis
    t_coords = x_shifted * c_th + y_shifted * s_th
    
    # Observed orthogonal projection compared to theoretical model values
    observed_dev = y_shifted * c_th - x_shifted * s_th
    expected_dev = np.exp(decay_factor * np.abs(t_coords)) * np.sin(0.3 * t_coords)
    
    return np.sum(np.abs(observed_dev - expected_dev))

def regress_parametric_curve(x_data, y_data):
    """Executes simplex optimization to estimate unknown variables."""
    # Bounded limits: theta (0 to 50 deg), M (-0.05 to 0.05), X (0 to 100)
    search_boundaries = [
        (0.0, 50.0 * np.pi / 180.0),
        (-0.05, 0.05),
        (0.0, 100.0)
    ]
    starting_points = [0.4, 0.0, 50.0]
    
    opt_output = minimize(
        compute_l1_discrepancy, starting_points, args=(x_data, y_data),
        bounds=search_boundaries, method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}
    )
    
    if not opt_output.success:
        print("Note: Simplex search failed to converge:", opt_output.message)
        
    return opt_output.x, opt_output.fun

def generate_curve_visualization(x_data, y_data, rad_angle, decay_factor, shift_x, filename="fit_plot_new.png"):
    """Saves a high-quality visualization of the fitted curve regression."""
    print("Generating visualization...")
    plt.figure(figsize=(9, 7))
    
    # Plot raw data in slate green
    plt.scatter(x_data, y_data, color='#16a085', s=6, alpha=0.5, label='Raw Coordinates')
    
    # Generate points along the fitted curve
    t_range = np.linspace(6.0, 60.0, 1500)
    c_th, s_th = np.cos(rad_angle), np.sin(rad_angle)
    
    x_model = t_range * c_th - np.exp(decay_factor * np.abs(t_range)) * np.sin(0.3 * t_range) * s_th + shift_x
    y_model = 42.0 + t_range * s_th + np.exp(decay_factor * np.abs(t_range)) * np.sin(0.3 * t_range) * c_th
    
    # Plot curve in purple
    plt.plot(x_model, y_model, color='#8e44ad', linewidth=2, label='Simplex L1 Fit')
    
    # Aesthetics
    plt.title('Parametric Regression Model', fontsize=12, fontweight='bold')
    plt.xlabel('Horizontal Coordinate')
    plt.ylabel('Vertical Coordinate')
    
    deg_angle = rad_angle * 180.0 / np.pi
    legend_details = (
        f"Model Estimates:\n"
        f"Angle = {deg_angle:.4f}°\n"
        f"Decay = {decay_factor:.5f}\n"
        f"Shift = {shift_x:.5f}"
    )
    plt.gca().text(0.05, 0.95, legend_details, transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='#fafafa', alpha=0.85, edgecolor='#bdc3c7'))
    
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Plot exported to: {filename}")

def main():
    target_csv = "xy_data (1).csv"
    x_coords, y_coords = import_xy_dataset(target_csv)
    
    optimized_vars, objective_value = regress_parametric_curve(x_coords, y_coords)
    angle_est, decay_est, shift_est = optimized_vars
    
    print("\n" + "~"*45)
    print("         PARAMETRIC REGRESSION RESULTS")
    print("~"*45)
    print(f"Theta (rad)  : {angle_est:.8f}")
    print(f"Theta (deg)  : {angle_est * 180.0 / np.pi:.4f}°")
    print(f"M (decay)    : {decay_est:.8f}")
    print(f"X (shift)    : {shift_est:.8f}")
    print(f"Residual Sum : {objective_value:.8f}")
    print("~"*45)
    
    generate_curve_visualization(x_coords, y_coords, angle_est, decay_est, shift_est)

if __name__ == '__main__':
    main()
