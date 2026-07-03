import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df['x'].values, df['y'].values

def l1_loss(params, x_data, y_data):
    theta, M, X = params
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx, dy = x_data - X, y_data - 42.0
    t_est = dx * cos_t + dy * sin_t
    v_obs = -dx * sin_t + dy * cos_t
    v_model = np.exp(M * np.abs(t_est)) * np.sin(0.3 * t_est)
    return np.sum(np.abs(v_obs - v_model))

def fit_curve(x_data, y_data):
    bounds = [(0.0, np.radians(50.0)), (-0.05, 0.05), (0.0, 100.0)]
    result = minimize(l1_loss, [np.radians(25.0), 0.0, 50.0], args=(x_data, y_data), bounds=bounds, method='Nelder-Mead')
    return result.x, result.fun

def plot_fitting_results(x_data, y_data, theta, M, X, output_path="fit_plot_new.png"):
    plt.figure(figsize=(10, 8))
    plt.scatter(x_data, y_data, color='#3498db', s=8, alpha=0.6, label='Observed Data')
    t_vals = np.linspace(6.0, 60.0, 2000)
    x_fitted = t_vals * np.cos(theta) - np.exp(M * np.abs(t_vals)) * np.sin(0.3 * t_vals) * np.sin(theta) + X
    y_fitted = 42.0 + t_vals * np.sin(theta) + np.exp(M * np.abs(t_vals)) * np.sin(0.3 * t_vals) * np.cos(theta)
    plt.plot(x_fitted, y_fitted, color='#e74c3c', linewidth=2.5, label='Fitted Parametric Curve')
    plt.title('Parametric Curve Fitting using L1 Optimization', fontsize=14, fontweight='bold')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    info_box = f"$\\theta$ = {np.degrees(theta):.4f}°\n$M$ = {M:.6f}\n$X$ = {X:.6f}"
    plt.gca().text(0.05, 0.95, info_box, transform=plt.gca().transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#fafafa', alpha=0.9, edgecolor='#bdc3c7'))
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    x_data, y_data = load_data("xy_data (1).csv")
    best_params, min_loss = fit_curve(x_data, y_data)
    theta_opt, M_opt, X_opt = best_params
    
    print("\n" + "="*50)
    print("                OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Theta (rad) : {theta_opt:.6f} ({np.degrees(theta_opt):.4f}°)")
    print(f"M           : {M_opt:.6f}")
    print(f"X (offset)  : {X_opt:.6f}")
    print(f"Total Loss  : {min_loss:.6f}")
    print("="*50)
    
    plot_fitting_results(x_data, y_data, theta_opt, M_opt, X_opt)

if __name__ == '__main__':
    main()
