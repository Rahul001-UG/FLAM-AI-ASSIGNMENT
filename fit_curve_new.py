import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def l1_loss(params, x, y):
    theta, M, X = params
    ct, st = np.cos(theta), np.sin(theta)
    dx, dy = x - X, y - 42.0
    t = dx * ct + dy * st
    v_obs = -dx * st + dy * ct
    v_model = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    return np.sum(np.abs(v_obs - v_model))

def main():
    df = pd.read_csv("xy_data (1).csv")
    x, y = df['x'].values, df['y'].values
    
    bounds = [(0.0, np.radians(50.0)), (-0.05, 0.05), (0.0, 100.0)]
    res = minimize(l1_loss, [np.radians(25.0), 0.0, 50.0], args=(x, y), bounds=bounds, method='Nelder-Mead')
    theta, M, X = res.x
    
    print(f"Theta (rad): {theta:.6f} ({np.degrees(theta):.4f} deg)")
    print(f"M:           {M:.6f}")
    print(f"X:           {X:.6f}")
    print(f"L1 Loss:     {res.fun:.6f}")
    
    # Generate Desmos LaTeX
    th_r, m_r, x_r = round(theta, 4), round(M, 4), round(X, 4)
    print(f'LaTeX: "\\left(t*\\cos({th_r})-e^{{{m_r}\\left|t\\right|}}\\cdot\\sin(0.3t)\\sin({th_r})+{x_r},42+t*\\sin({th_r})+e^{{{m_r}\\left|t\\right|}}\\cdot\\sin(0.3t)\\cos({th_r})\\right)"')
    
    # Plotting
    plt.figure(figsize=(9, 7))
    plt.scatter(x, y, s=5, alpha=0.5, label="Data")
    t_vals = np.linspace(6, 60, 1000)
    x_fit = t_vals * np.cos(theta) - np.exp(M * np.abs(t_vals)) * np.sin(0.3 * t_vals) * np.sin(theta) + X
    y_fit = 42 + t_vals * np.sin(theta) + np.exp(M * np.abs(t_vals)) * np.sin(0.3 * t_vals) * np.cos(theta)
    plt.plot(x_fit, y_fit, color='red', label="Fit")
    plt.legend()
    plt.grid(True)
    plt.savefig("fit_plot_new.png")
    print("Saved plot to fit_plot_new.png")

if __name__ == "__main__":
    main()
