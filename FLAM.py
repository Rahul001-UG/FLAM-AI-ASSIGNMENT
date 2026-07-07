import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class ParametricCurveRegressor:
    """
    Class-based parametric curve regressor that fits the data to the target curve
    using L1 loss optimization with Nelder-Mead simplex method.
    """
    def __init__(self, data_path="xy_data (1).csv"):
        self.data_path = data_path
        self.x_data = None
        self.y_data = None
        self.optimal_params = None
        self.min_loss = None

    def read_csv(self):
        """Loads data coordinates from the CSV file."""
        print(f"Loading data points from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        self.x_data = df['x'].values
        self.y_data = df['y'].values

    def loss_function(self, params):
        """Computes L1 loss using rotated matrix transformation."""
        angle, exp_coef, x_offset = params
        c_th, s_th = np.cos(angle), np.sin(angle)
        
        # Apply transformation matrix to shift and rotate points
        coords = np.vstack([self.x_data - x_offset, self.y_data - 42.0])
        rot_matrix = np.array([[c_th, s_th], [-s_th, c_th]])
        rotated_coords = rot_matrix @ coords
        
        t_est = rotated_coords[0, :]
        v_obs = rotated_coords[1, :]
        
        # Calculate theoretical curve model displacement
        v_model = np.exp(exp_coef * np.abs(t_est)) * np.sin(0.3 * t_est)
        
        # Sum of absolute deviations (L1 loss)
        return np.sum(np.abs(v_obs - v_model))

    def run_optimization(self):
        """Finds optimized parameters using Nelder-Mead search method."""
        # Boundaries: angle (0 to 50 deg), exp_coef (-0.05 to 0.05), x_offset (0 to 100)
        bounds = [
            (0.0, np.radians(50.0)),
            (-0.05, 0.05),
            (0.0, 100.0)
        ]
        init_guess = [np.radians(25.0), 0.0, 50.0]
        
        print("Optimizing parameters with Nelder-Mead algorithm...")
        res = minimize(
            self.loss_function, 
            init_guess, 
            bounds=bounds, 
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        if not res.success:
            print("Warning: Optimization did not fully converge:", res.message)
            
        self.optimal_params = res.x
        self.min_loss = res.fun
        return res.x, res.fun

    def plot_fit(self, output_img="fit_plot_new.png"):
        """Plots observed dataset and fitted model curve."""
        if self.optimal_params is None:
            raise ValueError("Regressor must be optimized before plotting.")
            
        angle, exp_coef, x_offset = self.optimal_params
        
        print("Rendering results visualization...")
        plt.figure(figsize=(10, 8))
        
        # Draw dataset in dark purple
        plt.scatter(self.x_data, self.y_data, color='#3c1b5b', s=7, alpha=0.55, label='Observed points')
        
        # Construct fitted curve points for t in [6, 60]
        t_line = np.linspace(6.0, 60.0, 2000)
        c_th, s_th = np.cos(angle), np.sin(angle)
        
        x_fitted = t_line * c_th - np.exp(exp_coef * np.abs(t_line)) * np.sin(0.3 * t_line) * s_th + x_offset
        y_fitted = 42.0 + t_line * s_th + np.exp(exp_coef * np.abs(t_line)) * np.sin(0.3 * t_line) * c_th
        
        # Draw curve in bright orange
        plt.plot(x_fitted, y_fitted, color='#e67e22', linewidth=2.5, label='Optimal L1 Curve')
        
        # Set chart styles
        plt.title('Nonlinear Parametric Regression Fit', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
        text_legend = (
            f"Optimized Parameters:\n"
            f"$\\theta$ = {np.degrees(angle):.4f}° ({angle:.6f} rad)\n"
            f"$M$ = {exp_coef:.6f}\n"
            f"$X$ = {x_offset:.6f}"
        )
        plt.gca().text(0.05, 0.95, text_legend, transform=plt.gca().transAxes, fontsize=10.5,
                        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#fafafa', alpha=0.9, edgecolor='#bdc3c7'))
        
        plt.legend(loc='lower right')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_img, dpi=300)
        plt.close()
        print(f"Fitted chart saved to: {output_img}")

def main():
    regressor = ParametricCurveRegressor(data_path="xy_data (1).csv")
    regressor.read_csv()
    
    params, loss = regressor.run_optimization()
    angle, exp_coef, x_offset = params
    
    print("\n" + "#"*50)
    print("           PARAMETRIC REGRESSION COMPLETED")
    print("#"*50)
    print(f"Optimal Theta (rad) : {angle:.8f}")
    print(f"Optimal Theta (deg) : {np.degrees(angle):.4f}°")
    print(f"Optimal M (decay)   : {exp_coef:.8f}")
    print(f"Optimal X (offset)  : {x_offset:.8f}")
    print(f"Minimum L1 Loss     : {loss:.8f}")
    print(f"Mean Residual Error : {loss / len(regressor.x_data):.8f} per point")
    print("#"*50 + "\n")
    
    regressor.plot_fit(output_img="fit_plot_new.png")

if __name__ == '__main__':
    main()
