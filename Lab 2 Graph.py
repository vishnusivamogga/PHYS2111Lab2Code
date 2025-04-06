# -*- coding: utf-8 -*-
"""
Most of this code is a function which fits the curve, plots it, obtains the error bounds and computes the distances from the maximima with errors. 
The obtained experimental data is put through this function to get the relevant data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the single-slit diffraction pattern function
def single_slit(x, I0, a, x0):
    beta = np.pi * a * (x - x0)
    return I0 * (np.sinc(beta / np.pi)**2)

# Function to perform fitting, calculate minima, and plot results
def process_and_plot_slit_data(distance, voltage, slit_width_label):
    # Fit the data
    popt, pcov = curve_fit(single_slit, distance, voltage, p0=[1.0, 0.1, 6])
    
    # Extract the fitted parameters and their errors
    I0, a, x0 = popt
    I0_err, a_err, x0_err = np.sqrt(np.diag(pcov))
    
    # Print the fitted parameters and their uncertainties
    print(f"Fitted parameters for {slit_width_label} slit:")
    print(f"I0 = {I0:.2f} ± {I0_err:.2f}")
    print(f"a = {a:.4f} ± {a_err:.4f} mm⁻¹")
    print(f"x0 = {x0:.2f} ± {x0_err:.2f} mm")
    
    # Calculate minima positions
    right_minima = x0 + 1 / a
    left_minima = x0 - 1 / a
    
    # Compute distances to minima
    distance_to_right_minima = abs(right_minima - x0)
    distance_to_left_minima = abs(left_minima - x0)
    average_distance = (distance_to_right_minima + distance_to_left_minima) / 2
    
    # Propagate error for average distance
    propagated_error = (1 / 2) * np.sqrt(
        a_err**2 * (1 / a**2)**2 + 2 * x0_err**2
    )
    
    # Print the results
    print(f"Central Maximum Position: x₀ = {x0:.2f} mm")
    print(f"Right Minima Position: x = {right_minima:.2f} mm")
    print(f"Left Minima Position: x = {left_minima:.2f} mm")
    print(f"Distance to Right Minima: {distance_to_right_minima:.2f} mm")
    print(f"Distance to Left Minima: {distance_to_left_minima:.2f} mm")
    print(f"Average Distance to Minima: {average_distance:.2f} ± {propagated_error:.2f} mm")
    
    # Generate the fitted curve
    x_fit = np.linspace(min(distance), max(distance), 500)
    y_fit = single_slit(x_fit, *popt)
    
    # Plot the measured data and fitted curve
    plt.figure(figsize=(10, 6))
    plt.scatter(distance, voltage, color="#4caf50", label="Measured Data", s=60, edgecolors="black", zorder=3, marker='x')
    plt.plot(x_fit, y_fit, color="#ff5722", linewidth=2.5, label="Fitted Curve", zorder=2)
    plt.xlabel("Distance (mm)", fontsize=14, labelpad=10)
    plt.ylabel("Voltage (V)", fontsize=14, labelpad=10)
    plt.title(f"{slit_width_label} Slit Diffraction Pattern (with Fit)", fontsize=16, weight="bold")
    plt.grid(color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    plt.legend(fontsize=12, loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.show()

#Running the function through the obtained data
distance_01mm = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9, 9.25, 9.5, 9.75, 10, 10.25, 10.5, 10.75, 11, 11.25, 11.5, 11.75, 12, 12.25, 12.5, 12.75, 13, 13.25, 13.5, 13.75])
voltage_01mm = np.array([0.006, 0.008, 0.011, 0.017, 0.025, 0.035, 0.05, 0.068, 0.091, 0.114, 0.139, 0.165, 0.204, 0.245, 0.282, 0.316, 0.346, 0.371, 0.396, 0.422, 0.448, 0.468, 0.488, 0.503, 0.508, 0.507, 0.49, 0.472, 0.452, 0.43, 0.403, 0.383, 0.353, 0.315, 0.282, 0.248, 0.217, 0.193, 0.161, 0.135, 0.108, 0.088, 0.063, 0.046, 0.033, 0.025, 0.017, 0.014, 0.012, 0.013, 0.015, 0.018, 0.021, 0.024, 0.026, 0.028])
process_and_plot_slit_data(distance_01mm, voltage_01mm, "0.1mm")

distance_02mm = np.array([0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2, 2.125, 2.25, 2.375, 2.5, 2.625, 2.75, 2.875, 3, 3.125, 3.25, 3.375, 3.5, 3.625, 3.75, 3.875, 4, 4.125, 4.25, 4.375, 4.5, 4.625, 4.75, 4.875, 5, 5.125, 5.25, 5.375, 5.5, 5.625, 5.75, 5.875, 6, 6.125, 6.25, 6.375, 6.5, 6.625, 6.75, 6.875, 7, 7.125, 7.25, 7.375, 7.5, 7.625])
voltage_02mm = np.array([0.039, 0.025, 0.015, 0.013, 0.028, 0.057, 0.108, 0.176, 0.274, 0.387, 0.527, 0.673, 0.839, 1.03, 1.247, 1.458, 1.663, 1.81, 1.973, 2.069, 2.26, 2.385, 2.506, 2.617, 2.663, 2.735, 2.742, 2.738, 2.71, 2.645, 2.545, 2.406, 2.272, 2.072, 1.876, 1.729, 1.542, 1.337, 1.149, 0.98, 0.851, 0.747, 0.66, 0.524, 0.393, 0.294, 0.202, 0.142, 0.077, 0.041, 0.019, 0.007, 0.006, 0.012, 0.024, 0.038, 0.055, 0.068, 0.082, 0.091, 0.097, 0.099][::-1])
process_and_plot_slit_data(distance_02mm, voltage_02mm, "0.2mm")


distance_04mm = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2])
voltage_04mm = np.array([0.207, 0.238, 0.287, 0.352, 0.397, 0.4, 0.366, 0.304, 0.224, 0.141, 0.093, 0.11, 0.23, 0.494, 0.933, 1.524, 2.347, 3.402, 4.572, 5.835, 7.221, 8.285, 8.285,  8.285, 8.285, 8.285, 8.285, 7.564, 6.309, 4.957, 3.76, 2.69, 1.848, 1.161, 0.632, 0.353, 0.192, 0.144, 0.177, 0.239, 0.312, 0.369, 0.398])
process_and_plot_slit_data(distance_04mm, voltage_04mm, "0.4mm")


