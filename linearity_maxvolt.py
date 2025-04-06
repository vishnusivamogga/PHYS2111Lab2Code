# -*- coding: utf-8 -*-
"""
This is the code used to plot Figure 6 of the Lab Report. It scatters the points and fits a linear fit to the data. 
"""

import matplotlib.pyplot as plt
import numpy as np
slit_arr = np.array([0.1, 0.2, 0.4])
volt_arr = np.array([0.50, 2.74, 8.83])
plt.scatter(slit_arr, volt_arr, marker='x')
plt.xlabel('Slit Width(mm)')
plt.ylabel('Recorded Maximum Voltage(V)')
plt.grid(True)
coefficients = np.polyfit(slit_arr, volt_arr, 1)  # Linear fit (degree = 1)
poly = np.poly1d(coefficients)  # Create a polynomial object
plt.plot(slit_arr, poly(slit_arr), color='green', label='Best Fit Line')

plt.show()
