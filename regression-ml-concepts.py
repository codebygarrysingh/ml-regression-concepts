# -*- coding: utf-8 -*-
"""
Regression ML SkLearn Python script
Author: Garry Singh

This script covers regression ml examples and supporting plots for describing fundamental concepts

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate some example data (temperature vs. lemonade sales)
temperature = np.array([80, 85, 90, 95, 100, 105, 110, 115])
lemonade_sales = np.array([20, 25, 30, 35, 40, 45, 50, 55])

# Create a linear regression model
model = LinearRegression()

# Reshape the data
temperature = temperature.reshape(-1, 1)

# Fit the model to the data
model.fit(temperature, lemonade_sales)

# Predict lemonade sales using the model
predicted_sales = model.predict(temperature)

# Create an interactive plot
plt.figure()
plt.scatter(temperature, lemonade_sales, color='blue', label='Data Points')
plt.plot(temperature, predicted_sales, color='red', linestyle='--', label='Linear Regression Line')
plt.xlabel('Temperature (°F)')
plt.ylabel('Lemonade Sales')
plt.legend()
plt.title('Linear Regression: Lemonade Sales vs. Temperature')

# Save the plot as a PNG file in the "plots" folder
plt.savefig('plots/linear_regression_plot.png')

# Show the interactive plot
plt.show()
