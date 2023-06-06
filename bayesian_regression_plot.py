import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge

# Generate sample data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# Fit the Bayesian regression model
regressor = BayesianRidge()
regressor.fit(x.reshape(-1, 1), y)

# Create a scatter plot of the data points
plt.scatter(x, y, color='blue', label='Data Points')

# Generate predictions from the Bayesian regressor
x_test = np.linspace(min(x), max(x), 100)
y_pred, y_std = regressor.predict(x_test.reshape(-1, 1), return_std=True)

# Plot the Bayesian regression curve
plt.plot(x_test, y_pred, color='red', label='Bayesian Regression')

# Plot the uncertainty (±1 standard deviation)
plt.fill_between(x_test, y_pred - y_std, y_pred + y_std, color='gray', alpha=0.3)

# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Bayesian Regression')

# Add legend
plt.legend()

# Display the plot
plt.show()
