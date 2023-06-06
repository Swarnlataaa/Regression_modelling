import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# Fit the polynomial regression model
degree = 2
coefficients = np.polyfit(x, y, degree)

# Create a scatter plot of the data points
plt.scatter(x, y, color='blue', label='Data Points')

# Generate polynomial features
x_poly = np.linspace(min(x), max(x), 100)
y_poly = np.polyval(coefficients, x_poly)

# Plot the polynomial regression curve
plt.plot(x_poly, y_poly, color='red', label='Polynomial Regression Curve')

# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Regression')

# Add legend
plt.legend()

# Display the plot
plt.show()
