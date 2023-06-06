import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# Generate sample data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# Fit the K-Nearest Neighbors regression model
regressor = KNeighborsRegressor(n_neighbors=5)
regressor.fit(x.reshape(-1, 1), y)

# Create a scatter plot of the data points
plt.scatter(x, y, color='blue', label='Data Points')

# Generate predictions from the K-Nearest Neighbors regressor
x_test = np.linspace(min(x), max(x), 100)
y_pred = regressor.predict(x_test.reshape(-1, 1))

# Plot the K-Nearest Neighbors regression curve
plt.plot(x_test, y_pred, color='red', label='K-Nearest Neighbors Regression')

# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-Nearest Neighbors Regression')

# Add legend
plt.legend()

# Display the plot
plt.show()
