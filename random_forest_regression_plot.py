import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Generate sample data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# Fit the random forest regression model
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(x.reshape(-1, 1), y)

# Create a scatter plot of the data points
plt.scatter(x, y, color='blue', label='Data Points')

# Generate predictions from the random forest regressor
x_test = np.linspace(min(x), max(x), 100)
y_pred = regressor.predict(x_test.reshape(-1, 1))

# Plot the random forest regression curve
plt.plot(x_test, y_pred, color='red', label='Random Forest Regression')

# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Random Forest Regression')

# Add legend
plt.legend()

# Display the plot
plt.show()
