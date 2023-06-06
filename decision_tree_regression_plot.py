import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Generate sample data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# Fit the decision tree regression model
regressor = DecisionTreeRegressor()
regressor.fit(x.reshape(-1, 1), y)

# Create a scatter plot of the data points
plt.scatter(x, y, color='blue', label='Data Points')

# Generate predictions from the decision tree regressor
x_test = np.linspace(min(x), max(x), 100)
y_pred = regressor.predict(x_test.reshape(-1, 1))

# Plot the decision tree regression curve
plt.plot(x_test, y_pred, color='red', label='Decision Tree Regression')

# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Decision Tree Regression')

# Add legend
plt.legend()

# Display the plot
plt.show()
