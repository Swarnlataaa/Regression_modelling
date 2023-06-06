import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# Generate sample data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# Fit the Neural Network regression model
regressor = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=1000)
regressor.fit(x.reshape(-1, 1), y)

# Create a scatter plot of the data points
plt.scatter(x, y, color='blue', label='Data Points')

# Generate predictions from the Neural Network regressor
x_test = np.linspace(min(x), max(x), 100)
y_pred = regressor.predict(x_test.reshape(-1, 1))

# Plot the Neural Network regression curve
plt.plot(x_test, y_pred, color='red', label='Neural Network Regression')

# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Neural Network Regression')

# Add legend
plt.legend()

# Display the plot
plt.show()
