import numpy as np
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i, y[idx]))
                    self.bias -= self.learning_rate * y[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)


# Create a dataset with a pattern
np.random.seed(0)
X = np.random.randn(500, 2)
y = np.concatenate([-np.ones(250), np.ones(250)])
X[:250] += 2  # Shift the first 250 samples

# Fit SVM model
svm_model = SVM()
svm_model.fit(X, y)

# Plot decision boundary
x1 = np.linspace(-3, 5, 100)
x2 = np.linspace(-3, 5, 100)
X1, X2 = np.meshgrid(x1, x2)
X_grid = np.c_[X1.ravel(), X2.ravel()]
predictions = svm_model.predict(X_grid)
predictions = predictions.reshape(X1.shape)

plt.figure(figsize=(10, 6))
plt.contourf(X1, X2, predictions, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary')
plt.colorbar(label='Predicted Class')
plt.show()

