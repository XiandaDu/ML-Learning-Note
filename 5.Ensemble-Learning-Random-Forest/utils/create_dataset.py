from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


def create_dataset():
    # Create a sample dataset waiting to be classified
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'yo', alpha=0.6)
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 1], 'bs', alpha=0.6)
    plt.show()
    return X, y, X_train, X_test, y_train, y_test
