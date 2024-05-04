from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import numpy as np


def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x1, x2, y_pred, cmap=custom_cmap, alpha=0.3)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58', '#4c4c7f', '#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'yo', alpha=0.6)
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 1], 'bs', alpha=0.6)
    plt.axis(axes)
    plt.xlabel('x1')
    plt.xlabel('x2')


# Create a sample dataset waiting to be classified
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'yo', alpha=0.6)
plt.plot(X[:, 0][y == 0], X[:, 1][y == 1], 'bs', alpha=0.6)
plt.show()

# Create a Bagging Classifier, it can also be considered as a random forest:
# - 500 base classifiers
# - Maximum 100 samples per classifier
# - Bootstrap sampling， meaning one sample can be chosen multiple times
# - Parallel processing with all available CPU cores
# - Random seed for reproducibility set to 42
random_forest_clf = BaggingClassifier(DecisionTreeClassifier(),
                                      n_estimators=500,
                                      max_samples=100,
                                      bootstrap=True,
                                      n_jobs=-1,
                                      random_state=42
                                      )
random_forest_clf.fit(X_train, y_train)
y_pred = random_forest_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

# Compare the accuracy difference when using bagging and not using bagging.
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_tree))

# Plot the decision boundary
plt.figure(figsize=(12, 5))
plt.subplot(121)
plot_decision_boundary(tree_clf, X, y)
plt.title('Decision Tree')
plt.subplot(122)
plot_decision_boundary(random_forest_clf, X, y)
plt.title('Decision Tree With Bagging')
plt.show()
