from utils.create_dataset import create_dataset
from utils.plot_boundary import plot_decision_boundary
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


X, y, X_train, X_test, y_train, y_test = create_dataset()

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
