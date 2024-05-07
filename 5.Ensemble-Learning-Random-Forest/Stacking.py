import matplotlib.pyplot as plt
import numpy as np
from utils.create_dataset import create_dataset
from utils.plot_boundary import plot_decision_boundary
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

X, y, X_train, X_test, y_train, y_test = create_dataset()

# Define base learners
base_learners = [
    SVC(kernel='rbf', C=0.05, random_state=42),
    LogisticRegression(),
    KMeans(n_clusters=2, random_state=42)
]

# Initialize meta learner
meta_learner = DecisionTreeClassifier(random_state=42)

# Train base learners and make predictions
base_predictions_train = []
for base_learner in base_learners:
    base_learner.fit(X_train, y_train)
    base_predictions_train.append(base_learner.predict(X_train))

# Combine base learner predictions
stacked_X_train = np.column_stack(base_predictions_train)

# Train meta learner on combined predictions
meta_learner.fit(stacked_X_train, y_train)

# Make predictions on test data
base_predictions_test = []
for base_learner in base_learners:
    base_predictions_test.append(base_learner.predict(X_test))
    base_accuracy = accuracy_score(y_test, base_learner.predict(X_test))
    print("Accuracy of Base Model: {:.2f}%".format(base_accuracy * 100))
    plot_decision_boundary(base_learner, X, y)

stacked_X_test = np.column_stack(base_predictions_test)
meta_predictions_test = meta_learner.predict(stacked_X_test)

# Calculate accuracy of meta learner
meta_accuracy = accuracy_score(y_test, meta_predictions_test)
print("Accuracy of Stacked Model: {:.2f}%".format(meta_accuracy * 100))

plt.show()


