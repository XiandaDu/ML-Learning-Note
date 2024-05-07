import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from utils.create_dataset import create_dataset
from utils.plot_boundary import plot_decision_boundary

X, y, X_train, X_test, y_train, y_test = create_dataset()

m = len(X_train)

plt.figure(figsize=(14, 5))
# Plot a comparison using AdaBoosting(Adaptive Boosting) method
for subplot, learning_rate in ((121, 1), (122, 0.5)):
    # At first, the weight of each param is all the same
    sample_weights = np.ones(m)
    plt.subplot(subplot)

    # Train 5 times for each learning rate
    for i in range(5):
        svm_clf = SVC(kernel='rbf', C=0.05, random_state=42)
        svm_clf.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred = svm_clf.predict(X_train)
        # Increase the weight of each WRONG param.
        sample_weights[y_pred != y_train] *= (1 + learning_rate)
        plot_decision_boundary(svm_clf, X, y)
        plt.title('learning_rate = {}'.format(learning_rate))

plt.figure(figsize=(14, 5))
# Plot a comparison using AdaBoosting(Adaptive Boosting) method
for subplot, learning_rate in ((121, 0.5), (122, 0.5)):
    # At first, the weight of each param is all the same
    sample_weights = np.ones(m)
    plt.subplot(subplot)

    svm_clf = SVC(kernel='rbf', C=0.05, random_state=42)
    svm_clf.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred = svm_clf.predict(X_train)
    # Increase the weight of each WRONG param.
    # sample_weights[y_pred != y_train] *= (1 + learning_rate)
    plot_decision_boundary(svm_clf, X, y)
    plt.title('learning_rate = {}'.format(learning_rate))
plt.show()
