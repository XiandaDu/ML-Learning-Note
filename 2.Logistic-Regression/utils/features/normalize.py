"""Normalize features"""

import numpy as np


def normalize(features):

    features_normalized = np.copy(features).astype(float)

    # Calculate mean
    features_mean = np.mean(features, 0)

    # calculate std deviation
    features_deviation = np.std(features, 0)

    # normalize the data by getting the relative difference
    if features.shape[0] > 1:
        features_normalized -= features_mean

    # Scale it with the awareness of not divided by 0
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation
