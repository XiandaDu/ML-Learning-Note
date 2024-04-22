"""Normalize features"""

import numpy as np


def normalize(features):

    features_normalized = np.copy(features).astype(float)

    # for each column, calculate the mean
    features_mean = np.mean(features, 0)

    # for each column, calculate the std deviation
    features_deviation = np.std(features, 0)

    # If there is more than one row, we can just focus on the difference between each column.
    if features.shape[0] > 1:
        features_normalized -= features_mean

    # Do a scale on the normalized data, if the std deviation is 0, which means the feature
    # in this column is consistent, then set it to 1, which means no scaling is required.
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation
