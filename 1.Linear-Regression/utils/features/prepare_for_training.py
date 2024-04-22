"""Prepares the dataset for training"""

import numpy as np
from utils.features.normalize import normalize
from utils.features.generate_sinusoids import generate_sinusoids
from utils.features.generate_polynomials import generate_polynomials


def prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):

    # Calculate the number of samples (number of rows)
    num_examples = data.shape[0]

    data_processed = np.copy(data)

    # pre-processing
    features_mean = 0
    features_deviation = 0
    data_normalized = data_processed
    if normalize_data:
        (
            data_normalized,
            features_mean,
            features_deviation
        ) = normalize(data_processed)

        data_processed = data_normalized

    # sinusoidal feature transform
    if sinusoid_degree > 0:
        sinusoids = generate_sinusoids(data_normalized, sinusoid_degree)
        data_processed = np.concatenate((data_processed, sinusoids), axis=1)

    # polynomial feature transform
    if polynomial_degree > 0:
        polynomials = generate_polynomials(data_normalized, polynomial_degree, normalize_data)
        data_processed = np.concatenate((data_processed, polynomials), axis=1)

    # add a column of 1s at the left, to make the ML program understand intercept term
    # 1*b0 + x1*b1 +...
    data_processed = np.hstack((np.ones((num_examples, 1)), data_processed))

    return data_processed, features_mean, features_deviation
