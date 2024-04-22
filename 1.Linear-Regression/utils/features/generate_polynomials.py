"""Add polynomial features to the features set"""

import numpy as np
from utils.features.normalize import normalize


def generate_polynomials(dataset, polynomial_degree, normalize_data=False):
    """transform methods:
    x1, x2, x1^2, x2^2, x1*x2, x1*x2^2, etc.
    """

    # Split into left and right two parts.
    features_split = np.array_split(dataset, 2, axis=1)
    dataset_1 = features_split[0]
    dataset_2 = features_split[1]

    (num_examples_1, num_features_1) = dataset_1.shape
    (num_examples_2, num_features_2) = dataset_2.shape

    if num_examples_1 != num_examples_2:
        raise ValueError('Can not generate polynomials for two sets with different number of rows')

    if num_features_1 == 0 and num_features_2 == 0:
        raise ValueError('Can not generate polynomials for two sets with no columns')

    if num_features_1 == 0:
        dataset_1 = dataset_2
    elif num_features_2 == 0:
        dataset_2 = dataset_1

    # Choose the smaller one as the num_features
    num_features = num_features_1 if num_features_1 < num_examples_2 else num_features_2
    dataset_1 = dataset_1[:, :num_features]
    dataset_2 = dataset_2[:, :num_features]

    polynomials = np.empty((num_examples_1, 0))

    # Generate all combinations of same degree x1^?*x2^? pair.
    for i in range(1, polynomial_degree + 1):
        for j in range(i + 1):
            polynomial_feature = (dataset_1 ** (i - j)) * (dataset_2 ** j)
            polynomials = np.concatenate((polynomials, polynomial_feature), axis=1)

    if normalize_data:
        polynomials = normalize(polynomials)[0]

    return polynomials

data = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
generate_polynomials(data, 4)