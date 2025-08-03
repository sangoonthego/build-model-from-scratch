# pipeline: gen data -> create model Linear to train (fit and predict)
# -> evaluate (using mse and visual)

import numpy as np

def generate_data(sample_nums=100, feature_nums=1, noise=5):
    np.random.seed(42)
    X = np.random.rand(sample_nums, 1) * 100
    true_w = np.random.randn(feature_nums, 1)
    true_b = np.random.randn()
    y = np.dot(X, true_w) + true_b + np.random.randn(sample_nums, 1) * noise
    return X, y