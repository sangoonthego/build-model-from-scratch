import numpy as np

def train_test_split(X, y, test_size=0.2):
    sample_nums = 100
    indices = np.arange(sample_nums)
    np.random.shuffle(indices)

    test_count = int(sample_nums * test_size)
    test_index = indices[:test_count]
    train_index = indices[test_count:]

    return X[train_index], X[test_index], y[train_index], y[test_index]