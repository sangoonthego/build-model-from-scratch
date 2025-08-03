import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.0001, iter_nums=1000, normalize=False):
        self.learning_rate = learning_rate
        self.iter_nums = iter_nums
        self.normalize = normalize
        self.w = None
        self.b = 0
        self.losses = []

    def fit(self, X, y):
        if self.normalize:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0)
            X = (X - self.X_mean) / self.X_std
        else:
            self.X_mean = None
            self.X_std = None

        sample_nums, feature_nums = X.shape
        self.w = np.zeros((feature_nums, 1))

        for i in range(self.iter_nums):
            y_pred = self.predict(X)
            error = y - y_pred
            mse = np.mean(error ** 2)
            self.losses.append(mse)

            # early stopping
            if i > 0 and abs(self.losses[-1] - self.losses[-2]) < 1e-6:
                break
            # gradient
            dw = -(2 / sample_nums) * np.dot(X.T, error)
            db = -(2 / sample_nums) * np.sum(error)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        if self.normalize and self.X_mean is not None:
            X = (X - self.X_mean) / self.X_std
        return np.dot(X, self.w) + self.b