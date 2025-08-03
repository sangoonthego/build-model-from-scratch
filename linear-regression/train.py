from data import generate_data
from model import LinearRegression
from evaluate import plot_linear_regression, plot_learning_curve
from split import train_test_split

# generate data
X, y = generate_data(sample_nums=100, feature_nums=1, noise=5)

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train model
linear_model = LinearRegression(learning_rate=0.0001, iter_nums=1000, normalize=True)
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_train)

print(f"Weights: w = {linear_model.w.flatten()}, Bias: b = {linear_model.b:.2f}")

# visualize
plot_linear_regression(X_train, y_train, y_pred)
plot_learning_curve(linear_model)