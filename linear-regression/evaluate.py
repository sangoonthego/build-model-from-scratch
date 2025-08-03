import numpy as np
import matplotlib.pyplot as plt

def plot_linear_regression(X, y, y_pred):
    plt.scatter(X, y, color="blue", label="Data")
    plt.plot(X, y_pred, color="red", label="Prediction")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression")
    plt.show()

def plot_learning_curve(linear_model):
    plt.plot(range(len(linear_model.losses)), linear_model.losses, color="green")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Learning Curve")
    plt.show()