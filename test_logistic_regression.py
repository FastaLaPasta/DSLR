import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_data():
    data = pd.read_csv('datasets/test.csv')
    return data


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def logloss(y, y_pred):
    return -(y * np.log(y_pred)) - ((1 - y) * np.log(1 - y_pred))


def cost_function(y, y_pred):
    N = len(y)
    cost = np.sum(logloss(y, y_pred)) / N
    return cost


def gradient_descent(X, y, parameters, L, Ir):
    N = X.shape[0]
    for iteration in range(Ir):
        z = np.dot(X, parameters)
        prediction = sigmoid(z)

        error = prediction - y

        gradient = np.dot(X.T, error) / N
        
        parameters -= L * gradient
    return parameters
            


def main():
    learning_rates = 0.01
    iteration_rate = 1000
    data = get_data()

    
    print(data)


if __name__ == '__main__':
    main()
