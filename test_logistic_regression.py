import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_data():
    data = pd.read_csv('datasets/test.csv')
    return data


def sigmazoid(z):
    return 1/(1 + np.exp(-z))


def logloss(y, y_pred):
    return -(y * np.log(y_pred)) - ((1 - y) * np.log(1 - y_pred))


def cost_function(y, y_pred):
    N = len(y)
    cost = np.sum(logloss(y, y_pred)) / N
    return cost


def main():
    data = get_data()
    print(data)


if __name__ == '__main__':
    main()
