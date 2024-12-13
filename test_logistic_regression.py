import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import datasets


class LogisticRegression():

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.classes = None
        self.models = []

    def compute_cost(self, y, predictions):
        cost = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        return cost

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        for cls in self.classes:
            weights = np.zeros(n_features)
            bias = 0

            y_binary = np.where(y == cls, 1, 0)

            for _ in range(self.n_iters):
                linear_pred = np.dot(X, weights) + bias
                predictions = sigmoid(linear_pred)

                dw = (1/n_samples) * np.dot(X.T, (predictions - y_binary))
                db = (1/n_samples) * np.sum(predictions - y_binary)

                weights = weights - self.lr * dw
                bias = bias - self.lr * db

                if _ % 100 == 0:
                    cost = self.compute_cost(y, predictions)
                    print(f"Iteration {_}, Cost: {cost:.4f}")
            self.models.append((weights, bias))

    def predict(self, X):
        class_probs = []
        for weights, bias in self.models:
            linear_pred = np.dot(X, weights) + bias
            probs = sigmoid(linear_pred)
            class_probs.append(probs)

        class_probs = np.array(class_probs).T
        y_pred = np.argmax(class_probs, axis=1)
        return y_pred


def get_data():
    data = pd.read_csv('datasets/dataset_train.csv')
    return data


def sigmoid(z):
    return (1 / (1 + np.exp(-z)))


def accurcy(y_pred, y_test):
    return np.sum(y_pred == y_test)/len(y_test)


def preprocessing_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def main():
    lr = 0.1
    n_iters = 2000

    data = get_data()
    X = data.drop('Hogwarts House', axis=1).select_dtypes(include=[int, float]).fillna(0).values
    le = LabelEncoder()
    y = le.fit_transform(data['Hogwarts House'].values)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    X_train, X_test = preprocessing_data(X_train, X_test)

    clf = LogisticRegression(lr, n_iters)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accurcy(y_pred, y_test)
    print(acc, y_pred)


if __name__ == '__main__':
    main()
