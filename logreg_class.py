import numpy as np


class LogisticRegression():

    def sigmoid(self, z):
        return (1 / (1 + np.exp(-z)))

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.classes = None
        self.models = []

    def compute_cost(self, y, preds):
        cost = -np.mean(y * np.log(preds) + (1 - y) * np.log(1 - preds))
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
                predictions = self.sigmoid(linear_pred)

                dw = (1/n_samples) * np.dot(X.T, (predictions - y_binary))
                db = (1/n_samples) * np.sum(predictions - y_binary)

                weights = weights - self.lr * dw
                bias = bias - self.lr * db

                # if _ % 100 == 0:
                #     cost = self.compute_cost(y, predictions)
                #     print(f"Iteration {_}, Cost: {cost:.4f}")
            self.models.append((weights, bias))
        return (self.models, bias)

    def predict(self, X):
        class_probs = []
        for weights, bias in self.models:
            linear_pred = np.dot(X, weights) + bias
            probs = self.sigmoid(linear_pred)
            class_probs.append(probs)

        class_probs = np.array(class_probs).T
        y_pred = np.argmax(class_probs, axis=1)
        return y_pred
