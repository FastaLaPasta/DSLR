import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from logreg_class import LogisticRegression
import sys


def get_data(path):
    data = pd.read_csv(path)
    return data


def accurcy(y_pred, y_test):
    return np.sum(y_pred == y_test)/len(y_test)


def preprocessing_data(X_train):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    return X_train


def save_weights_to_csv(models, path='weights.csv'):
    weights_list = []

    for weights, bias in models:
        row = list(weights) + [bias]
        weights_list.append(row)

    columns = [f'W_{i}' for i in range(len(weights_list[0]) - 1)] + ['Bias']
    df = pd.DataFrame(weights_list, columns=columns)
    df.to_csv(path, index=False)


def main():
    lr = 0.01
    n_iters = 5000

    try:
        train_data = get_data(sys.argv[1])
    except Exception as e:
        print(e)
        return

    X = train_data.drop(['Hogwarts House', 'Index'], axis=1).select_dtypes(include=[int, float]).fillna(0).values
    le = LabelEncoder()
    y = le.fit_transform(train_data['Hogwarts House'].values)
    X_train = preprocessing_data(X)

    clf = LogisticRegression(lr, n_iters)
    clf.fit(X_train, y)
    save_weights_to_csv(clf.models, 'weights.csv')


if __name__ == '__main__':
    main()
