import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from logreg_class import LogisticRegression
import sys


def get_data(path):
    data = pd.read_csv(path)
    return data


def accurcy(y_pred, y_test):
    return np.sum(y_pred == y_test)/len(y_test)


def preprocessing_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def load_weights_from_csv(path='weights.csv'):
    df = pd.read_csv(path)
    n_features = len(df.columns) - 1
    models = []

    for index, row in df.iterrows():
        weights = row[:n_features].values
        bias = row.iloc[n_features]
        models.append((weights, bias))

    return models


def output_file(y_pred, t_data, path='prediction.csv'):
    house = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    pred = [house[p] for p in y_pred]
    output = pd.DataFrame({'Index': t_data['Index'], 'Hogwarts House': pred})
    output.to_csv('predictions.csv', index=False)
    print('predictions.csv file create and fill')


def main():
    lr = 0.01
    n_iters = 5000

    try:
        train_data = get_data(sys.argv[1])
        test_data = get_data(sys.argv[2])
    except Exception as e:
        print(e)
        return

    X = train_data.drop(['Hogwarts House', 'Index'], axis=1).select_dtypes(include=[int, float]).fillna(0).values
    X_test = test_data.drop(['Hogwarts House', 'Index'], axis=1).select_dtypes(include=[int, float]).fillna(0).values

    # le = LabelEncoder()
    # y_test = le.fit_transform(test_data['Hogwarts House'].values)

    X_train, X_test = preprocessing_data(X, X_test)

    clf = LogisticRegression(lr, n_iters)
    clf.models = load_weights_from_csv('weights.csv')
    y_pred = clf.predict(X_test)
    output_file(y_pred, test_data, 'prediciton.csv',)

    # print(accurcy(test_data['Hogwarts House'], output['Hogwarts House']))


if __name__ == '__main__':
    main()
