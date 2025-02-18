import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from logreg_class import LogisticRegression
import sys
from sklearn.metrics import accuracy_score


def get_data(path):
    data = pd.read_csv(path)
    return data


def accurcy(y_pred):
    datatrue = pd.read_csv("datasets/dataset_train.csv")
    print(f'Sklearn: {accuracy_score(datatrue["Hogwarts House"], y_pred)}')
    return np.sum(datatrue["Hogwarts House"] == y_pred)/len(y_pred)


def preprocessing_data(X_test):
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)

    return X_test


def load_weights_from_csv(path):
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
    return output


def main():
    lr = 0.01
    n_iters = 5000

    try:
        test_data = get_data(sys.argv[1])
    except Exception as e:
        print(e)
        return

    X_test = test_data.drop(['Hogwarts House', 'Index'], axis=1).select_dtypes(
        include=[int, float]).fillna(0).values

    X_test = preprocessing_data(X_test)

    clf = LogisticRegression(lr, n_iters)
    clf.models = load_weights_from_csv(sys.argv[2])
    y_pred = clf.predict(X_test)
    output = output_file(y_pred, test_data, 'prediciton.csv',)
    print(f"Homemade: {accurcy(output['Hogwarts House'])}")


if __name__ == '__main__':
    main()
