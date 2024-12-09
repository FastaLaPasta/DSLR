import os
import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def pair_plot(file):
    # loading dataset using seaborn
    df = pd.read_csv(file)
    sns.pairplot(df, hue='Hogwarts House')
    plt.show()


def main():
    if len(sys.argv) == 2:
        if os.path.isfile(sys.argv[1]):
            pair_plot(sys.argv[1])
        else:
            raise FileExistsError(f'FileExistsError: \
Wrong File/Path for : {sys.argv[1]}')
    else:
        raise ValueError(f'ValueError: \
Need 2 arguments, current number of arguments: {len(sys.argv)}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
