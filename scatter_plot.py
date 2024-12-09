import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
from stats import mean as mn


def pearson_corr(df, col1, col2):
    x = df[col1]
    y = df[col2]

    mean_x = mn(x)
    mean_y = mn(y)

    # compute sum of gap from mean as numerator
    numerator = sum((x - mean_x) * (y - mean_y))
    # compute sum of standard deviation as denominator
    denominator = (sum((x - mean_x)**2) * sum((y - mean_y)**2))**0.5

    # Pearson correlation method
    correlation = numerator / denominator
    return correlation


def pearson_correlation_matrix(file):
    df = pd.read_csv(file)
    sort_data = df.select_dtypes(include=[int, float]).dropna()
    corr_matrix = pd.DataFrame(index=sort_data.columns,
                               columns=sort_data.columns)

    for col1 in sort_data.columns:
        for col2 in sort_data.columns:
            corr_matrix.at[col1, col2] = pearson_corr(sort_data, col1, col2)

    # Find the pair with the bether correlation (the one closest to 1/-1)
    # 1 meaning it's possitivly correlated, iin other words when a col is
    # growing the other is growing too
    # -1 meaning when a col is growing the other is decreasing

    # matrix_corr = df.corr()  # Heavy lift thanks to pandas
    un = corr_matrix.unstack()
    un = un[un < 1]
    print(f'the most similar features are: {un.idxmin()}')
    scatter_plot_pearson(un.idxmin(), df)


def scatter_plot_pearson(corr_col, df):
    house_colors = {
        "Ravenclaw": "lightblue",
        "Slytherin": "green",
        "Gryffindor": "red",
        "Hufflepuff": "yellow"
    }

    # Map the colors based on the 'Hogwarts House' column
    col = df['Hogwarts House'].map(house_colors)
    plt.scatter(df[corr_col[0]], df[corr_col[1]], c=col, alpha=0.6)
    plt.xlabel(corr_col[0])
    plt.ylabel(corr_col[1])
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w',
                        markerfacecolor=color, markersize=10, label=house)
                        for house, color in house_colors.items()],
               title="Hogwarts House")
    plt.show()


def main():
    if len(sys.argv) == 2:
        if os.path.isfile(sys.argv[1]):
            pearson_correlation_matrix(sys.argv[1])
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
