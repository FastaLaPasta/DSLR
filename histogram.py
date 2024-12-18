import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from stats import mean


def disp_histogram(data, subject):
    data = data.dropna(subset=[subject, 'Hogwarts House'])

    house_colors = {
        'Gryffindor': 'red',
        'Ravenclaw': 'blue',
        'Hufflepuff': 'yellow',
        'Slytherin': 'green'
    }

    plt.figure(figsize=(10, 6))
    for house, color in house_colors.items():
        house_data = data[data['Hogwarts House'] == house][subject]
        plt.hist(house_data, bins=20, alpha=0.5, label=house, color=color)

    plt.xlabel('Standardized Values')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {subject} by House')
    plt.legend(loc='upper right')
    plt.show()


def disp_homo(data):
    report_card = {
                'Gryffindor': [],
                'Ravenclaw': [],
                'Hufflepuff': [],
                'Slytherin': []
            }
    grades = data.drop("Index", axis=1)\
        .select_dtypes(include=[float, int]).columns

    # Standardize Data do every grades are on a mean of 0,
    # with a standard deviation of 1
    stand_data = data.dropna().copy()
    for grade in grades:
        stand_data[grade] = (data[grade] - data[grade].mean()) /\
                            data[grade].std()

    len_grade = stand_data.dropna().groupby('Hogwarts House')
    for house, group_df in len_grade:
        for course in grades:
            values = group_df[course].dropna()
            report_card[house].append(mean(values))

    df = pd.DataFrame(report_card, index=grades)
    # unsing standard deviation here allows us to know what is the gap from
    # the mean for each house (remember we set the mean to 0 before)
    stddd = df.std(axis=1)
    print(stddd.T.sort_values())

    stddd.T.sort_values().plot(kind='bar')
    plt.savefig("bar_chart.png")
    plt.close()
    return stddd.idxmin()


def main():
    if len(sys.argv) == 2:
        if os.path.isfile(sys.argv[1]):
            data = pd.read_csv(sys.argv[1])
            subject = disp_homo(data)
            disp_histogram(data, subject)
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
