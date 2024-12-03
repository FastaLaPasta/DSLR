import matplotlib.pyplot as plt
import pandas as pd
from stats import mean, std


def disp_histogram(file):
    data = pd.read_csv(file)
    report_card = {
                'Gryffindor': [],
                'Ravenclaw': [],
                'Hufflepuff': [],
                'Slytherin': []
            }
    grades = data.select_dtypes(include=[float, int]).columns

    standardized_data = data.dropna().copy()
    for grade in grades:
        standardized_data[grade] = (data[grade] - data[grade].mean()) / data[grade].std()

    len_grade = standardized_data.dropna().groupby('Hogwarts House')
    for house, group_df in len_grade:
        for course in grades:
            values = group_df[course].dropna()
            report_card[house].append(mean(values))

    df = pd.DataFrame(report_card, index=grades)
    stddd = df.std(axis=1)
    print(stddd)

    stddd.T.sort_values().plot(kind='bar')
    plt.show()
    return 0
