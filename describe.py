from stats import ft_statistics
import pandas as pd


def treat_data(dataset):
    data = pd.read_csv(dataset)
    summary = {}

    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]) and data[col].count() != 0:
            column_data = data[col].dropna()

            count = len(column_data.dropna())
            mean, median, std, quartile, minimum, maximum = ft_statistics(
                                                        column_data,
                                                        mean='mean',
                                                        median='median',
                                                        std='std',
                                                        quartile='quartile',
                                                        minimum='min',
                                                        maximum='max')

            summary[col] = {
                'count': count,
                'mean': mean,
                'std': std,
                'min': minimum,
                '25%': quartile[0],
                '50%': median,
                '75%': quartile[1],
                'max': maximum
            }

    summary_df = pd.DataFrame(summary)
    return summary_df.transpose()
