import numpy as np
import pandas


SMALL = 1
MEDIUM = 2
LARGE = 3


def load_data2(path):
    dataset = pandas.read_csv(path)

    dataset = dataset.loc[(np.isnan(dataset['oil prices']) == False) & (dataset['country'] != '') &
                          (np.isnan(dataset['year']) == False)]

    dataset['country'] = dataset['country'].apply(list(set(dataset['country'])).index)
    dataset['oil prices'] = dataset['oil prices'].apply(round)

    X = dataset[['country', 'year']]
    y = dataset['oil prices']

    return X, y


def load_data(path):
    dataset = pandas.read_csv(path)

    dataset = dataset.loc[(np.isnan(dataset['oil prices']) == False) & (np.isnan(dataset['unemploymentrate']) == False) & (np.isnan(dataset['inflationrate']) == False)]

    def calculate_unemployment_rate_degree(value):
        if 0.02 <= value <= 0.04:
            return SMALL
        if 0.05 <= value <= 0.06:
            return MEDIUM
        if value >= 0.07:
            return LARGE

    def calculate_inflation_rate_degree(value):
        if value <= 0.06:
            return SMALL
        if 0.07 <= value <= 0.1:
            return MEDIUM
        if value >= 0.11:
            return LARGE

    dataset['oil prices'] = dataset['oil prices'].apply(round)
    dataset['unemploymentrate'] = dataset['unemploymentrate'].apply(calculate_unemployment_rate_degree)
    dataset['inflationrate'] = dataset['inflationrate'].apply(calculate_inflation_rate_degree)

    X = dataset[['unemploymentrate', 'inflationrate']]
    y = dataset['oil prices']

    return X, y
