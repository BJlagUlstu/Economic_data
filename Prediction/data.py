import numpy as np
import pandas


def load_data(path):
    dataset = pandas.read_csv(path)

    dataset = dataset.loc[(np.isnan(dataset['oil prices']) == False) & (dataset['country'] != '') &
                          (np.isnan(dataset['year']) == False)]

    dataset['country'] = dataset['country'].apply(list(set(dataset['country'])).index)
    dataset['oil prices'] = dataset['oil prices'].apply(round)

    X = dataset[['country', 'year']]
    y = dataset['oil prices']

    return X, y
