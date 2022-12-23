import numpy as np
import pandas


def load_data(path):
    dataset = pandas.read_csv(path)

    dataset = dataset.loc[(np.isnan(dataset['oil prices']) == False) & (np.isnan(dataset['unemploymentrate']) == False)]

    dataset['oil prices'] = dataset['oil prices'].apply(round)

    return dataset[['unemploymentrate', 'oil prices']]
