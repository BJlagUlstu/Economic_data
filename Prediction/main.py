from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor

from utils import train
from data import load_data

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    X, y = load_data('../Economic Data - 9 Countries (1980-2020).csv')

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlp_classifier = MLPClassifier(
        hidden_layer_sizes=(100, 100),
        activation='logistic',
        solver='lbfgs',
        alpha=1e-6,
        max_iter=2000,
    )

    mlp_regressor = MLPRegressor(
        hidden_layer_sizes=(100, 100),
        activation='logistic',
        solver='lbfgs',
        alpha=1e-6,
        max_iter=2000,
    )

    train(mlp_classifier, X_train, X_test, Y_train, Y_test)
    train(mlp_regressor, X_train, X_test, Y_train, Y_test)
