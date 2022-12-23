from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from data import load_data
from utils import train

if __name__ == '__main__':
    X, y = load_data('../Economic Data - 9 Countries (1980-2020).csv')

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train(DecisionTreeClassifier(), X_train, X_test, Y_train, Y_test)
    train(KNeighborsClassifier(), X_train, X_test, Y_train, Y_test)
