from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler


def train(model, x_train, x_test, y_train, y_test):
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    score = model.score(x_test, y_test)
    mape = mean_absolute_percentage_error(y_test, y_predict)

    print(f'\n{type(model).__name__}\nScore: {score}\nMAPE: {mape}')
