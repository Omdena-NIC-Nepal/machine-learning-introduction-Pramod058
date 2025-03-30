from sklearn.linear_model import LinearRegression


def create_regression_model():
    return LinearRegression()


def train_regression_model(xtrain, ytrain, whichModel):
    lin_model = whichModel.fit(xtrain, ytrain)

    return lin_model