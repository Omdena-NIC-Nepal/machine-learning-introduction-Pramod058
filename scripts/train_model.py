from sklearn.linear_model import LinearRegression


def create_regression_model():
    # initiate linear regression model
    return LinearRegression()


def train_regression_model(xtrain, ytrain, whichModel):
    lin_model = whichModel.fit(xtrain, ytrain)

    return lin_model

def y_pred(xtest, whichModel):
    y_pred = whichModel.predict(xtest)
    return y_pred