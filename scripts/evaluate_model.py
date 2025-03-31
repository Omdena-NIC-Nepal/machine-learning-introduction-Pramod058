from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def meanSquaredError(a, b):
    return mean_squared_error(a, b)

def meanAbsoluteError(a, b):
    return mean_absolute_error(a, b)

def r2Score(a, b):
    return r2_score(a, b)


def scatterplot(Y, ytest, ypred, alpha):
    plt.scatter(ytest, ypred, alpha=0.5)
    plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color= "red")
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title("Predicted vs Actual values")

    return plt

def residualplot(ypred, Hey, alpha):
    plt.scatter(ypred, Hey, alpha=alpha)
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot (Linear Regression)")
    return plt
