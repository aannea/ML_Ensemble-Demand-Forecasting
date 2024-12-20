from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np

def svr(X_train, y_train, C=1.0, kernel='rbf', gamma='scale'):
    # Create SVM model with hyperparameters
    model = SVR(C=C, kernel=kernel, gamma=gamma)
    model.fit(X_train, y_train)
    return model

def linearRegression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def randomForest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model

# def MLP(X_train, y_train, hidden=125, epoch=500):
#     model = MLPRegressor(hidden_layer_sizes=hidden, max_iter=epoch)
#     model.fit(X_train, y_train)
#     return model

def ensemble_predict(models, X):
    predictions = np.column_stack([model.predict(X) for model in models])
    return np.mean(predictions, axis=1)