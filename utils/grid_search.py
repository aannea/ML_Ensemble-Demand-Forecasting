from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from utils.data import load_and_preprocess_data

# File path and target column
file_path = 'synthetic_data.csv'
target_column = 'Target (Total orders)'

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, target_column)

# Define the parameter grid
param_grid = {
    'C': [0.01, 0.1, 0.9, 0.8, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

# Create the SVR model
svr_model = SVR()

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(svr_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"Best Parameters: {best_params}")

if __name__ == "__main__":
    print('Grid Search done')