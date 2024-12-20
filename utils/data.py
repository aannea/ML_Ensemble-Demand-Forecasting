import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path='dataset/Daily_Demand_Forecasting_Orders.csv', target_column="Target (Total orders)"):
    # Load dataset
    data = pd.read_csv(file_path, delimiter=',')

    # Select relevant features
    features = [
        'Week of the month (first week, second, third, fourth or fifth week',
        'Day of the week (Monday to Friday)',
        'Non-urgent order', 'Urgent order', 'Order type A', 'Order type B', 'Order type C',
        'Fiscal sector orders', 'Orders from the traffic controller sector',
        'Banking orders (1)', 'Banking orders (2)', 'Banking orders (3)'
    ]
    X = data[features]
    y = data[target_column]

    # Convert categorical features to numerical
    X = pd.get_dummies(X, drop_first=True)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Normalize Features
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    # Normalize target
    y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    return X_train, X_test, y_train, y_test

def k_fold_data(X, y, n_splits=5, random_state=42):
    """
    Perform k-fold cross-validation splitting.

    Parameters:
        X (array-like): Features matrix.
        y (array-like): Target vector.
        n_splits (int): Number of folds.
        random_state (int): Random seed for reproducibility.

    Returns:
        generator: Yields training and validation splits for each fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if isinstance(y, pd.Series):
        y = y.values

    for train_index, val_index in kf.split(X):
        # Ensure compatibility with Pandas objects
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        yield X_train, X_val, y_train, y_val