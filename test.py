from utils.data import load_and_preprocess_data
from models.model import svr
from utils.eval_metrics import evaluate_model
from utils.visualization import Visualization

def main():
    # File path and target column
    file_path = 'dataset/Daily_Demand_Forecasting_Orders.csv'
    target_column = 'Target (Total orders)'

    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, target_column)

    # Set hyperparameters
    C = 0.099
    # {'linear', 'precomputed', 'sigmoid', 'rbf', 'poly'}
    kernel = 'sigmoid'

    # Build and train model with hyperparameters
    model = svr(X_train, y_train, C=C, kernel=kernel)

    # Evaluate model
    mse, r2, y_pred = evaluate_model(model, X_test, y_test)

    # Evaluation results
    print(f'MSE: {mse}')
    print(f'R2 Score: {r2}')
    print(f'Predictions: {y_pred}')

    # Visualize results
    viz = Visualization(y_test, y_pred)
    viz.plot_results()

if __name__ == "__main__":
    main()