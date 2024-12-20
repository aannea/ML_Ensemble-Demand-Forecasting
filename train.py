from utils.data import load_and_preprocess_data, k_fold_data
from models.model import svr, linearRegression, randomForest, ensemble_predict
from utils.eval_metrics import evaluate_model
from utils.visualization import Visualization
import numpy as np

def main():
    # File path and target column
    # file_path = 'dataset/Daily_Demand_Forecasting_Orders.csv'
    file_path = 'utils/synthetic_data.csv'
    target_column = 'Target (Total orders)'

    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, target_column)

    n_splits = 10
    kfold_results = {"train_mse": [], "train_r2": [], "val_rmse": [], "val_r2": []}
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    for fold, (X_train_fold, X_val_fold, y_train_fold, y_val_fold) in enumerate(
            k_fold_data(X_train, y_train, n_splits=n_splits)):
        print(f"Processing Fold {fold + 1}...")
        print(f"X_train_fold shape: {X_train_fold.shape}, X_val_fold shape: {X_val_fold.shape}")
        print(f"y_train_fold shape: {y_train_fold.shape}, y_val_fold shape: {y_val_fold.shape}")

        # Set hyperparameters
        C = 0.1
        kernel = 'rbf'
        gamma = 'scale'
        # {'C': 0.1, 'gamma': 'scale', 'kernel': 'rbf'}

        # Train individual models
        svr_model = svr(X_train_fold, y_train_fold, C=C, kernel=kernel, gamma=gamma)
        lr_model = linearRegression(X_train_fold, y_train_fold)
        rf_model = randomForest(X_train_fold, y_train_fold)

        # Ensemble prediction
        ensemble_models = [svr_model, lr_model, rf_model]
        y_pred_val = ensemble_predict(ensemble_models, X_val_fold)

        # Evaluate ensemble model on validation data
        val_rmse = np.sqrt(np.mean((y_val_fold - y_pred_val) ** 2))
        # print(np.mean((y_val_fold - y_pred_val) ** 2))
        val_r2 = 1 - (np.sum((y_val_fold - y_pred_val) ** 2) / np.sum((y_val_fold - np.mean(y_val_fold)) ** 2))

        # Store results
        kfold_results["val_rmse"].append(val_rmse)
        kfold_results["val_r2"].append(val_r2)

        print(f"Fold {fold + 1} Results:")
        print(f"Validation RMSE: {val_rmse}, Validation R2: {val_r2}")

        viz = Visualization(y_val_fold, y_pred_val, isTrain=True)
        viz.plot_results()

    # Average results across folds
    avg_val_rmse = np.mean(kfold_results["val_rmse"])
    avg_val_r2 = np.mean(kfold_results["val_r2"])

    print("\nK-Fold Validation Summary:")
    print(f"Average Validation RMSE: {avg_val_rmse}")
    print(f"Average Validation R2: {avg_val_r2}")

    # Final evaluation on the test set
    print("\nEvaluating on the Test Set...")
    final_svr_model = svr(X_train, y_train, C=C, kernel=kernel)
    final_lr_model = linearRegression(X_train, y_train)
    final_rf_model = randomForest(X_train, y_train)

    final_ensemble_models = [final_svr_model, final_lr_model, final_rf_model]
    y_pred_test = ensemble_predict(final_ensemble_models, X_test)

    rmse_test = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
    r2_test = 1 - (np.sum((y_test - y_pred_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

    print(f"Test MSE: {rmse_test}")
    print(f"Test R2 Score: {r2_test}")

    # Visualize results on test data
    viz = Visualization(y_test, y_pred_test, isTrain=False)
    viz.plot_results()


    # for fold, (X_train_fold, X_val_fold, y_train_fold, y_val_fold) in enumerate(
    #         k_fold_data(X_train, y_train, n_splits=n_splits)):
    #     print(f"Processing Fold {fold + 1}...")
    #     print(f"X_train_fold shape: {X_train_fold.shape}, X_val_fold shape: {X_val_fold.shape}")
    #     print(f"y_train_fold shape: {y_train_fold.shape}, y_val_fold shape: {y_val_fold.shape}")
    #
    #     # Set hyperparameters
    #     C = 0.09
    #     kernel = 'sigmoid'
    #     #
    #     # # model
    #     # model = svr(X_train_fold, y_train_fold, C=C, kernel=kernel)
    #     ensemble_models = [svr(X_train_fold, y_train_fold, C=C, kernel=kernel),
    #                        linearRegression(X_train_fold, y_train_fold),
    #                        randomForest(X_train_fold, y_train_fold)]
    #     y_pred_ensemble = ensemble_predict(ensemble_models, X_val_fold)
    #     ensemble_mse, ensemble_r2 = evaluate_model(ensemble_models[0], X_val_fold, y_val_fold)
    #
    #     # Evaluate model on training data
    #     train_mse, train_r2, y_pred_train = evaluate_model(model, X_train_fold, y_train_fold)
    #
    #     # Evaluate model on validation data
    #     val_mse, val_r2, y_pred_val = evaluate_model(model, X_val_fold, y_val_fold)
    #
    #     # Store results
    #     kfold_results["train_mse"].append(train_mse)
    #     kfold_results["train_r2"].append(train_r2)
    #     kfold_results["val_mse"].append(val_mse)
    #     kfold_results["val_r2"].append(val_r2)
    #
    #     print(f"Fold {fold + 1} Results:")
    #     print(f"Train MSE: {train_mse}, Train R2: {train_r2}")
    #     print(f"Validation MSE: {val_mse}, Validation R2: {val_r2}")
    #
    #     viz = Visualization(y_val_fold, y_pred_val)
    #     viz.plot_results()
    #
    # # Average results across folds
    # avg_train_mse = np.mean(kfold_results["train_mse"])
    # avg_train_r2 = np.mean(kfold_results["train_r2"])
    # avg_val_mse = np.mean(kfold_results["val_mse"])
    # avg_val_r2 = np.mean(kfold_results["val_r2"])
    #
    # print("\nK-Fold Validation Summary:")
    # print(f"Average Train MSE: {avg_train_mse}")
    # print(f"Average Train R2: {avg_train_r2}")
    # print(f"Average Validation MSE: {avg_val_mse}")
    # print(f"Average Validation R2: {avg_val_r2}")
    #
    # # Final evaluation on the test set
    # print("\nEvaluating on the Test Set...")
    # final_model = svr(X_train, y_train, C=C, kernel=kernel)
    # mse_test, r2_test, y_pred_test = evaluate_model(final_model, X_test, y_test)
    #
    # print(f"Test MSE: {mse_test}")
    # print(f"Test R2 Score: {r2_test}")
    #
    # # Visualize results on test data
    # viz = Visualization(y_test, y_pred_test)
    # viz.plot_results()

    # # Set hyperparameters
    # C = 0.09
    # kernel = 'sigmoid'

    # # Build and train model with hyperparameters
    # model = svr(X_train, y_train, C=C, kernel=kernel)

    # # Evaluate model
    # mse, r2, y_pred = evaluate_model(model, X_train, y_train)
    # print(y_train.shape)
    # print(y_pred.shape)
    # print(y_test.shape)
    # print(X_test.shape)
    # mse_test, r2_test, y_pred_test = evaluate_model(model, X_test, y_test)
    #
    # # Evaluation results
    # print(f'MSE: {mse}')
    # print(f'R2 Score: {r2}')
    # print(f'Predictions: {y_pred}')
    # # Evaluation results
    # print(f'MSE test: {mse_test}')
    # print(f'R2 Score test: {r2_test}')
    # print(f'Predictions test: {y_pred_test}')
    #
    # # Visualize results
    # viz = Visualization(y_train, y_pred)
    # viz.plot_results()

if __name__ == "__main__":
    main()