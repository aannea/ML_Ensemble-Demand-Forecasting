﻿# Ensemble Demand Forecasting

## Project Overview
<p align="center">
  <img src="./Vis/ensemble_demand.png" alt="Forecasting Results">
</p>

This project focuses on daily demand forecasting for orders using an ensemble of three machine learning models: Support Vector Regression (SVR), Linear Regression, and Random Forest. The ensemble approach aims to improve the accuracy and robustness of the demand forecasts.

## Project Structure
The project is organized into the following main components:

1. `train.py`: Main script for training the models, performing k-fold cross-validation, and evaluating the ensemble model.
2. `utils/`: Directory containing utility scripts for data loading, preprocessing, model definitions, evaluation metrics, and visualization.
  - `data.py`: Functions for loading and preprocessing data.
  - `grid_search.py`: Script for performing grid search to find the best hyperparameters for the SVR model.
  - `synthetic_data.py`: Script for generating synthetic data using a Variational Autoencoder (VAE).
  - `visualization.py`: Class for visualizing the results of the model predictions.
3. `models/`: Directory containing model definitions.
  - `model.py`: Functions for training individual models (SVR, Linear Regression, Random Forest) and making ensemble predictions.

## Installation

To run this project, you need to have Python installed along with the required libraries. You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

1. **Load and preprocess data**: The data is loaded from a CSV file and preprocessed to normalize the features.
2. **Generate synthetic data**: Optionally, you can generate synthetic data using the VAE model provided in `utils/synthetic_data.py`.

### Training and Evaluation

Run the `train.py` script to train the models, perform k-fold cross-validation, and evaluate the ensemble model:

```bash
python train.py
```

### Visualization

The `Visualization` class in `utils/visualization.py` is used to plot the results of the model predictions. The plots are saved with a timestamp to ensure unique filenames.

## Results

The results of the k-fold cross-validation and the final evaluation on the test set are printed to the console. The visualizations of the predictions are saved in the `Vis/` directory.

## Project Files

- `train.py`: Main script for training and evaluation.
- `utils/`:
  - `data.py`: Data loading and preprocessing functions.
  - `grid_search.py`: Grid search for hyperparameter tuning.
  - `synthetic_data.py`: Synthetic data generation using VAE.
  - `visualization.py`: Visualization of model predictions.
- `models/`:
  - `model.py`: Model definitions and ensemble prediction function.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

This project uses the following libraries:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `torch`

Special thanks to the authors and contributors of these libraries.

## Authors
1. Bintang Rizqi Pasha
2. Farhan Aryo Pangestu
3. Sani Akhzam Prakistiyanto
