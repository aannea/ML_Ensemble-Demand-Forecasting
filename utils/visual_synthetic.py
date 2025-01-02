import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the synthetic data
synthetic_data = pd.read_csv('synthetic_data.csv')

# Plot histograms for each feature
plt.figure(figsize=(15, 10))
for i, column in enumerate(synthetic_data.columns):
    plt.subplot(4, 5, i + 1)
    sns.histplot(synthetic_data[column], kde=True)
    plt.title(f'Histogram of {column}')
plt.tight_layout()
plt.show()

# Plot pairplot for pairs of features
sns.pairplot(synthetic_data)
plt.suptitle('Pairplot of Synthetic Data', y=1.02)
plt.show()

# Plot heatmap for correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = synthetic_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Synthetic Data')
plt.show()

if __name__ == "__main__":
    print('Data Visualization done')

