import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils.data import load_and_preprocess_data

file_path = 'synthetic_data.csv'
target_column = 'Target (Total orders)'

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, target_column)

# Convert to DataFrame for easier plotting
features = [
        'Week of the month',
        'Day of the week',
        'Non-urgent order', 'Urgent order', 'Order type A', 'Order type B', 'Order type C',
        'Fiscal sector orders', 'Orders from the TCS',
        'Banking orders (1)', 'Banking orders (2)', 'Banking orders (3)'
    ]
X_train_df = pd.DataFrame(X_train, columns=[f'{features[i]}' for i in range(X_train.shape[1])])
y_train_df = pd.DataFrame(y_train, columns=['Target'])

# Plot histograms for features
plt.figure(figsize=(15, 10))
for i, column in enumerate(X_train_df.columns):
    plt.subplot(4, 5, i + 1)
    sns.histplot(X_train_df[column], kde=True)
    plt.title(f'Histogram of {column}')
plt.tight_layout()
plt.savefig('../Vis/Distribution/histograms_data.png')
plt.show()

# Plot box plots for features
plt.figure(figsize=(15, 10))
sns.boxplot(data=X_train_df)
plt.title('Box Plot of Features')
plt.xticks(rotation=90)
plt.savefig('../Vis/Distribution/boxplot_data.png')
plt.show()

# Plot histogram for target
plt.figure(figsize=(6, 4))
sns.histplot(y_train_df['Target'], kde=True)
plt.title('Histogram of Target')
plt.savefig('../Vis/Distribution/histogram_target.png')
plt.show()

# Plot box plot for target
plt.figure(figsize=(6, 4))
sns.boxplot(y=y_train_df['Target'])
plt.title('Box Plot of Target')
plt.savefig('../Vis/Distribution/boxplot_target.png')
plt.show()

if __name__ == "__main__":
    print('Data Distribution Visualization done')