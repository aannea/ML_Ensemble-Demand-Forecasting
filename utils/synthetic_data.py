import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

file_path='../dataset/Daily_Demand_Forecasting_Orders.csv'
target_column="Target (Total orders)"
# Load the data from CSV
data = pd.read_csv(file_path, delimiter=';')

# Select relevant features
features = [
    'Week of the month (first week, second, third, fourth or fifth week',
    'Day of the week (Monday to Friday)',
    'Non-urgent order', 'Urgent order', 'Order type A', 'Order type B', 'Order type C',
    'Fiscal sector orders', 'Orders from the traffic controller sector',
    'Banking orders (1)', 'Banking orders (2)', 'Banking orders (3)', 'Target (Total orders)'
]
X = data[features]

# Normalize the data (important for VAE)
X_normalized = (X - X.mean()) / X.std()

# Convert to PyTorch tensor
X_tensor = torch.tensor(X_normalized.values, dtype=torch.float32)

# Create DataLoader
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=512):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean of latent space
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Log-variance of latent space
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.fc21(h2), self.fc22(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        h4 = torch.relu(self.fc4(h3))
        return torch.sigmoid(self.fc5(h4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KL

# Instantiate the model, optimizer, and set the number of epochs
input_dim = X_tensor.shape[1]  # Number of features
latent_dim = 10
hidden_dim = 256
vae = VAE(input_dim, latent_dim, hidden_dim)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
epochs = 10000

# Early stopping setup
patience = 100  # Epochs without improvement
best_loss = float('inf')
counter = 0

# Training the model
for epoch in range(epochs):
    vae.train()
    train_loss = 0
    for batch_idx, (data,) in enumerate(dataloader):
        recon_batch, mu, logvar = vae(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    # Early stopping check
    if train_loss < best_loss:
        best_loss = train_loss
        counter = 0
    else:
        counter += 1
    if counter >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

    scheduler.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(dataloader.dataset)}')

# After training, generate synthetic data
vae.eval()
with torch.no_grad():
    z = torch.randn(1000, latent_dim)
    synthetic_data = vae.decode(z)
    synthetic_data = synthetic_data.numpy()

synthetic_data = synthetic_data * X.std().values + X.mean().values

# The synthetic data is in 'synthetic_data' and can be saved to CSV
synthetic_df = pd.DataFrame(synthetic_data, columns=features)
synthetic_df.to_csv('synthetic_data.csv', index=False)

if __name__ == "__main__":
    print('Synthetic data generation complete!')