import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt

# Linear Couette flow
H = 1
z = np.arange(0, 1.1, 0.1)
U = 10
u = z * U / H + np.random.randn(len(z))

# Convert to PyTorch tensors
z = torch.tensor(z, dtype=torch.float32)
u = torch.tensor(u, dtype=torch.float32)

# Linear model
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x.unsqueeze(1)).squeeze(1)

# Mean Square Error Loss
def loss_fn(x_p, y):
    return torch.mean((x_p - y) ** 2)

# Split data into training and validation sets
def split_data(z, u, validate_ratio=0.2):
    num = len(z)
    shuffle_ind = torch.randperm(num)
    val_size = int(num * validate_ratio)
    val_indices = shuffle_ind[:val_size]
    train_indices = shuffle_ind[val_size:]
    
    z_train, u_train = z[train_indices], u[train_indices]
    z_val, u_val = z[val_indices], u[val_indices]
    
    return (z_train, u_train), (z_val, u_val)

# Training function with early stopping
def training(model, train_data, val_data, optimizer, n_epoch, weight_decay, patience=5):
    z_train, u_train = train_data
    z_val, u_val = val_data
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, n_epoch + 1):
        model.train()
        optimizer.zero_grad()
        x_p_train = model(z_train)
        train_loss = loss_fn(x_p_train, u_train)
        
        # Add L2 regularization
        l2_lambda = weight_decay
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        train_loss = train_loss + l2_lambda * l2_norm
        
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            x_p_val = model(z_val)
            val_loss = loss_fn(x_p_val, u_val)
        
        if epoch % 100 == 0:
            print(f'Epoch: {epoch}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
    
    return model

# Initialize model and optimizer
model = LinearModel()
params = {
    "number_epoch": 5000,
    "learning_rate": 1e-2,
    "weight_decay": 1e-5  # L2 regularization strength
}

optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])

# Split data
train_data, val_data = split_data(z, u, validate_ratio=0.2)

# Training
trained_model = training(model, train_data, val_data, optimizer, params["number_epoch"], params["weight_decay"])

# Plot the result
model.eval()
with torch.no_grad():
    z_p = trained_model(z).numpy()

plt.figure(figsize=(10, 6), dpi=100)
plt.plot(z_p, z.numpy(), label='Predicted')
plt.plot(u.numpy(), z.numpy(), 'o', label='Original Data')
plt.xlabel("u velocity (m/s)")
plt.ylabel("z height (m)")
plt.legend()
plt.title("Couette Flow Regression")
plt.show()

# Print final parameters
for name, param in trained_model.named_parameters():
    print(f"{name}: {param.data}")