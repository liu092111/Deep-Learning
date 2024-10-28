import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt

# Linear Couette flow
H = 1
z = np.arange(0, 1.1, 0.1)
U = 10
u = z * U / H + np.random.randn(len(z))

# Plot the initial data
plt.scatter(u, z)
plt.xlabel("u velocity (m/s)")
plt.ylabel("z height (m)")
plt.show(block=False)

# Convert data to torch tensors
z = torch.tensor(z, dtype=torch.float32)
u = torch.tensor(u, dtype=torch.float32)

# Linear model
def model(x, w, b):
    return w * x + b

# Mean Squared Error Loss with L2 Regularization
def loss_fn(x_p, y, params, weight_decay):
    mse_loss = ((x_p - y) ** 2).mean()
    l2_reg = weight_decay * (params[0]**2 + params[1]**2)  # L2 regularization term
    return mse_loss + l2_reg  # Total loss

# Parameters initialization
params = torch.tensor([10.0, 1.0], requires_grad=True)  # weight and bias
learning_rate = 1e-5
weight_decay = 0.01  # L2 regularization factor

# Define optimizer
optimizer = optim.SGD([params], lr=learning_rate)

# Training function
def training(x, y, params, n_epoch, optimizer, weight_decay):
    for epoch in range(1, n_epoch + 1):
        w, b = params
        
        # Forward pass
        x_p = model(x, w, b)
        
        # Calculate loss
        loss = loss_fn(x_p, y, params, weight_decay)
        
        # Zero gradients, perform backpropagation, and update parameters
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Backpropagation to compute gradients
        optimizer.step()  # Update parameters

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print('Epoch: %d, Loss: %f' % (epoch, loss.item()))
    
    return params

# Training configuration
config = {
    "number epoch": 10000,  # number of epochs
}

# Train the model
params = training(z, u, params, config["number epoch"], optimizer, weight_decay)

# Plot the result
w, b = params
z_p = model(z, w, b)
plt.figure(dpi=100)
plt.plot(z_p.detach().numpy(), z.detach().numpy(), label='Model Prediction')
plt.plot(u.numpy(), z.numpy(), 'o', label='Observed Data')
plt.xlabel("u velocity (m/s)")
plt.ylabel("z height (m)")
plt.legend()
plt.show()
