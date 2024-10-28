import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from matplotlib import pyplot as plt

#dataset
R=1
r=np.arange(-1,1.01,0.01)
G=1
mu=0.001
u=G/(4*mu)*(R**2-r**2)+10*np.random.randn(len(r))

plt.figure(figsize=(10, 6))
plt.scatter(u,r)
plt.title("Initial Data Scatter Plot")
plt.xlabel("u velocity (m/s)")
plt.ylabel("r radius (m)")
plt.show()

r=torch.tensor(r)
u=torch.tensor(u)
len(r)

#split the dataset
val_ratio=0.2
n_sample=len(r)
shuffle_ind=torch.randperm(len(r))
train_ind=shuffle_ind[:-int(val_ratio*n_sample)]
val_ind=shuffle_ind[-int(val_ratio*n_sample):]
train_set_r=r[train_ind]
train_set_r=r[train_ind].unsqueeze(1)
train_set_u=u[train_ind].unsqueeze(1)
val_set_r=r[val_ind].unsqueeze(1)
val_set_u=u[val_ind].unsqueeze(1)

train_set_r=train_set_r.to(torch.float32)
train_set_u=train_set_u.to(torch.float32)
val_set_r=val_set_r.to(torch.float32)
val_set_u=val_set_u.to(torch.float32)

#Model
#Fully connective nueral netwoek
''' Homework!!! You have to fill the number in this block to achieve basic score '''
#Attention! XX1 in first layer is identical in second layer, so as XX2 XX3 XX4
#suggest  all XX range  from 1~600, and gradually diverge then converge
fc_model=nn.Sequential(
        nn.Linear(1,64),
        nn.Tanh(),
        nn.Linear(64,128),
        nn.Tanh(),

#'''Homework!!! You can add more layer to let the model deeper, 
#withder or changing activative fiunction to achieve intermediate level'''
        nn.Linear(128,256),
        nn.Tanh(),
        nn.Linear(256,128),
        nn.Tanh(),
        nn.Linear(128,1))

#Loss Function
loss_fn=nn.MSELoss()

#Configuration
''' Homework!!! You have to fill the number in this block to achieve basic grade '''
lr=1e-4 # learning rate, suggest from 1e-3 to 1e-5
n_epoch=10000 # number of epoch, suggest from 10000 to 100000
best_loss=1e5 #initial best loss, suggest large enogh like 100000

#Optimizer
optimizer=optim.SGD(fc_model.parameters(),lr)

# Training loop
def train(z_train, u_train, z_val, u_val, n_epoch, model, optimizer, loss_fn, best_loss):
    for epoch in range(1, n_epoch + 1):
        u_p = model(z_train)
        train_loss = loss_fn(u_p, u_train)
        
        u_v = model(z_val)
        val_loss = loss_fn(u_v, u_val)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        if val_loss < best_loss:
            best_loss = val_loss
            print(f"New best loss: {best_loss:.4f}")
        
        if epoch % 500 == 0:  # Print every 100 epochs
            print(f"Epoch {epoch}, Training loss {train_loss.item():.4f}, "
                f"Validation loss {val_loss.item():.4f}")

# Training!
train(train_set_r, train_set_u, val_set_r, val_set_u, n_epoch, fc_model, optimizer, loss_fn, best_loss)

# Plotting
plt.figure(figsize=(12, 5))

# Original data plot
plt.subplot(1, 2, 1)
plt.scatter(u.numpy(), r.numpy(), alpha=0.5)
plt.title("Initial Data Scalar Plot")
plt.xlabel("u velocity (m/s)")
plt.ylabel("r radius (m)")

# Model prediction plot
plt.subplot(1, 2, 2)
r_plot = torch.linspace(-1, 1, 200).unsqueeze(1)
u_pred = fc_model(r_plot).detach()

plt.scatter(u.numpy(), r.numpy(), alpha=0.5, label='Original Data')
plt.plot(u_pred.numpy(), r_plot.numpy(), 'r-', label='Model Prediction')
plt.title("Hagen-Poiseuille Flow Regression")
plt.xlabel("u velocity (m/s)")
plt.ylabel("r radius (m)")
plt.legend()

plt.tight_layout()
plt.show()