import numpy as np
import math
import torch
import torch.optim as optim
from tqdm import tqdm
from matplotlib import pyplot as plt

#Linear Couette flow
H=1
z=np.arange(0,1.1,0.1)
U=10
u=z*U/H+np.random.randn(len(z))
plt.scatter(u,z)
plt.xlabel("u velocity (m/s)")
plt.ylabel("z height (m)")
plt.show(block = False)

#Put datas into torch tensor
z=torch.tensor(z)
u=torch.tensor(u)

#Linear model
def model(x,w,b):
    return w*x+b

#Mean Square Error Loss
def loss_fn(x_p,y):
    loss_fn=(x_p-y)**2
    return loss_fn.mean()

#Taking derivative by calculus
def grad_fn(x_p,x,y):
    dloss_dw=2*(x_p-y)*x
    dloss_db=2*(x_p-y)*1.0
    return torch.stack([dloss_dw.mean(), dloss_db.mean()])

#Gradient descent
def optim(params,lr,grad):
    
    params=params-lr*grad
    return params

'''Homework!!! Changing optimizer to this code, 
Additionally, implement L2 regularization code tAo achieve intermediate level'''
#optimizer = optim.SGD([params], lr=learning_rate, weight_decay=weight_decay)

#Parameters update 
def training(x,y,params,n_epoch,lr):
    for epoch in range(1, n_epoch+1):
        w, b=params
        x_p=model(x,w,b)
        loss=loss_fn(x_p,y)
        grad=grad_fn(x_p,x,y)
        params=optim(params,lr,grad)
        '''Homework!!! Changing  optimize step to thiscode to achieve intermediate level'''
        #optimizer.zero_grad()
        #train_loss.backward()
        #optimizer.step()
        print('Epoch: %d, Loss: %f' %(epoch, loss))
    return params

''' Homework!!! You have to fill the number in this block to achieve basic grade '''
params=torch.tensor([10, 1]) # weight and bias, arbitrary number
config={
    "number epoch": 5000, # number of epoch, suggest from 100 to 50000
    "learning rate": 1e-5, # learning rate, suggest from 1e-2 to 1e-5
    }

#Training
params=training(z,u,params,config["number epoch"],config["learning rate"])

#Plot the result
w,b=params
z_p=model(z,w,b)
fig=plt.figure(dpi=100)
plt.plot(z_p.detach().numpy(),z.detach().numpy())
plt.plot(u.numpy(),z.numpy(), 'o')
plt.xlabel("u velocity (m/s)")
plt.ylabel("z height (m)")
plt.show()