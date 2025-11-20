import numpy as np
import deepxde as dde
import torch
from mpl_toolkits.mplot3d import Axes3D
dde.backend.set_default_backend('pytorch')
import matplotlib.pyplot as plt

#matplotlib.get_backend()

# Setting PDE
c=1
def pde(x,y):
    d2y_dx2=dde.grad.hessian(y,x,i=0,j=0)
    d2y_dt2=dde.grad.hessian(y,x,i=1,j=1)
    return d2y_dx2-c**2*d2y_dt2

# Setting Exact Solution
A=2
def func(x):
    x, t=np.split(x,2,axis=1)
    return np.sin(np.pi*x)*np.cos(c*np.pi*t)+np.sin(A*np.pi*x)*np.cos(A*c*np.pi*t)

# Defining Domain
geom=dde.geometry.Interval(0,1)
timedomain=dde.geometry.TimeDomain(0,1)
geomtime=dde.geometry.GeometryXTime(geom,timedomain)

# Setting Boundary Conditions and Initial Conditions
bc=dde.icbc.DirichletBC(geomtime, func, lambda _,on_boundary: on_boundary)
ic_1=dde.icbc.IC(geomtime, func, lambda _, on_boundary: on_boundary)
ic_2=dde.icbc.OperatorBC(geomtime, lambda x, y, _ : dde.grad.jacobian(y, x, i=0, j=1), 
                        lambda x,_ : dde.utils.isclose(x[1],0))

#Construct PDE data
'''Homework!!! Setting the point of training'''
data=dde.data.TimePDE(geomtime, pde, [bc, ic_1, ic_2], num_domain=1000, num_boundary=800, num_initial=200, solution=func, num_test=1000)
#All the value are suggest range from 100~1000

# Construct Neural Network
'''Homework!!! Setting layer size'''
'''Homework!!! Adding hidden layer and different width of layer'''
layer_size=[2]+[32]*3+[1] #first XXXX is the width of neural, second XX is layer of hidden layer 
activation='tanh'
initializer='Glorot uniform'
net=dde.nn.FNN(layer_size, activation, initializer)

# Construct Model
'''Homework!!! Setting learning rate'''
model=dde.Model(data,net)
model.compile('adam', lr=1e-4) #suggest range 1e-4~1e-2

#Training Model
'''Homework!!! Setting iteration(number of training) '''
losshistory, train_state=model.train(iterations=5000) #suggest range 1000~50000

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
plt.xlabel('x')
plt.ylabel('t')
plt.show()

#x=np.linspace(0,1,10)
#x=x[:,np.newaxis]
#y=np.linspace(0,1,10)
#y=y[:,np.newaxis]
#x1=np.concatenate((x,y), axis=1)
#%matplotlib notebook
x, t=np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
Z=np.sin(np.pi*x)*np.cos(c*np.pi*t)+np.sin(A*np.pi*x)*np.cos(A*c*np.pi*t)
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.plot_surface(x,t,Z, rstride=2, cstride=2, cmap=plt.get_cmap('coolwarm'), edgecolor='gray', linewidth=0.5)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('E')
ax.set_title('PINN solution')
#ax.colorbar()
plt.show()
