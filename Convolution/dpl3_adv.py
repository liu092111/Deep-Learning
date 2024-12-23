import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets 
import matplotlib.pyplot as plt
from torchvision import transforms

def count_parameters(model):
    """Count the total number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Datasets (CiFAR10)
data_path='./cifar10_datasets/'
cifar10_train=datasets.CIFAR10(data_path, train=True, download=True,transform=transforms.ToTensor())
cifar10_val=datasets.CIFAR10(data_path, train=False, download=True, transform=transforms.ToTensor())
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#Plot the Figure

plt.figure(figsize=(10,10))

plt.subplot(3,3,1)
for i in range(9):    
    img, label=cifar10_train[i]
    plt.subplot(3,3,i+1)
    #img.shape
    plt.imshow(img.permute(1,2,0))
    plt.axis('off')
    plt.title(classes[label], x=0.5, y=0.96)
plt.suptitle('Training Set')

plt.subplot(3,3,4) 
for i in range(9):    
    img, label=cifar10_val[i]
    plt.subplot(3,3,i+1)
    #img.shape
    plt.imshow(img.permute(1,2,0))
    plt.axis('off')
    plt.title(classes[label], x=0.5, y=0.96)
plt.suptitle('Validation Set')

#Normalization
imgs=torch.stack([img for img, _ in cifar10_train], dim=-1)
imgs.shape
imgs_t=imgs.view(3,-1)
imgs_t.shape
imgs_mean=imgs_t.mean(dim=1)
imgs_std=imgs_t.std(dim=1)

normalize_cifar10_train=datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(imgs_mean, imgs_std)]))
normalize_cifar10_val=datasets.CIFAR10(data_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(imgs_mean, imgs_std)]))

plt.subplot(3,3,7)
for i in range(9):
    img, label=normalize_cifar10_train[i]
    plt.subplot(3,3,i+1)
    plt.imshow(img.permute(1,2,0))
    plt.axis('off')
    plt.title(classes[label], x=0.5, y=0.96)
plt.suptitle('Normalized Training Set')

plt.subplot(3,3,9)
for i in range(9):
    img, label=normalize_cifar10_val[i]
    plt.subplot(3,3,i+1)
    plt.imshow(img.permute(1,2,0))
    plt.axis('off')
    plt.title(classes[label], x=0.5, y=0.96)
plt.suptitle('Normalized Validation Set')

#Configuration
''' Homework!!! You have to fill the number in this block to achieve basic score '''
input_dim=3072
lr=1e-3 # learning rate, suggest from 1e-3 to 1e-5
batch_size=32 # batch size, suggest 10~250
n_epoch=1000  # number of epoch, suggest from 1000 to 5000
best_loss=100000 # initial best loss, suggest large enogh like 100000
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Model (Linear)
fc_model=nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.Tanh(),
    nn.Linear(128,64),
    nn.Tanh(),
    nn.Linear(64,10))
fc_model.to(device)

#Model(Convolution)
class ConvNetWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        # Fully connected layers
        self.linear1 = nn.Linear(8*8*8, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.act3 = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        # First block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.pool1(out)
        
        # Second block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.pool2(out)
        
        # Fully connected layers
        out = out.view(-1, 8*8*8)
        out = self.linear1(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.linear2(out)
        return out

# 創建模型實例
fc_model = fc_model.to(device)
conv_model = ConvNetWithBN().to(device)

# 印出參數數量
print(f"FC Model total parameters: {count_parameters(fc_model):,}")
print(f"CNN Model with BatchNorm total parameters: {count_parameters(conv_model):,}")

#Loss Function
loss_fn=nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.SGD(conv_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

#Data Loader
train_loader=torch.utils.data.DataLoader(normalize_cifar10_train, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader=torch.utils.data.DataLoader(normalize_cifar10_val, batch_size=batch_size, shuffle=True)

#Training Loop
def training(model, n_epoch, loss_fn, optimizer, scheduler, train_loader, val_loader, best_loss):
    for epoch in range(n_epoch):
        # Training
        model.train()
        train_loss = 0.0
        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device)
            label_p = model(img)
            loss = loss_fn(label_p, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader) 
            
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for img, label in val_loader:
                img = img.to(device)
                label = label.to(device)
                label_vp = model(img)
                loss_v = loss_fn(label_vp, label)
                val_loss += loss_v.item()
                
                _, predicted = torch.max(label_vp.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                
        val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        # 更新學習率
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), './best_model_conv_bn.pt')
            print(f"Saving best model with loss: {best_loss:.4f} and accuracy: {accuracy:.2f}%")
            
        print(f"Epoch: {epoch}, Train Loss: {train_loss/len(train_loader):.4f}, "
            f"Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

#Training!
training(model=conv_model, 
        n_epoch=n_epoch, 
        loss_fn=loss_fn, 
        optimizer=optimizer, 
        scheduler=scheduler,  # 新增scheduler參數
        train_loader=train_loader, 
        val_loader=val_loader, 
        best_loss=best_loss)

# Validation
correction = 0
total_samples = 25
conv_model.load_state_dict(torch.load('./best_model_conv_s.pt'))
conv_model.eval()

plt.figure(figsize=(10, 10))
for i in range(total_samples):
    # 获取原始图像和标签
    img, label = cifar10_val[i]    
    plt.subplot(5, 5, i+1)
    plt.imshow(img.permute(1, 2, 0))
    plt.axis('off')
    
    # 获取归一化后的图像进行预测
    imgs, _ = normalize_cifar10_val[i]
    imgs = imgs.to(device)
    imgs = imgs.unsqueeze(0)
    
    with torch.no_grad():
        imgs_p = conv_model(imgs)
        _, label_p = torch.max(imgs_p, dim=1)
    
    # 设置标题：预测类别 (真实类别)
    plt.title(f'{classes[label_p[0]]} ({classes[label]})', 
            x=0.5, y=0.96,
            color='green' if label_p[0] == label else 'red',
            fontsize=8)
    
    if label_p[0] == label:
        correction += 1

# 计算准确率
correction_ratio = (correction / total_samples) * 100
print(f"Accuracy: {correction_ratio:.2f}%")

# 显示图像
plt.tight_layout()
plt.show()

