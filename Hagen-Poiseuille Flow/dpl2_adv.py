import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Dataset generation (這部分保持不變)
R = 1
r = np.arange(-1, 1.01, 0.01)
G = 1
mu = 0.001
u = G / (4 * mu) * (R**2 - r**2) + 10 * np.random.randn(len(r))

r = torch.tensor(r, dtype=torch.float32)
u = torch.tensor(u, dtype=torch.float32)

# 資料集分割 (這部分保持不變)
val_ratio = 0.2
n_sample = len(r)
shuffle_ind = torch.randperm(len(r))
train_ind = shuffle_ind[:-int(val_ratio * n_sample)]
val_ind = shuffle_ind[-int(val_ratio * n_sample):]
train_set_r = r[train_ind].unsqueeze(1)
train_set_u = u[train_ind].unsqueeze(1)
val_set_r = r[val_ind].unsqueeze(1)
val_set_u = u[val_ind].unsqueeze(1)

# 定義不同的模型
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.fc(x)

class WiderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.fc(x)

class DeeperModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.fc(x)

class ReLUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.fc(x)

# 訓練函數
def train_model(model, train_set_r, train_set_u, val_set_r, val_set_u, n_epoch=10000, lr=1e-4, model_name="model"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_loss = float('inf')
    
    for epoch in range(1, n_epoch + 1):
        u_p = model(train_set_r)
        train_loss = loss_fn(u_p, train_set_u)
        
        u_v = model(val_set_r)
        val_loss = loss_fn(u_v, val_set_u)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        if val_loss < best_loss:
            best_loss = val_loss.item()
            # 儲存模型參數
            torch.save(model.state_dict(), f"{model_name}_best.ckpt")
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss.item():.4f}, "
                  f"Validation loss {val_loss.item():.4f}")
    
    return best_loss

# 訓練和比較模型
models = {
    "Base Model": BaseModel(),
    "Wider Model": WiderModel(),
    "Deeper Model": DeeperModel(),
    "ReLU Model": ReLUModel()
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}:")
    best_loss = train_model(model, train_set_r, train_set_u, val_set_r, val_set_u, model_name=name.replace(" ", "_"))
    results[name] = best_loss
    print(f"Best validation loss for {name}: {best_loss:.4f}")

# 繪製結果比較圖
plt.figure(figsize=(12, 6))
plt.bar(list(results.keys()), list(results.values()))
plt.title("Model Comparison - Best Validation Loss")
plt.ylabel("Validation Loss")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 找出最佳模型並加載其參數
best_model_name = min(results, key=results.get)
best_model = models[best_model_name]
best_model.load_state_dict(torch.load(f"{best_model_name.replace(' ', '_')}_best.ckpt"))

# 繪製最佳模型的預測結果
plt.figure(figsize=(12, 5))

# 原始數據圖
plt.subplot(1, 2, 1)
plt.scatter(u.detach().numpy(), r.detach().numpy(), alpha=0.5)
plt.title("Initial Data Scatter Plot")
plt.xlabel("u velocity (m/s)")
plt.ylabel("r radius (m)")

# 模型預測圖
plt.subplot(1, 2, 2)
r_plot = torch.linspace(-1, 1, 200).unsqueeze(1)
with torch.no_grad():  # 禁用梯度計算
    u_pred = best_model(r_plot)

plt.scatter(u.detach().numpy(), r.detach().numpy(), alpha=0.5, label='Original Data')
plt.plot(u_pred.numpy(), r_plot.numpy(), 'r-', label='Model Prediction')
plt.title(f"Hagen-Poiseuille Flow Regression\n(Best Model: {best_model_name})")
plt.xlabel("u velocity (m/s)")
plt.ylabel("r radius (m)")
plt.legend()

plt.tight_layout()
plt.show()

print("\nObservations:")
print("1. Wider Model: Increased model capacity, potentially allowing for better feature extraction.")
print("2. Deeper Model: Increased depth allows for more complex function approximation, but may be harder to train.")
print("3. ReLU Model: ReLU activation can help with vanishing gradient problem, but may lead to 'dying ReLU' issue.")
print("\nThe best model's parameters have been loaded from the checkpoint file for the final prediction plot.")