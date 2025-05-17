import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os

# 创建保存权重的目录
os.makedirs('weights', exist_ok=True)

# 1. 数据准备 --------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据
df = pd.read_excel('核桃仁表型信息.xlsx', sheet_name='Sheet1')
X = df.drop(columns=['g']).values
y = df['g'].values.reshape(-1, 1)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量并发送到CUDA
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.FloatTensor(y_test).to(device)

# 创建DataLoader
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)


# 2. 模型定义 --------------------------------------------------------
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return x


model = Net(X_train.shape[1]).to(device)

# 3. 训练配置 --------------------------------------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)  # L2正则化
epochs = 300

# 记录训练过程
train_losses = []
test_losses = []
r2_scores = []
rmse_scores = []

# 4. 训练循环 --------------------------------------------------------
for epoch in range(epochs):
    # 训练模式
    model.train()
    epoch_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # 评估模式
    model.eval()
    with torch.no_grad():
        # 训练集评估
        train_pred = model(X_train_t)
        train_loss = criterion(train_pred, y_train_t)

        # 测试集评估
        test_pred = model(X_test_t)
        test_loss = criterion(test_pred, y_test_t)

        # 计算指标
        r2 = r2_score(y_test, test_pred.cpu().numpy())
        rmse = torch.sqrt(test_loss).item()
    if epoch % 5 == 0:
        # 保存权重
        torch.save(model.state_dict(), f'weights/epoch_{epoch + 1}.pth')

    # 记录指标
    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())
    r2_scores.append(r2)
    rmse_scores.append(rmse)

    print(f'Epoch {epoch + 1}/{epochs}')
    print(f'Train Loss: {train_loss.item():.4f} | Test Loss: {test_loss.item():.4f}')
    print(f'R2: {r2:.4f} | RMSE: {rmse:.4f}\n')

# 5. 结果可视化 --------------------------------------------------------
# 损失曲线
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()

# 最终预测散点图
with torch.no_grad():
    final_pred = model(X_test_t).cpu().numpy()

plt.figure(figsize=(8, 8))
plt.scatter(y_test, final_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title(f'Final Prediction (R2={r2_scores[-1]:.4f}, RMSE={rmse_scores[-1]:.4f})')
plt.show()
