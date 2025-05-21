import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import joblib

# ====================== 配置参数 ====================== #
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 300
LEARNING_RATE = 0.005
TEST_SIZE = 0.2

RBF_LAYER_CONFIG = [
    (128, 0.2),  # 第一层：输入特征数->128
    (64, 0.1)  # 第二层：128->64
]


# (16, 0.02)  # 第二层：128->64


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def plot_scatter(y_true, y_pred, seed, mae, r2, save_dir='scatter_plots/rbnn_radius'):
    os.makedirs(save_dir, exist_ok=True)
    errors = np.abs(y_true - y_pred)
    max_error = np.max(errors)
    max_idx = np.argmax(errors)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(y_true, y_pred, c=errors, cmap='viridis', alpha=0.7,
                          edgecolors='w', s=50, vmin=0, vmax=np.percentile(errors, 95))

    # 理想线
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', lw=2, alpha=0.7)

    # 最大误差点
    plt.scatter(y_true[max_idx], y_pred[max_idx],
                edgecolors='red', facecolors='none', s=150,
                linewidths=2, label=f'Max Error: {max_error:.2f}')

    # 统计信息
    stats_text = (f'Seed: {seed}\nMAE: {mae:.2f}\nR²: {r2:.2f}\nMax Error: {max_error:.2f}')
    plt.text(0.05, 0.85, stats_text, transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Absolute Error', rotation=270, labelpad=15)
    count = (errors > 0.3).sum()
    # 标签和标题
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    count = (errors > 0.3).sum()
    plt.title(f'Model Predictions (Seed: {seed})\n count:{count},precent={count / len(errors):.2f}', fontsize=14)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'scatter_seed_{seed}.png'), dpi=150)
    plt.close()


# ====================== 模型定义 ====================== #
class RBFLayer(nn.Module):
    def __init__(self, in_features, num_centers, gamma_init=0.1):
        super().__init__()
        self.in_features = in_features
        self.num_centers = num_centers
        self.shortcut = nn.Linear(in_features, num_centers) if in_features != num_centers else nn.Identity()
        self.gamma = nn.Parameter(torch.tensor(gamma_init))
        self.centers = nn.Parameter(torch.Tensor(num_centers, in_features))
        nn.init.xavier_normal_(self.centers)

    def forward(self, x):
        identity = self.shortcut(x)
        x = x.unsqueeze(1)
        c = self.centers.unsqueeze(0)
        distances = torch.norm(x - c, p=2, dim=2)
        return torch.exp(-self.gamma * distances ** 2) + identity


class RBFNN(nn.Module):
    def __init__(self, in_features, layer_config):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = in_features
        for units, gamma in layer_config:
            self.layers.extend([
                RBFLayer(prev_dim, units, gamma),
                nn.BatchNorm1d(units),
                nn.Dropout(0.3)
            ])
            prev_dim = units
        self.output = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


# ====================== 训练流程 ====================== #
sum_mae, sum_r2 = 0, 0

for seed in range(100):
    print("-" * 63)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载和预处理
    df = pd.read_excel("核桃仁表型信息_重新标定.xlsx", sheet_name="Sheet1")
    FEATURES = ['hutao_area', 'hutao_perimeter', 'hutao_area/hutao_perimeter',
                'hutao_a', 'hutao_b', 'arithmetic_a_b_h_avg', 'geometry_a_b_h_avg',
                'hutao_SI', 'hutao_ET', 'hutao_EV', 'fai', 'arithmetic_a_b_avg',
                'geometry_a_b_avg']

    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])
    y = df['g'].values.reshape(-1, 1).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)

    # 数据加载器
    train_loader = DataLoader(CustomDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(CustomDataset(X_test, y_test),
                             batch_size=BATCH_SIZE, shuffle=False)

    # 模型和优化器
    model = RBFNN(X.shape[1], RBF_LAYER_CONFIG).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    # --------------------- 训练循环 ---------------------
    best_val_loss = float('inf')
    for epoch in range(300):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.MSELoss()(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        predictions = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs.to(device))
                val_loss += nn.MSELoss()(outputs, targets.to(device)).item()
                predictions.append(outputs.cpu().numpy())

        # 学习率调度
        scheduler.step(val_loss)

        # 每20个epoch打印进度
        if (epoch + 1) % 20 == 0:
            y_pred = np.concatenate(predictions).flatten()
            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] '
                  f'Train Loss: {train_loss / len(train_loader):.4f} '
                  f'Val Loss: {val_loss / len(test_loader):.4f}')

        # 最终评估
        y_pred, y_true = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs.to(device))
                y_pred.extend(outputs.cpu().numpy().flatten().tolist())
                y_true.extend(targets.numpy().flatten().tolist())
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = r2_score(y_true, y_pred)

        # 绘制散点图
        plot_scatter(np.array(y_true), np.array(y_pred),
                     seed=seed, mae=mae, r2=r2)

        sum_mae += mae
        sum_r2 += r2

print(f'\nAverage MAE: {sum_mae / 100:.2f}  R²: {sum_r2 / 100:.2f}')
