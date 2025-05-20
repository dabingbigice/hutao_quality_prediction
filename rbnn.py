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
from torch.utils.data import Dataset, DataLoader

# ====================== 配置参数修改 ====================== #
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 300  # 增加训练轮次
LEARNING_RATE = 0.005  # 调整学习率
TEST_SIZE = 0.2

# 定义双层RBF参数 [(第一层参数), (第二层参数)]
RBF_LAYER_CONFIG = [
    (128, 0.2),
    # 第一层：128个中心，gamma=0.2
    # (64, 0.1)  # 第二层：64个中心，gamma=0.1
]
# ======================================================== #
for RANDOM_SEED in range(100):
    # 设置随机种子
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------- 数据预处理 ---------------------
    df = pd.read_excel("核桃仁表型信息_重新标定.xlsx", sheet_name="Sheet1")
    FEATURES = ['hutao_area', 'hutao_perimeter', 'hutao_area/hutao_perimeter',
                'hutao_a', 'hutao_b', 'arithmetic_a_b_h_avg', 'geometry_a_b_h_avg',
                'hutao_SI', 'hutao_ET', 'hutao_EV', 'fai', 'arithmetic_a_b_avg',
                'geometry_a_b_avg']
    features = df[FEATURES]
    target = df['g'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    y = target.reshape(-1, 1).astype(np.float32)

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )


    # --------------------- 数据集定义 ---------------------
    class CustomDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    # --------------------- 模型定义（已修改为双层）---------------------
    class RBFLayer(nn.Module):
        def __init__(self, in_features, num_centers, gamma):
            super(RBFLayer, self).__init__()
            self.num_centers = num_centers
            self.gamma = gamma
            self.centers = nn.Parameter(torch.Tensor(num_centers, in_features))
            nn.init.normal_(self.centers, 0, 1)

        def forward(self, x):
            x = x.unsqueeze(1)
            c = self.centers.unsqueeze(0)
            distances = torch.norm(x - c, dim=2)
            return torch.exp(-self.gamma * (distances ** 2))


    class RBFNN(nn.Module):
        def __init__(self, in_features, rbf_layers_config):
            super(RBFNN, self).__init__()
            self.rbf_layers = nn.ModuleList()
            prev_units = in_features

            # 构建多层RBF
            for num_centers, gamma in rbf_layers_config:
                self.rbf_layers.append(RBFLayer(prev_units, num_centers, gamma))
                prev_units = num_centers

            # 输出层
            self.linear = nn.Linear(prev_units, 1)

            # 添加Dropout层（可选）
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            for layer in self.rbf_layers:
                x = layer(x)
            x = self.dropout(x)  # 应用Dropout
            return self.linear(x)


    # ====================== 模型初始化修改 ====================== #
    model = RBFNN(
        in_features=X_train.shape[1],
        rbf_layers_config=RBF_LAYER_CONFIG
    ).to(device)
    # ======================================================== #

    # --------------------- 训练配置 ---------------------
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)


    # --------------------- 训练循环 ---------------------
    def plot_predictions(epoch, y_true, y_pred, save_dir='plots'):
        os.makedirs(save_dir + f'_{RANDOM_SEED}', exist_ok=True)
        errors = np.abs(y_true - y_pred)
        count = (errors > 0.3).sum()
        plt.figure(figsize=(10, 7))
        plt.scatter(y_true, y_pred, c=errors, cmap='plasma', alpha=0.7, edgecolors='w', s=60)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)

        # 添加统计信息
        stats_text = (f'MAE: {np.mean(errors):.2f}\n'
                      f'Max Error: {np.max(errors):.2f}\n'
                      f'R²: {r2_score(y_true, y_pred):.2f}')
        plt.gcf().text(0.15, 0.85, stats_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

        plt.colorbar(label='Absolute Error')
        plt.xlabel('Actual Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.title(
            f'RANDOM_SEED ={RANDOM_SEED}\nEpoch {epoch + 1} Predictions,\n errors > 0.3_%={count / len(errors):.2f}',
            fontsize=14)
        plt.savefig(f'{save_dir}/epoch_{epoch + 1}.png', dpi=150, bbox_inches='tight')
        plt.close()


    train_losses, val_losses = [], []

    for epoch in range(NUM_EPOCHS):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            epoch_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        predictions = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                predictions.append(outputs.cpu().numpy())

        # 学习率调度
        scheduler.step(val_loss)

        # 记录损失
        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss / len(test_loader))

        # 每20个epoch可视化
        if (epoch + 1) % 20 == 0:
            y_pred = np.concatenate(predictions).flatten()
            plot_predictions(epoch, y_test.flatten(), y_pred)
            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] '
                  f'Train Loss: {train_losses[-1]:.4f} '
                  f'Val Loss: {val_losses[-1]:.4f} '
                  f'LR: {optimizer.param_groups[0]["lr"]:.2e}')


    # --------------------- 最终评估 ---------------------
    def evaluate(model, loader):
        model.eval()
        total_mae, total_r2 = 0.0, 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                total_mae += torch.abs(outputs - targets).sum().item()
                total_r2 += r2_score(targets.cpu(), outputs.cpu()) * inputs.size(0)
        return total_mae / len(loader.dataset), total_r2 / len(loader.dataset)


    test_mae, test_r2 = evaluate(model, test_loader)
    print(f'\nFinal Performance: MAE={test_mae:.2f}  R²={test_r2:.2f}')

    # 保存完整模型
    torch.save({
        'model': model.state_dict(),
        'scaler': scaler,
        'config': RBF_LAYER_CONFIG,
        'features': FEATURES
    }, 'dual_rbf_model.pth')

    # 可视化训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig('training_curves.png', bbox_inches='tight')
