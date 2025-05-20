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

# ====================== 配置参数 ====================== #
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 200
LEARNING_RATE = 0.005
TEST_SIZE = 0.2

RBF_LAYER_CONFIG = [
    (128, 0.2),
]


# ====================== 自定义数据集 ====================== #
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ====================== 改进的可视化函数 ====================== #
def plot_scatter(y_true, y_pred, seed, mae, r2, save_dir='rbf_scatters'):
    os.makedirs(save_dir, exist_ok=True)

    # 转换为NumPy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    errors = np.abs(y_true - y_pred)
    max_error = np.max(errors)
    max_idx = np.argmax(errors)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(y_true, y_pred,
                          c=errors,
                          cmap='viridis',
                          alpha=0.7,
                          edgecolors='w',
                          s=60,
                          vmin=0,
                          vmax=np.percentile(errors, 95))

    # 理想参考线
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', lw=2, alpha=0.7)

    # 标注最大误差点
    plt.scatter(y_true[max_idx], y_pred[max_idx],
                edgecolors='red',
                facecolors='none',
                s=150,
                linewidths=2,
                label=f'Max Error: {max_error:.2f}')

    # 统计信息框
    stats_text = (f'Seed: {seed}\n'
                  f'MAE: {mae:.2f}\n'
                  f'R²: {r2:.2f}\n'
                  f'Max Error: {max_error:.2f}')
    plt.text(0.05, 0.85, stats_text,
             transform=plt.gca().transAxes,
             fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))

    # 颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Absolute Error', rotation=270, labelpad=15)
    count = (errors > 0.3).sum()

    # 标签和标题
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'RBF Network Predictions (Seed: {seed})\n count:{count} | precent={count / len(errors):.2f}',
              fontsize=14)
    plt.grid(True, alpha=0.3, linestyle=':')

    # 保存文件
    plt.savefig(
        os.path.join(save_dir, f'seed_{seed}_R2_{r2:.2f}_MAE_{mae:.2f}.png'),
        dpi=150,
        bbox_inches='tight'
    )
    plt.show()
    plt.close()


# ====================== RBF模型定义 ====================== #
class RBFLayer(nn.Module):
    def __init__(self, in_features, num_centers, gamma):
        super().__init__()
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
        super().__init__()
        self.rbf_layers = nn.ModuleList()
        prev_units = in_features

        # 动态构建RBF层
        for num_centers, gamma in rbf_layers_config:
            self.rbf_layers.append(RBFLayer(prev_units, num_centers, gamma))
            prev_units = num_centers

        # 输出层
        self.linear = nn.Linear(prev_units, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        for layer in self.rbf_layers:
            x = layer(x)
        x = self.dropout(x)
        return self.linear(x)


# ====================== 训练流程 ====================== #
def train():
    sum_test_mae = 0
    sum_test_r2 = 0
    all_metrics = []

    for RANDOM_SEED in range(100):
        # 设置随机种子
        torch.manual_seed(RANDOM_SEED)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 数据加载和预处理
        df = pd.read_excel("../核桃仁表型信息_重新标定.xlsx", sheet_name="Sheet1")
        FEATURES = ['hutao_area', 'hutao_perimeter', 'hutao_area/hutao_perimeter',
                    'hutao_a', 'hutao_b', 'arithmetic_a_b_h_avg', 'geometry_a_b_h_avg',
                    'hutao_SI', 'hutao_ET', 'hutao_EV', 'fai', 'arithmetic_a_b_avg',
                    'geometry_a_b_avg']

        # 标准化处理
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[FEATURES])
        y = df['g'].values.reshape(-1, 1).astype(np.float32)

        # 数据集划分
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )

        # 创建数据加载器
        train_loader = DataLoader(
            CustomDataset(X_train, y_train),
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        test_loader = DataLoader(
            CustomDataset(X_test, y_test),
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        # 模型初始化
        model = RBFNN(
            in_features=X_train.shape[1],
            rbf_layers_config=RBF_LAYER_CONFIG
        ).to(device)

        # 训练配置
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

        # 训练循环
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            # 验证阶段
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()

            # 记录损失
            train_losses.append(epoch_loss / len(train_loader))
            val_losses.append(val_loss / len(test_loader))
            scheduler.step(val_loss)

        # 最终评估
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                y_true.extend(targets.cpu().numpy().flatten())
                y_pred.extend(outputs.cpu().numpy().flatten())

        # 计算指标
        mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
        r2 = r2_score(y_true, y_pred)

        # 保存指标
        sum_test_mae += mae
        sum_test_r2 += r2
        all_metrics.append((mae, r2))

        # 绘制散点图
        plot_scatter(y_true, y_pred, RANDOM_SEED, mae, r2)
        os.makedirs('rbnn', exist_ok=True)  # 新增目录创建
        # 保存模型
        torch.save({
            'model': model.state_dict(),
            'scaler': scaler,
            'config': RBF_LAYER_CONFIG,
            'features': FEATURES
        }, f'rbnn/seed_{RANDOM_SEED}.pth')
        print('训练结束')
    # 输出最终统计结果
    print(f"\nAverage Performance (100 seeds):")
    print(f"MAE: {sum_test_mae / 100:.4f}")
    print(f"R²: {sum_test_r2 / 100:.4f}")


if __name__ == '__main__':
    train()
