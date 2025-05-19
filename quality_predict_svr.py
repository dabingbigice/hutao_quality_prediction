import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 配置参数
DL_EPOCHS = 100
DL_BATCH_SIZE = 4


def main(RANDOM_STATE):
    # 设置随机种子
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")


    DATA_PATH = '核桃仁表型信息_重新标定.xlsx'
    TEST_SIZE = 0.2
    MODEL_NAME = 'svr_model.pkl'
    SCALER_NAME = 'scaler.pkl'

    # 1. 数据加载与预处理
    print("加载数据...")
    df = pd.read_excel(DATA_PATH)
    features = ['e', 'hutao_area', 'hutao_perimeter', 'hutao_area/hutao_perimeter',
                'hutao_a', 'hutao_b', 'hutao_a/b', 'arithmetic_a_b_h_avg',
                'geometry_a_b_h_avg', 'hutao_SI', 'hutao_ET', 'hutao_EV', 'fai']
    target = 'g'
    X = df[features].values
    y = df[target].values

    # 2. 数据标准化
    print("标准化数据...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 划分数据集
    print("划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1).to(device)  # 直接迁移到设备
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=DL_BATCH_SIZE, shuffle=True)

    # 4. 多分支并行卷积结构
    class MultiScaleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            # 并行卷积分支
            self.branch1 = nn.Conv1d(1, 8, kernel_size=1, padding=0)
            self.branch3 = nn.Conv1d(1, 8, kernel_size=3, padding=1)
            self.branch5 = nn.Conv1d(1, 8, kernel_size=5, padding=2)
            self.branch7 = nn.Conv1d(1, 8, kernel_size=7, padding=3)
            self.branch9 = nn.Conv1d(1, 8, kernel_size=9, padding=4)

            # 特征融合
            self.fusion = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(40, 32, kernel_size=1),  # 1x1卷积降维
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)  # 全局平均池化
            )
            # 回归头
            self.regressor = nn.Linear(32, 1)

        def forward(self, x):
            # 并行分支处理
            b1 = self.branch1(x)
            b3 = self.branch3(x)
            b5 = self.branch5(x)
            b7 = self.branch7(x)
            b9 = self.branch9(x)

            # 特征拼接与融合
            combined = torch.cat([b1, b3, b5, b7, b9], dim=1)
            features = self.fusion(combined).squeeze(-1)
            return self.regressor(features).squeeze()

    # 初始化模型
    model = MultiScaleCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练CNN
    print("训练多尺度CNN特征提取器...")
    best_loss = float('inf')
    for epoch in range(DL_EPOCHS):
        model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            # 数据迁移到设备
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_multiscale_cnn.pth')
    # 5. 特征提取器定义
    class FeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.multiscale = nn.ModuleList([
                nn.Conv1d(1, 8, k, padding=(k // 2))
                for k in [1, 3, 5, 7, 9]
            ])
            self.fusion = nn.Sequential(
                nn.Conv1d(40, 32, 1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )

        def forward(self, x):
            features = [conv(x) for conv in self.multiscale]
            combined = torch.cat(features, dim=1)
            return self.fusion(combined).squeeze(-1)

    feature_model = FeatureExtractor().to(device)
    feature_model.load_state_dict(torch.load('best_multiscale_cnn.pth', map_location=device), strict=False)
    feature_model.eval()

    # 提取特征
    with torch.no_grad():
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1).to(device)
        X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1).to(device)
        X_train_features = feature_model(X_train_tensor).cpu().numpy()
        X_test_features = feature_model(X_test_tensor).cpu().numpy()
    # 深度特征标准化
    deep_scaler = StandardScaler()
    X_train_deep = deep_scaler.fit_transform(X_train_features)
    X_test_deep = deep_scaler.transform(X_test_features)

    # 5. 训练深度SVR
    param_dist = {
        'C': stats.loguniform(0.5, 50),
        'gamma': stats.loguniform(0.005, 0.5),
        'epsilon': stats.uniform(0.005, 0.1)
    }

    grid_search = RandomizedSearchCV(
        SVR(kernel='rbf'),
        param_distributions=param_dist,
        n_iter=50,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42
    )

    grid_search.fit(X_train_deep, y_train)
    best_model = grid_search.best_estimator_

    # 模型评估
    y_pred = best_model.predict(X_test_deep)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"测试集RMSE: {rmse:.4f}")
    print(f"测试集R²: {r2:.4f}")

    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='w', s=80)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Mass (g)', fontsize=12)
    plt.ylabel('Predicted Mass (g)', fontsize=12)
    plt.title('CNN-SVR Prediction Performance', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.show()

    # 保存模型
    joblib.dump(best_model, MODEL_NAME)
    joblib.dump(scaler, SCALER_NAME)
    print(f"模型已保存为 {MODEL_NAME}")
    print(f'random={RANDOM_STATE}')
    print(f'--------------------------------------------------------')

    return rmse, r2


if __name__ == "__main__":
    sum_rmse, sum_r2 = 0, 0
    for i in range(50):
        rmse, r2 = main(i)
        sum_rmse += rmse
        sum_r2 += r2
    print(f'rmse_avg:{sum_rmse / 200:.3f}, r2_avg={sum_r2 / 200:.3f}')