"""
深度SVR完整实现（PyTorch版本）
功能：特征深度学习提取 + 支持向量回归
作者：智能助手
日期：2024-01-20
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import scipy.stats as stats
from tqdm import tqdm


# ================== 配置区 ==================
class Config:
    # 数据配置
    DATA_PATH = '核桃仁表型信息.xlsx'  # 数据文件路径
    FEATURES = ['area_num', 'perimeter', 'a', 'b', 'a/b', 'area_num/perimeter', 'e']  # 特征列
    TARGET = 'g'  # 目标列
    TEST_SIZE = 0.2 # 测试集比例

    # 模型配置
    DL_EPOCHS = 300  # 预训练轮次
    DL_BATCH_SIZE = 64  # 批大小
    DL_LAYERS = [128, 64, 32]  # 特征提取网络结构
    DL_LEARNING_RATE = 0.001  # 学习率
    DL_PATIENCE = 10  # 早停耐心值

    # SVR配置
    SVR_N_ITER = 100  # 随机搜索次数
    CV_SPLITS = 5  # 交叉验证折数

    # 系统配置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动选择设备
    RANDOM_SEED = 42  # 全局随机种子
    MODEL_SAVE_PATH = 'models'  # 模型保存路径


# ================== 数据预处理 ==================
def load_data():
    """加载并预处理数据"""
    print(f"\n{'=' * 30} 数据加载 {'=' * 30}")
    df = pd.read_excel(Config.DATA_PATH)

    # 提取特征和目标
    X = df[Config.FEATURES]
    y = df[Config.TARGET]

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


# ================== 深度特征提取器 ==================
class DeepFeatureExtractor(nn.Module):
    """深度学习特征提取网络"""

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, Config.DL_LAYERS[0]),
            nn.BatchNorm1d(Config.DL_LAYERS[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(Config.DL_LAYERS[0], Config.DL_LAYERS[1]),
            nn.BatchNorm1d(Config.DL_LAYERS[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(Config.DL_LAYERS[1], Config.DL_LAYERS[2]),
            nn.BatchNorm1d(Config.DL_LAYERS[2]),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


def pretrain_feature_extractor(X_train, y_train):
    """预训练特征提取器"""
    print(f"\n{'=' * 30} 特征提取器预训练 {'=' * 30}")

    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X_train).to(Config.DEVICE)
    y_tensor = torch.FloatTensor(y_train.values).to(Config.DEVICE)

    # 初始化模型
    model = nn.Sequential(
        DeepFeatureExtractor(X_train.shape[1]),
        nn.Linear(Config.DL_LAYERS[-1], 1)
    ).to(Config.DEVICE)

    # 定义优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=Config.DL_LEARNING_RATE)
    criterion = nn.MSELoss()

    # 早停机制
    best_loss = np.inf
    patience_counter = 0

    # 训练循环
    progress_bar = tqdm(range(Config.DL_EPOCHS), desc="预训练进度")
    for epoch in progress_bar:
        model.train()
        optimizer.zero_grad()

        # 前向传播
        outputs = model(X_tensor).squeeze()
        loss = criterion(outputs, y_tensor)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 验证损失
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_tensor).squeeze()
            val_loss = criterion(val_outputs, y_tensor).item()

        # 早停判断
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(Config.MODEL_SAVE_PATH, 'best_extractor.pth'))
        else:
            patience_counter += 1

        if patience_counter >= Config.DL_PATIENCE:
            print(f"\n早停触发于第{epoch + 1}轮")
            break

        # 更新进度条
        progress_bar.set_postfix({'train_loss': loss.item(), 'val_loss': val_loss})

    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(Config.MODEL_SAVE_PATH, 'best_extractor.pth')))
    return model[0]  # 返回特征提取部分


# ================== 主流程 ==================
def main_run(random_state):
    """完整训练流程"""
    # 设置随机种子
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # 1. 数据加载
    X, y, scaler = load_data()

    # 2. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=Config.TEST_SIZE,
        random_state=random_state
    )

    # 3. 预训练特征提取器
    feature_extractor = pretrain_feature_extractor(X_train, y_train)

    # 4. 提取深度特征
    with torch.no_grad():
        feature_extractor.eval()
        X_train_deep = feature_extractor(
            torch.FloatTensor(X_train).to(Config.DEVICE)
        ).cpu().numpy()
        X_test_deep = feature_extractor(
            torch.FloatTensor(X_test).to(Config.DEVICE)
        ).cpu().numpy()

    # 二次标准化
    deep_scaler = StandardScaler()
    X_train_deep = deep_scaler.fit_transform(X_train_deep)
    X_test_deep = deep_scaler.transform(X_test_deep)

    # 5. SVR模型训练
    print(f"\n{'=' * 30} SVR参数优化 {'=' * 30}")
    param_dist = {
        'C': stats.loguniform(0.1, 100),
        'gamma': stats.loguniform(1e-4, 1),
        'epsilon': stats.uniform(0.01, 0.2)
    }

    svr_search = RandomizedSearchCV(
        SVR(kernel='rbf'),
        param_distributions=param_dist,
        n_iter=Config.SVR_N_ITER,
        cv=KFold(n_splits=Config.CV_SPLITS, shuffle=True, random_state=random_state),
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=random_state
    )

    svr_search.fit(X_train_deep, y_train)
    best_svr = svr_search.best_estimator_

    # 6. 模型评估
    y_pred = best_svr.predict(X_test_deep)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n评估结果 (随机种子{random_state})：")
    print(f"最佳参数: {svr_search.best_params_}")
    print(f"测试集RMSE: {rmse:.4f}")
    print(f"测试集R²: {r2:.4f}")

    # 7. 可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='w', s=80)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Value', fontsize=12)
    plt.ylabel('Predicted Value', fontsize=12)
    plt.title(f'DeepSVR Prediction (Seed={random_state})', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, f'prediction_{random_state}.png'))
    plt.close()

    # 8. 模型保存
    joblib.dump({
        'svr': best_svr,
        'feature_scaler': scaler,
        'deep_scaler': deep_scaler
    }, os.path.join(Config.MODEL_SAVE_PATH, f'deep_svr_{random_state}.pkl'))

    return rmse, r2


# ================== 主程序 ==================
if __name__ == "__main__":
    # 初始化配置
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)

    # 多次运行测试
    total_runs = 200
    results = []

    print(f"\n{'=' * 30} 深度SVR模型训练 {'=' * 30}")
    for run_id in tqdm(range(total_runs), desc="总体进度"):
        rmse, r2 = main_run(run_id)
        results.append((rmse, r2))

    # 统计最终结果
    final_rmse = np.mean([x[0] for x in results])
    final_r2 = np.mean([x[1] for x in results])

    print(f"\n{'=' * 30} 最终平均结果 ({total_runs}次运行) {'=' * 30}")
    print(f"平均RMSE: {final_rmse:.4f}")
    print(f"平均R²: {final_r2:.4f}")

    # 保存最终报告
    with open(os.path.join(Config.MODEL_SAVE_PATH, 'final_report.txt'), 'w') as f:
        f.write(f"DeepSVR Final Report ({total_runs} runs)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Average RMSE: {final_rmse:.4f}\n")
        f.write(f"Average R²: {final_r2:.4f}\n")
        f.write("\nConfiguration:\n")
        f.write(str(vars(Config)))