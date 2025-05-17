# 导入所需库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import scipy.stats as stats
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 配置参数


# 主函数
def main(RANDOM_STATE):
    DATA_PATH = '核桃仁表型信息.xlsx'  # 数据文件路径
    TEST_SIZE = 0.2  # 测试集比例
    RANDOM_STATE = RANDOM_STATE  # 随机种子
    MODEL_NAME = 'svr_model.pkl'  # 模型保存名称
    SCALER_NAME = 'scaler.pkl'  # 标准化器保存名称
    # 1. 数据加载与预处理
    print("加载数据...")
    df = pd.read_excel(DATA_PATH)

    # 提取特征和目标
    features = ['area_num', 'perimeter', 'a', 'b', 'a/b', 'area_num/perimeter', 'e']
    target = 'g'
    X = df[features]
    y = df[target]

    # 2. 数据标准化
    print("标准化数据...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 划分数据集
    print("划分训练集和测试集...")
    # 3. 划分数据集（保持原代码不变）
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # 4. 深度特征提取（新增核心模块）
    def build_feature_extractor(input_dim):
        """构建深度学习特征提取器"""
        model = Sequential([
            Dense(DL_LAYERS[0], activation='relu', input_shape=(input_dim,)),
            Dense(DL_LAYERS[1], activation='relu'),
            Dense(DL_LAYERS[2], activation='relu')
        ])
        return model

    # 预训练特征提取器
    feature_extractor = build_feature_extractor(X_train.shape[1])

    # 添加回归输出层进行预训练
    pretrain_model = Sequential([
        feature_extractor,
        Dense(1)
    ])

    # 编译预训练模型
    pretrain_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    # 训练带早停的预训练模型
    pretrain_model.fit(
        X_train, y_train,
        epochs=DL_EPOCHS,
        batch_size=DL_BATCH_SIZE,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )

    # 提取深度特征
    X_train_deep = feature_extractor.predict(X_train)
    X_test_deep = feature_extractor.predict(X_test)

    # 深度特征标准化
    deep_scaler = StandardScaler()
    X_train_deep = deep_scaler.fit_transform(X_train_deep)
    X_test_deep = deep_scaler.transform(X_test_deep)

    # 5. 训练深度SVR（参数调优部分保持结构，修改输入数据）
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

    # 使用深度特征训练
    grid_search.fit(X_train_deep, y_train)
    best_model = grid_search.best_estimator_

    # 5. 模型评估
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"测试集RMSE: {rmse:.4f}")
    print(f"测试集R²: {r2:.4f}")

    # 6. 可视化结果
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='w', s=80)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Mass (g)', fontsize=12)
    plt.ylabel('Predicted Mass (g)', fontsize=12)
    plt.title('SVR Prediction Performance', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.show()

    # 7. 保存模型
    joblib.dump(best_model, MODEL_NAME)
    joblib.dump(scaler, SCALER_NAME)
    print(f"模型已保存为 {MODEL_NAME}")
    print(f'random={RANDOM_STATE}')

    print(f'--------------------------------------------------------')
    return rmse, r2


if __name__ == "__main__":
    sum_rmse, sum_r2 = 0, 0
    for i in range(200):
        rmse, r2 = main(i)
        sum_rmse += rmse
        sum_r2 += r2
    print(f'rmse_avg:{sum_rmse / 200:.3f},r2_avg={sum_r2 / 200:.3f}')
