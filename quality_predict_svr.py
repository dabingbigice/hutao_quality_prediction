# 导入所需库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 配置参数
DATA_PATH = '核桃仁表型信息.xlsx'  # 数据文件路径
TEST_SIZE = 0.2  # 测试集比例
RANDOM_STATE = 31  # 随机种子
MODEL_NAME = 'svr_model.pkl'  # 模型保存名称
SCALER_NAME = 'scaler.pkl'  # 标准化器保存名称


# 主函数
def main():
    # 1. 数据加载与预处理
    print("加载数据...")
    df = pd.read_excel(DATA_PATH)

    # 提取特征和目标
    features = ['area_num', 'perimeter', 'a', 'b', 'a/b', 'area_num/perimeter','e']
    target = 'g'
    X = df[features]
    y = df[target]

    # 2. 数据标准化
    print("标准化数据...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 划分数据集
    print("划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # 4. 训练SVR模型
    print("训练模型中...")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 1],
        'epsilon': [0.01, 0.1, 0.5]
    }

    grid_search = GridSearchCV(
        SVR(kernel='rbf'),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
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


if __name__ == "__main__":
    main()