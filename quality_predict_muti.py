import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 配置参数
MODEL_CHOICE = 'rf'  # 可选 'svr', 'rf', 'xgb', 'lgb'
MODEL_NAMES = {
    'svr': 'Support Vector Regression',
    'rf': 'Random Forest',
    'xgb': 'XGBoost',
    'lgb': 'LightGBM'
}


def main(RANDOM_STATE):
    # 数据配置
    DATA_PATH = '核桃仁表型信息.xlsx'
    TEST_SIZE = 0.8
    FEATURES = ['hutao_area','hutao_perimeter','hutao_area/hutao_perimeter','hutao_a','hutao_b','hutao_a/b']
    TARGET = 'g'

    # 数据加载与预处理
    df = pd.read_excel(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    # 标准化处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 模型参数配置
    model_config = {
        'svr': {
            'model': SVR(kernel='rbf'),
            'params': {
                'C': stats.loguniform(0.5, 50),
                'gamma': stats.loguniform(0.005, 0.5),
                'epsilon': stats.uniform(0.005, 0.1)
            },
            'n_iter': 50
        },
        'rf': {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators': stats.randint(100, 500),
                'max_depth': stats.randint(5, 20),
                'min_samples_split': stats.randint(2, 10)
            },
            'n_iter': 30
        },
        'xgb': {
            'model': XGBRegressor(objective='reg:squarederror'),
            'params': {
                'learning_rate': stats.loguniform(0.01, 0.3),
                'n_estimators': stats.randint(100, 500),
                'max_depth': stats.randint(3, 8),
                'subsample': stats.uniform(0.6, 0.4)
            },
            'n_iter': 30
        },
        'lgb': {
            'model': LGBMRegressor(),
            'params': {
                'learning_rate': stats.loguniform(0.01, 0.3),
                'n_estimators': stats.randint(100, 500),
                'num_leaves': stats.randint(20, 50),
                'min_child_samples': stats.randint(10, 30)
            },
            'n_iter': 30
        }
    }

    # 模型初始化
    config = model_config[MODEL_CHOICE]

    # 交叉验证策略
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 参数搜索
    search = RandomizedSearchCV(
        config['model'],
        param_distributions=config['params'],
        n_iter=config['n_iter'],
        cv=cv,
        scoring='neg_mean_squared_error',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # 模型评估（测试集）
    y_pred_test = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)

    # 训练集评估
    y_pred_train = best_model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)

    # 误差分析
    errors = np.abs(y_test.values - y_pred_test)
    max_error = errors.max()
    max_idx = errors.argmax()
    print(f'max_index={max_idx}')
    max_actual = y_test.values[max_idx]
    max_predicted = y_pred_test[max_idx]

    # 可视化设置
    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    # 绘制主散点图
    scatter = ax.scatter(
        y_test, y_pred_test,
        c=errors,
        cmap='viridis',
        alpha=0.7,
        edgecolor='w',
        s=80,
        label='Samples'
    )

    # 标注最大误差点
    ax.scatter(
        max_actual, max_predicted,
        color='red',
        s=150,
        edgecolor='black',
        zorder=4,
        label=f'Max Error: {max_error:.2f}g'
    )

    # 添加误差注释
    ax.annotate(
        f'Actual: {max_actual:.2f}g\nPred: {max_predicted:.2f}g',
        xy=(max_actual, max_predicted),
        xytext=(max_actual + 0.1 * (y.max() - y.min()),
                max_predicted + 0.1 * (y.max() - y.min())),
        arrowprops=dict(arrowstyle='->', color='black', lw=1),
        fontsize=10,
        ha='left'
    )

    # 参考线
    ref_line = [y.min(), y.max()]
    ax.plot(ref_line, ref_line, 'r--', lw=2, label='Ideal Prediction')

    # 颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Absolute Error (g)', rotation=270, labelpad=15)

    # 图形装饰
    ax.set_xlabel(f'Actual Mass (g) seed={RANDOM_STATE}', fontsize=12)
    ax.set_ylabel('Predicted Mass (g)', fontsize=12)
    title = (f"{MODEL_NAMES[MODEL_CHOICE]} Performance\n"
             f"Train RMSE: {train_rmse:.2f}g | Test RMSE: {test_rmse:.2f}g\n"
             f"Train R²: {train_r2:.2f} | Test R²: {test_r2:.2f}")
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()
    output_dir = "results_" + f'{MODEL_CHOICE}'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        f"{output_dir}/R²_{test_r2}_RMSE_{RANDOM_STATE}_{max_error:.2f}.png",
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    plt.show()

    return train_rmse, train_r2, test_rmse, test_r2


if __name__ == "__main__":
    total_runs = 100
    metrics = {
        'train_rmse': [],
        'train_r2': [],
        'test_rmse': [],
        'test_r2': []
    }

    for i in range(total_runs):
        print(f"\nRun {i + 1}/{total_runs}")
        train_rmse, train_r2, test_rmse, test_r2 = main(i)
        metrics['train_rmse'].append(train_rmse)
        metrics['train_r2'].append(train_r2)
        metrics['test_rmse'].append(test_rmse)
        metrics['test_r2'].append(test_r2)

    # 汇总统计
    print(f"\n{'=' * 40}")
    print("[训练集统计]")
    print(f"平均 RMSE: {np.mean(metrics['train_rmse']):.3f} ± {np.std(metrics['train_rmse']):.3f}")
    print(f"平均 R²: {np.mean(metrics['train_r2']):.3f} ± {np.std(metrics['train_r2']):.3f}")
    print(f"最小 RMSE: {np.min(metrics['train_rmse']):.3f}")
    print(f"最大 R²: {np.max(metrics['train_r2']):.3f}")

    print("\n[测试集统计]")
    print(f"平均 RMSE: {np.mean(metrics['test_rmse']):.3f} ± {np.std(metrics['test_rmse']):.3f}")
    print(f"平均 R²: {np.mean(metrics['test_r2']):.3f} ± {np.std(metrics['test_r2']):.3f}")
    print(f"最小 RMSE: {np.min(metrics['test_rmse']):.3f}")
    print(f"最大 R²: {np.max(metrics['test_r2']):.3f}")
    print('=' * 40)