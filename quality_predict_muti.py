import os

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# 配置参数
MODEL_CHOICE = 'svr'  # 可选 'svr', 'rf', 'xgb', 'lgb'
MODEL_NAMES = {
    'svr': 'Support Vector Regression',
    'rf': 'Random Forest',
    'xgb': 'XGBoost',
}


def main(RANDOM_STATE):
    # 数据配置
    DATA_PATH = '核桃仁表型信息_重新标定.xlsx'
    TEST_SIZE = 0.2
    FEATURES = ['hutao_area', 'hutao_perimeter', 'hutao_area/hutao_perimeter', 'hutao_a', 'hutao_b',
                'arithmetic_a_b_h_avg', 'geometry_a_b_h_avg', 'hutao_SI', 'hutao_ET', 'hutao_EV', 'fai',
                'arithmetic_a_b_avg', 'geometry_a_b_avg']
    TARGET = 'g'

    # 数据加载与预处理
    df = pd.read_excel(DATA_PATH)
    # 误差筛选
    # df = df[df['error'] < 5]
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
            # 'params': {
            #     'C': stats.loguniform(0.5, 50),
            #     'gamma': stats.loguniform(0.005, 0.5),
            #     'epsilon': stats.uniform(0.005, 0.1)
            # },
            'params': {
                'C': stats.loguniform(0.1, 1000),
                'gamma': stats.loguniform(1e-5, 1),
                'epsilon': stats.loguniform(0.001, 0.5)
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
        }
    }

    # 模型初始化
    # config = model_config[MODEL_CHOICE]
    config_1 = model_config[MODEL_CHOICE]
    # config_2 = model_config['rf']
    # config_3 = model_config['rf']

    # 交叉验证策略
    # cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    # 1号模型
    # 参数搜索
    search_1 = RandomizedSearchCV(
        config_1['model'],
        param_distributions=config_1['params'],
        n_iter=config_1['n_iter'],
        cv=cv,
        scoring='neg_mean_squared_error',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    search_1.fit(X_train, y_train)
    best_model_1 = search_1.best_estimator_

    # 模型评估（测试集）
    y_pred_test_1 = best_model_1.predict(X_test)
    test_rmse_1 = np.sqrt(mean_squared_error(y_test, y_pred_test_1))
    test_r2_1 = r2_score(y_test, y_pred_test_1)

    # 训练集评估
    y_pred_train_1 = best_model_1.predict(X_train)
    train_rmse_1 = np.sqrt(mean_squared_error(y_train, y_pred_train_1))
    train_r2_1 = r2_score(y_train, y_pred_train_1)

    # 2号模型

    # search_2 = RandomizedSearchCV(
    #     config_2['model'],
    #     param_distributions=config_2['params'],
    #     n_iter=config_2['n_iter'],
    #     cv=cv,
    #     scoring='neg_mean_squared_error',
    #     random_state=RANDOM_STATE,
    #     n_jobs=-1
    # )
    # search_2.fit(X_train, y_train)
    # best_model_2 = search_2.best_estimator_
    #
    # # 模型评估（测试集）
    # y_pred_test_2 = best_model_2.predict(X_test)
    # test_rmse_2 = np.sqrt(mean_squared_error(y_test, y_pred_test_2))
    # test_r2_2 = r2_score(y_test, y_pred_test_2)
    #
    # # 训练集评估
    # y_pred_train_2 = best_model_2.predict(X_train)
    # train_rmse_2 = np.sqrt(mean_squared_error(y_train, y_pred_train_2))
    # train_r2_2 = r2_score(y_train, y_pred_train_2)

    # 3号模型

    # search_3 = RandomizedSearchCV(
    #     config_3['model'],
    #     param_distributions=config_3['params'],
    #     n_iter=config_3['n_iter'],
    #     cv=cv,
    #     scoring='neg_mean_squared_error',
    #     random_state=RANDOM_STATE,
    #     n_jobs=-1
    # )
    # search_3.fit(X_train, y_train)
    # best_model_3 = search_3.best_estimator_
    #
    # # 模型评估（测试集）
    # y_pred_test_3 = best_model_3.predict(X_test)
    # test_rmse_3 = np.sqrt(mean_squared_error(y_test, y_pred_test_3))
    # test_r2_3 = r2_score(y_test, y_pred_test_3)
    #
    # # 训练集评估
    # y_pred_train_3 = best_model_3.predict(X_train)
    # train_rmse_3 = np.sqrt(mean_squared_error(y_train, y_pred_train_3))
    # train_r2_3 = r2_score(y_train, y_pred_train_3)

    y_pred_test = y_pred_test_1
    test_r2 = test_r2_1
    test_rmse = test_rmse_1
    train_rmse = train_rmse_1
    train_r2 = train_r2_1

    # 误差分析
    errors = np.abs(y_test.values - y_pred_test)

    # 误差大于0.3g的数量
    count = (errors > 0.3).sum()
    max_error = errors.max()
    max_idx = errors.argmax()
    max_actual = y_test.values[max_idx]
    max_predicted = y_pred_test[max_idx]

    # ================== 新增模型保存功能 ==================
    # 创建模型保存目录
    model_save_dir = f"saved_models/{MODEL_CHOICE}"
    os.makedirs(model_save_dir, exist_ok=True)

    # 生成带随机种子的文件名
    filename_suffix = f"{MODEL_CHOICE}_seed{RANDOM_STATE}"

    # 保存训练好的模型
    joblib.dump(
        best_model_1,
        f"{model_save_dir}/model_{filename_suffix}.pkl"
    )

    # 保存标准化器（重要！）
    joblib.dump(
        scaler,
        f"{model_save_dir}/scaler_{filename_suffix}.pkl"
    )

    # 保存超参数配置（可选）
    pd.DataFrame(search_1.best_params_, index=[0]).to_csv(
        f"{model_save_dir}/params_{filename_suffix}.csv",
        index=False
    )

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
    # title = (f"{MODEL_NAMES[MODEL_CHOICE]} Performance\n"
    #          f"Train RMSE: {train_rmse:.2f}g | Test RMSE: {test_rmse:.2f}g\n"
    #          f"Train R²: {train_r2:.2f} | Test R²: {test_r2:.2f}"
    #          f"error>RMSE_count:{count}")
    title = (f"{MODEL_CHOICE} | Performance\n"
             f"Train RMSE: {train_rmse:.2f}g | Test RMSE: {test_rmse:.2f}g\n"
             f"Train R²: {train_r2:.2f} | Test R²: {test_r2:.2f}\n"
             f"error>0.3 _count:{count} | percent :{count / len(errors):.2f}\n"

             )
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()
    # output_dir = "results_" + f'{MODEL_CHOICE}'
    output_dir = f"results_{MODEL_CHOICE}/scatter"

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        f"{output_dir}/random_{RANDOM_STATE}_R²_{test_r2:.2f}_RMSE_{test_rmse:.2f}_{max_error:.2f}.png",
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    plt.show()

    return train_rmse, train_r2, test_rmse, test_r2, max_error, count, len(y_pred_test)


if __name__ == "__main__":
    total_runs = 100
    metrics = {
        'train_rmse': [],
        'train_r2': [],
        'test_rmse': [],
        'test_r2': []
    }
    max_error = 0
    error_gt_rmse = 0
    error_gt_rmse_precent = 0
    len_test = 0
    for i in range(total_runs):
        print(f"\nRun {i + 1}/{total_runs}")
        train_rmse, train_r2, test_rmse, test_r2, max_error_1, count, len_test = main(i)
        metrics['train_rmse'].append(train_rmse)
        metrics['train_r2'].append(train_r2)
        metrics['test_rmse'].append(test_rmse)
        metrics['test_r2'].append(test_r2)
        max_error += max_error_1
        error_gt_rmse += count
        len_test = len_test

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
    print(f"平均 max_error: {max_error / 100:.3f}")
    print(f"平均误差大于0.3百分比: {(error_gt_rmse / len_test) :.3f}")
    print('=' * 40)
