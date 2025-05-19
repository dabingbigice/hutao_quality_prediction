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
from sklearn.inspection import permutation_importance
from joblib import parallel_backend
import warnings

# 配置参数
MODEL_CHOICE = 'svr'  # 可选 'svr', 'rf', 'xgb'
MODEL_NAMES = {
    'svr': 'Support Vector Regression',
    'rf': 'Random Forest',
    'xgb': 'XGBoost',
}
FEATURES = ['geometry_a_b_avg', 'arithmetic_a_b_avg', 'hutao_c', 'e', 'hutao_area', 'hutao_perimeter',
            'hutao_area/hutao_perimeter',
            'hutao_a', 'hutao_b', 'hutao_a/b', 'arithmetic_a_b_h_avg',
            'geometry_a_b_h_avg', 'hutao_SI', 'hutao_ET', 'hutao_EV', 'fai']
TARGET = 'g'
DATA_PATH = '核桃仁表型信息_重新标定.xlsx'
TEST_SIZE = 0.2
TOTAL_RUNS = 100

warnings.filterwarnings('ignore', category=UserWarning)


# ---------------------- 核心功能函数 ----------------------
def create_directory(path):
    """创建存储目录"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def plot_scatter(y_true, y_pred, errors, metrics, model_name, run_id):
    """可视化预测结果"""
    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    # 绘制散点图
    scatter = ax.scatter(
        y_true, y_pred,
        c=errors,
        cmap='viridis',
        alpha=0.7,
        edgecolor='w',
        s=80,
        label='Samples'
    )

    # 标注最大误差点
    max_error = errors.max()
    max_idx = errors.argmax()
    max_actual = y_true.values[max_idx]
    max_pred = y_pred[max_idx]

    ax.scatter(
        max_actual, max_pred,
        color='red',
        s=150,
        edgecolor='black',
        zorder=4,
        label=f'Max Error: {max_error:.2f}g'
    )

    # 添加标注
    ax.annotate(
        f'Actual: {max_actual:.2f}g\nPred: {max_pred:.2f}g',
        xy=(max_actual, max_pred),
        xytext=(max_actual + 0.1 * (y_true.max() - y_true.min()),
                max_pred + 0.1 * (y_true.max() - y_true.min())),
        arrowprops=dict(arrowstyle='->', color='black', lw=1),
        fontsize=10,
        ha='left'
    )

    # 参考线
    ref_line = [y_true.min(), y_true.max()]
    ax.plot(ref_line, ref_line, 'r--', lw=2, label='Ideal Prediction')

    # 颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Absolute Error (g)', rotation=270, labelpad=15)

    # 标题和标签
    title = (f"{model_name} | Performance\n"
             f"Train RMSE: {metrics['train_rmse']:.2f}g | Test RMSE: {metrics['test_rmse']:.2f}g\n"
             f"Train R²: {metrics['train_r2']:.2f} | Test R²: {metrics['test_r2']:.2f}\n"
             f"Errors >0.25g: {metrics['error_count']} ({metrics['error_percent']:.1%})")

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel(f'Actual Mass (g) Run: {run_id}', fontsize=12)
    ax.set_ylabel('Predicted Mass (g)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, linestyle=':')

    return plt


def calculate_feature_importance(model, model_type, X, y, feature_names):
    """计算特征重要性（修复版本）"""
    if model_type in ['rf', 'xgb']:
        importance = model.feature_importances_
    elif model_type == 'svr':
        with parallel_backend('threading', n_jobs=-1):
            result = permutation_importance(
                model, X, y,
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )
        importance = result.importances_mean
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # 归一化处理
    importance = np.abs(importance)
    importance /= importance.sum()

    return pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })


def plot_feature_importance(importance_df, model_name, run_id=None):
    """可视化特征重要性（修复版本）"""
    sorted_df = importance_df.sort_values('importance', ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_df['feature'], sorted_df['importance'],
             color='dodgerblue', edgecolor='black')

    title = f'Feature Importance ({model_name})'
    if run_id is not None:
        title += f' - Run {run_id}'

    plt.title(title, fontsize=14)
    plt.xlabel('Normalized Importance Score')
    plt.ylabel('Features')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return plt


def train_model(config, X_train, y_train, cv):
    """模型训练函数"""
    search = RandomizedSearchCV(
        config['model'],
        param_distributions=config['params'],
        n_iter=config['n_iter'],
        cv=cv,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


# ---------------------- 主执行流程 ----------------------
def main(run_id):
    """主执行函数"""
    # 数据准备
    df = pd.read_excel(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    # 标准化处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=TEST_SIZE,
        random_state=run_id
    )

    # 模型配置
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
        }
    }
    config = model_config[MODEL_CHOICE]

    # 交叉验证
    cv = KFold(n_splits=5, shuffle=True, random_state=run_id)

    # 模型训练
    model = train_model(config, X_train, y_train, cv)

    # 模型评估
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 计算指标
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_r2': r2_score(y_test, y_pred_test),
    }

    # 误差分析
    errors = np.abs(y_test - y_pred_test)
    error_count = (errors > 0.25).sum()
    metrics.update({
        'max_error': errors.max(),
        'error_count': error_count,
        'error_percent': error_count / len(y_test)
    })

    # 特征重要性分析
    importance_df = calculate_feature_importance(model, MODEL_CHOICE, X_test, y_test, FEATURES)

    # 保存结果
    result_dir = create_directory(f"results_{MODEL_CHOICE}")

    # 保存散点图
    scatter_plot = plot_scatter(y_test, y_pred_test, errors, metrics, MODEL_NAMES[MODEL_CHOICE], run_id)
    scatter_plot.savefig(
        f"{result_dir}/scatter_run_{run_id}.png",
        dpi=300,
        bbox_inches='tight'
    )
    scatter_plot.close()

    # 保存特征重要性
    importance_dir = create_directory(f"{result_dir}/feature_importance")
    importance_df.to_csv(f"{importance_dir}/run_{run_id}.csv", index=False)

    # 保存重要性可视化
    importance_plot = plot_feature_importance(importance_df, MODEL_NAMES[MODEL_CHOICE], run_id)
    importance_plot.savefig(
        f"{importance_dir}/plot_run_{run_id}.png",
        dpi=300,
        bbox_inches='tight'
    )
    importance_plot.close()

    return {
        **metrics,
        'feature_importance': importance_df
    }

    def aggregate_results():
        """汇总所有运行结果（修复版本）"""
        # 加载特征重要性文件
        importance_dir = f"results_{MODEL_CHOICE}/feature_importance"
        all_files = [f for f in os.listdir(importance_dir) if f.endswith('.csv')]

        # 合并数据
        dfs = []
        for file in all_files:
            df = pd.read_csv(os.path.join(importance_dir, file))
            dfs.append(df)

        full_df = pd.concat(dfs, ignore_index=True)

        # 计算统计指标
        summary = full_df.groupby('feature')['importance'].agg(
            ['mean', 'std', 'min', 'max', 'median']
        ).sort_values('mean', ascending=False)

        # 保存汇总结果
        summary_dir = create_directory(f"results_{MODEL_CHOICE}/summary")
        summary.to_csv(f"{summary_dir}/feature_importance_summary.csv")

        # 可视化汇总结果
        plt.figure(figsize=(12, 8))
        summary['mean'].sort_values().plot.barh(
            xerr=summary['std'],
            color='darkorange',
            edgecolor='black',
            capsize=3
        )
        plt.title(f'Average Feature Importance ({MODEL_NAMES[MODEL_CHOICE]})\n{TOTAL_RUNS} Runs')
        plt.xlabel('Normalized Importance Score')
        plt.ylabel('Features')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{summary_dir}/aggregated_importance.png", dpi=300, bbox_inches='tight')
        plt.close()

        return summary


def aggregate_results():
    """汇总所有运行结果（修复版本）"""
    # 加载特征重要性文件
    importance_dir = f"results_{MODEL_CHOICE}/feature_importance"
    all_files = [f for f in os.listdir(importance_dir) if f.endswith('.csv')]

    # 合并数据
    dfs = []
    for file in all_files:
        file_path = os.path.join(importance_dir, file)
        df = pd.read_csv(file_path)
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    # 计算统计指标
    summary = full_df.groupby('feature')['importance'].agg(
        ['mean', 'std', 'min', 'max', 'median', 'count']
    ).sort_values('mean', ascending=False)

    # 保存汇总结果
    summary_dir = create_directory(f"results_{MODEL_CHOICE}/summary")
    summary.to_csv(os.path.join(summary_dir, "feature_importance_summary.csv"), index=True)

    # 可视化汇总结果
    plt.figure(figsize=(12, 8))
    summary['mean'].sort_values().plot.barh(
        xerr=summary['std'],
        color='darkorange',
        edgecolor='black',
        capsize=3
    )
    plt.title(f'Average Feature Importance ({MODEL_NAMES[MODEL_CHOICE]})\n{TOTAL_RUNS} Runs')
    plt.xlabel('Normalized Importance Score ± STD')
    plt.ylabel('Features')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(summary_dir, "aggregated_importance.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    return summary


if __name__ == "__main__":
    # 初始化指标收集器
    metrics_collector = {
        'train_rmse': [],
        'train_r2': [],
        'test_rmse': [],
        'test_r2': [],
        'max_error': [],
        'error_count': [],
        'error_percent': []
    }

    # 执行主循环
    for run_id in range(TOTAL_RUNS):
        print(f"\n正在执行第 {run_id + 1}/{TOTAL_RUNS} 次运行...")
        result = main(run_id)

        # 收集指标
        for key in metrics_collector:
            metrics_collector[key].append(result[key])

    # 打印性能汇总
    print("\n性能汇总:")
    print(f"{'指标':<15} | {'平均值':<8} | {'标准差':<8} | {'最小值':<8} | {'最大值':<8}")
    print("-" * 60)
    for metric in ['train_rmse', 'train_r2', 'test_rmse', 'test_r2']:
        values = metrics_collector[metric]
        print(f"{metric:<15} | {np.mean(values):<8.3f} | {np.std(values):<8.3f} | "
              f"{np.min(values):<8.3f} | {np.max(values):<8.3f}")

    # 误差分析
    print("\n误差分析:")
    print(f"最大误差平均值: {np.mean(metrics_collector['max_error']):.3f}g")
    print(f"误差>0.25g比例: {np.mean(metrics_collector['error_percent']):.1%}")

    # 生成特征重要性汇总
    print("\n生成特征重要性汇总...")
    importance_summary = aggregate_results()
    print("\n特征重要性Top 5:")
    print(importance_summary.head(5))

    # 保存最终报告
    report_path = f"results_{MODEL_CHOICE}/final_report.txt"
    with open(report_path, 'w') as f:
        f.write("最终分析报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"模型类型: {MODEL_NAMES[MODEL_CHOICE]}\n")
        f.write(f"总运行次数: {TOTAL_RUNS}\n\n")

        f.write("性能指标:\n")
        f.write(f"{np.mean(metrics_collector['test_rmse']):<10.3f} | "
                f"{np.mean(metrics_collector['test_r2']):<10.3f}\n\n")

        f.write("重要特征Top 3:\n")
        top_features = importance_summary.head(3)
        for idx, (feat, row) in enumerate(top_features.iterrows(), 1):
            f.write(f"{idx}. {feat}: {row['mean']:.3f} ± {row['std']:.3f}\n")

    print(f"\n分析完成！完整结果保存在 results_{MODEL_CHOICE} 目录中")
