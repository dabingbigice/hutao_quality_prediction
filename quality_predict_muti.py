import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
import joblib

# 1. 数据加载与探索性分析
df = pd.read_excel('核桃仁表型信息.xlsx')

# 检查特征相关性（关键改进1）
corr_matrix = df[['area_num', 'perimeter', 'a', 'b', 'a/b', 'area_num/perimeter','e']].corr()
plt.figure(figsize=(10, 6))
plt.matshow(corr_matrix, fignum=1)
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.colorbar()
plt.title('Feature Correlation Matrix')
plt.show()

# 2. 特征工程优化（关键改进2）
# 删除高相关性特征（保留a/b或area_num/perimeter）
features = ['area_num', 'perimeter', 'a', 'b']  # 删除area_num/perimeter

# 多项式特征扩展（二次项）
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df[features])
poly_feature_names = poly.get_feature_names_out(input_features=features)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# 3. 模型选择与超参数优化（关键改进3）
models = {
    'XGBoost': XGBRegressor(),
    'RandomForest': RandomForestRegressor(),
    'SVR': SVR(kernel='rbf'),
    'GBRT': GradientBoostingRegressor()
}

param_grids = {
    'XGBoost': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1]
    },
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10]
    },
    'SVR': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'epsilon': [0.05, 0.1]
    },
    'GBRT': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1]
    }
}

# 交叉验证优化（关键改进4）
best_models = {}
for name, model in models.items():
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        cv=KFold(n_splits=5, shuffle=True),
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_scaled, df['g'])
    best_models[name] = grid_search.best_estimator_
    print(f"{name} Best Params: {grid_search.best_params_}")
    print(f"{name} Best RMSE: {np.sqrt(-grid_search.best_score_):.4f}\n")

# 4. 最优模型评估
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['g'], test_size=0.2, random_state=42)

final_model = best_models['SVR']  # 假设XGBoost最优
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f'Final Model RMSE: {rmse:.4f}')
print(f'Final R-squared: {r2:.4f}')

# 5. 特征重要性分析（关键改进5）
if hasattr(final_model, 'feature_importances_'):
    plt.figure(figsize=(12, 6))
    importances = final_model.feature_importances_
    indices = np.argsort(importances)[-10:]  # 取前10重要特征
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [poly_feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

# 6. 模型保存
joblib.dump(final_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')