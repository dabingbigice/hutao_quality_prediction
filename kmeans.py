# 环境配置
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# === 样本分布 ===
# 0    414
# 1    328
# 2    229
# Name: count, dtype: int64
#
# === 聚类边界值 ===
# 边界1: 1.95 (Between Cluster 1 & 2)
# 边界2: 2.86 (Between Cluster 2 & 3)
#
# === 聚类质量 ===
# 轮廓系数: 0.572
# SSE误差: 134.75


# 数据加载
df = pd.read_excel('核桃仁表型信息.xlsx')
data = df['g'].values
X = data.reshape(-1, 1)  # 转换为二维数组

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 自动确定最佳K值（手肘法+轮廓系数法）
def find_optimal_k(max_k=10):
    sse = []
    silhouette_scores = []

    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)

        if k > 1:
            score = silhouette_score(X_scaled, kmeans.labels_)
            silhouette_scores.append(score)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, max_k + 1), sse, 'bo-')
    plt.title('Elbow Method')

    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k + 1), silhouette_scores, 'ro-')
    plt.title('Silhouette Analysis')
    plt.tight_layout()
    plt.show()


find_optimal_k(max_k=7)

# 模型训练（假设确定最佳K=3）
best_k = 3
kmeans = KMeans(
    n_clusters=best_k,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=42
)
kmeans.fit(X_scaled)

# 边界计算与可视化增强 ======================================================
# 获取原始尺度的聚类中心并排序
original_centers = scaler.inverse_transform(kmeans.cluster_centers_)
sorted_centers = np.sort(original_centers.ravel())

# 计算相邻中心之间的中点作为边界
boundaries = [(sorted_centers[i] + sorted_centers[i + 1]) / 2 for i in range(len(sorted_centers) - 1)]

# 可视化设置
plt.figure(figsize=(12, 7))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# 绘制数据点分布
for i in range(best_k):
    cluster_data = X[kmeans.labels_ == i]
    plt.scatter(
        x=cluster_data,
        y=[0] * len(cluster_data),
        color=colors[i],
        alpha=0.6,
        label=f'Cluster {i + 1}'
    )

# 绘制聚类边界垂直线
for b in boundaries:
    plt.axvline(x=b, color='black', linestyle='--',
                linewidth=2, alpha=0.8,
                label=f'Boundary {b:.2f}')

# 绘制聚类中心
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers, [0] * best_k, s=200,
            marker='X', c='black', label='Centroids')

# 图例与标注
plt.title(f'KMeans Clustering with Boundaries (K={best_k})')
plt.xlabel('核桃仁重量(g)')
plt.yticks([])
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 结果输出
print("=== 聚类中心 ===")
print(scaler.inverse_transform(kmeans.cluster_centers_))

print("\n=== 样本分布 ===")
print(pd.Series(kmeans.labels_).value_counts().sort_index())

print("\n=== 聚类边界值 ===")
for i, b in enumerate(boundaries):
    print(f"边界{i + 1}: {b:.2f} (Between Cluster {i + 1} & {i + 2})")

print("\n=== 聚类质量 ===")
print(f"轮廓系数: {silhouette_score(X_scaled, kmeans.labels_):.3f}")
print(f"SSE误差: {kmeans.inertia_:.2f}")
