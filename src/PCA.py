import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json

# ==========================================
# 1. 讀取資料與準備
# ==========================================
INPUT_FILE = './data/ready_for_pca_6to9.csv' # 上一步生成的乾淨檔案
OUTPUT_FILE = './data/august_data_with_clusters.csv' # 輸出給 Step 3 用
PROFILE_FILE = './data/cluster_profiling.csv' # 商業分析報告

print("Step 1: 讀取資料並鎖定 8 月份快照...")

# 讀取資料
df = pd.read_csv(INPUT_FILE)

df.drop(columns=['indresi', 'indext', 'indfall'], inplace=True, errors='ignore')

# 【關鍵】只取 2015-08-28 的資料進行分群
# 我們要分析的是「當下」的客戶狀態
df_aug = df[df['fecha_dato'] == '2015-08-28'].copy()

df_aug.info()

# 將 ID 設為 Index，將日期移除 (這些不能放進模型算)
df_aug.set_index('ncodpers', inplace=True)
df_aug.drop(columns=['fecha_dato'], inplace=True)

print(f"8月份資料集形狀: {df_aug.shape}")
print("準備進行 PCA...")

# ==========================================
# 2. PCA 降維分析
# ==========================================
# 設定保留 95% 的解釋變異量，讓演算法自動決定需要幾個主成分
pca = PCA(n_components=0.95)
pca_data = pca.fit_transform(df_aug)

n_components = pca.n_components_
print(f"PCA 完成。保留 95% 資訊量需要 {n_components} 個主成分。")

# --- 視覺化 1: PCA 碎石圖 (Scree Plot) ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.title('PCA Explained Variance Ratio')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance')
plt.grid(True)
plt.axhline(y=0.95, color='r', linestyle='-', label='95% Threshold')
plt.legend()
plt.savefig('./data/pca_scree_plot.png')
print("已儲存 PCA 碎石圖至 ./data/pca_scree_plot.png")

# ==========================================
# 3. K-Means 分群
# ==========================================
# 這裡預設分為 6 群 (這在銀行業通常能分出：學生/小資/中產/富裕/退休/流失)
# 若想更精確，可以使用 Elbow Method 尋找最佳 K 值
N_CLUSTERS = 6
print(f"Step 2: 執行 K-Means 分群 (K={N_CLUSTERS})...")

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(pca_data)

# 將分群結果寫回 DataFrame
df_aug['cluster_label'] = cluster_labels

print("分群完成！各群人數分布：")
print(df_aug['cluster_label'].value_counts().sort_index())

# --- 視覺化 2: PCA 2D 分群圖 ---
# 取前兩個主成分來畫圖，看看分得開不開
plt.figure(figsize=(12, 8))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=cluster_labels, palette='viridis', s=10, alpha=0.6)
plt.title('Customer Segments (PCA 1 vs PCA 2)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.savefig('./data/cluster_scatter_plot.png')
print("已儲存分群散佈圖至 ./data/cluster_scatter_plot.png")

# ==========================================
# 4. 客群特徵分析 (Profiling) - 最重要的部分
# ==========================================
print("\nStep 3: 生成客群畫像報告 (Profiling)...")

# 因為之前的數據經過 StandardScaler (是小數點)，不好閱讀
# 我們這裡建議讀取「Scaler 參數」還原，或者直接看相對大小
# 這裡我們採用「Z-Score」解讀法：正值代表高於平均，負值代表低於平均

# 計算每一群的平均特徵
profile = df_aug.groupby('cluster_label').mean()

# 為了讓報告更好讀，我們把「產品持有率」獨立出來看
# 找出所有 ind_ 開頭的產品欄位
product_cols = [c for c in df_aug.columns if c.startswith('ind_') and c != 'ind_actividad_cliente']
# 找出人口特徵
demographic_cols = ['age', 'renta', 'antiguedad', 'month_joined', 'ind_actividad_cliente']

# 1. 人口特徵分析
print("\n--- 各群人口特徵 (標準化後數值) ---")
demo_profile = profile[demographic_cols]
print(demo_profile)

# 2. 產品持有分析 (找出每群最愛買的前 3 名產品)
print("\n--- 各群熱門產品 TOP 3 ---")
for i in range(N_CLUSTERS):
    # 取出該群的產品平均值 (即持有率，因為原本是 0/1)
    cluster_products = profile.loc[i, product_cols]
    # 排序
    top_products = cluster_products.sort_values(ascending=False).head(3)
    
    top_str = ", ".join([f"{idx}({val:.2f})" for idx, val in top_products.items()])
    print(f"Cluster {i}: {top_str}")

# 儲存完整報告
profile.to_csv(PROFILE_FILE)
print(f"\n完整商業分析報告已存至: {PROFILE_FILE}")
print("提示：請用 Excel 打開此報告，觀察哪些群體 income (renta) 特別高，或特別喜歡買 credit card。")

# ==========================================
# 5. 輸出結果
# ==========================================
# 將分群標籤存檔，給下一步驟 (Step 3 預測模型) 使用
# 這裡我們只存 ncodpers 和 cluster_label 即可，因為其他特徵 train_ver2 都有
output_df = df_aug[['cluster_label']].reset_index()
output_df.to_csv(OUTPUT_FILE, index=False)

print(f"已儲存分群標籤至: {OUTPUT_FILE}")
print("Step 1 客群分析完成！")