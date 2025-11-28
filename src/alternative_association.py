import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform

# ==========================================
# 1. 讀取資料
# ==========================================
INPUT_FILE = './data/ready_for_pca_6to9.csv'
df = pd.read_csv(INPUT_FILE)

# 只取 8 月份快照
df_aug = df[df['fecha_dato'] == '2015-08-28'].copy()

# 產品欄位
product_cols = [c for c in df.columns if 'ind_' in c and 'ult1' in c and 'label' not in c]
# 轉成 0/1 矩陣
product_matrix = df_aug[product_cols].fillna(0).astype(int).T # 注意這裡轉置了(Transpose)，變成 行=產品, 列=用戶

print(f"分析產品數: {product_matrix.shape[0]}")

# ==========================================
# 方法一：Jaccard 相似度 (Item-Based CF)
# ==========================================
print("\n=== 方法 1: Jaccard 相似度分析 (Item-Based) ===")
print("Jaccard 係數 = (A 交集 B) / (A 聯集 B)")
print("比起相關係數，這更適合衡量 '兩個產品是否總是綁在一起'...")

# 計算 Jaccard 距離 (數值越小越相似)
# metric='jaccard'
jaccard_distances = pdist(product_matrix.values, metric='jaccard')
# 轉成相似度 (1 - 距離)
jaccard_similarity = 1 - squareform(jaccard_distances)

# 轉成 DataFrame 方便看
jaccard_df = pd.DataFrame(jaccard_similarity, index=product_cols, columns=product_cols)

# 畫圖
plt.figure(figsize=(16, 12))
sns.heatmap(jaccard_df, cmap='Blues', annot=False) # 顏色越深代表越相似
plt.title('Product Jaccard Similarity (Who buys A also buys B)')
plt.tight_layout()
plt.savefig('./data/product_jaccard_similarity.png')
print("已儲存 Jaccard 相似度圖: ./data/product_jaccard_similarity.png")

# 找出最相似的 Top 5 組合
print("\n[Jaccard 最相似的產品對]:")
# 拉平矩陣
sim_unstack = jaccard_df.unstack().sort_values(ascending=False)
sim_unstack = sim_unstack[sim_unstack < 1.0] # 排除自己跟自己
print(sim_unstack.head(10))


# ==========================================
# 方法二：簡單轉移機率 (Simple Transition Probability)
# ==========================================
print("\n=== 方法 2: 7月->8月 產品轉移機率矩陣 ===")
print("計算: 如果7月有產品 A，8月'新增'產品 B 的機率...")

# 準備 7 月和 8 月資料
df_jul = df[df['fecha_dato'] == '2015-07-28'].set_index('ncodpers')[product_cols]
df_aug = df[df['fecha_dato'] == '2015-08-28'].set_index('ncodpers')[product_cols]

# 找出共同客戶
common_idx = df_jul.index.intersection(df_aug.index)
df_jul = df_jul.loc[common_idx]
df_aug = df_aug.loc[common_idx]

# 初始化矩陣 (Rows: 7月持有, Cols: 8月新增)
transition_matrix = pd.DataFrame(0.0, index=product_cols, columns=product_cols)

# 計算 8 月的新增購買 (New Purchase Matrix)
# 邏輯: 8月有 - 7月有 = 1 (新增)
# 負數設為 0 (流失不算新增)
new_purchase_matrix = (df_aug - df_jul).clip(lower=0)

# 開始計算機率 P(New B | Old A)
for prod_A in product_cols:
    # 分母: 7月持有 A 的人數
    users_with_A = df_jul[df_jul[prod_A] == 1].index
    count_A = len(users_with_A)
    
    if count_A > 0:
        # 分子: 這群人中，8 月新增 B 的人數
        # 我們直接對這群人在 new_purchase_matrix 的 column 做加總
        new_buys_by_group_A = new_purchase_matrix.loc[users_with_A].sum()
        
        # 算出機率
        probs = new_buys_by_group_A / count_A
        transition_matrix.loc[prod_A] = probs

# 畫圖 (這是最有預測力的圖)
plt.figure(figsize=(20, 16))
sns.heatmap(transition_matrix, cmap='Greens', vmin=0, vmax=0.05) # vmax設小一點因為轉移率通常不高
plt.title('Product Transition Probability Matrix (P(New Col | Has Row))')
plt.xlabel('Target Product (New Purchase in Aug)')
plt.ylabel('Source Product (Held in Jul)')
plt.tight_layout()
plt.savefig('./data/product_transition_matrix.png')
print("已儲存轉移機率矩陣: ./data/product_transition_matrix.png")

# 找出轉移機率最高的組合
print("\n[轉移機率最高的組合 - P(New B | Old A)]:")
trans_unstack = transition_matrix.unstack().sort_values(ascending=False) # 這裡不用排除對角線，因為自己不能新增自己
print(trans_unstack.head(10))