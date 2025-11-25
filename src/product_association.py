import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

# ==========================================
# 1. 讀取資料與準備
# ==========================================
# 使用之前生成的乾淨檔案 (或者 ready_for_pca_6to9.csv)
INPUT_FILE = './data/ready_for_pca_6to9.csv'
OUTPUT_RULES = './data/association_rules.csv'
OUTPUT_HEATMAP = './data/product_correlation_heatmap.png'

print("Step 1: 讀取資料並鎖定 8 月份快照...")

df = pd.read_csv(INPUT_FILE)

# 只取 2015-08-28 的資料 (Snapshot)
# 商品關聯通常看「當下持有的組合」
df_aug = df[df['fecha_dato'] == '2015-08-28'].copy()

# 找出所有產品欄位 (ind_ 開頭，且排除 ind_actividad_cliente 等非產品欄位)
# 這裡我們用一個簡單的邏輯：排除 ID, 日期, 以及非 0/1 的特徵
# 假設之前處理過的產品欄位都是 int/float 且數值為 0 或 1
product_cols = [
    'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
    'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
    'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
    'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
    'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
    'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1'
]

# 確保欄位存在
valid_prod_cols = [c for c in product_cols if c in df_aug.columns]
print(f"將分析 {len(valid_prod_cols)} 個產品。")

# 提取產品矩陣 (只包含 0/1)
# 確保是 bool 或 int (0/1) 格式，Apriori 需要
product_matrix = df_aug[valid_prod_cols].fillna(0).astype(int)

# ==========================================
# 2. 相關係數矩陣 (Correlation Matrix)
# ==========================================
print("\nStep 2: 計算相關係數矩陣並繪製熱力圖...")

corr_matrix = product_matrix.corr()

plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, center=0, annot=False, square=True)
plt.title('Product Correlation Matrix (August 2015)', fontsize=16)
plt.tight_layout()
plt.savefig(OUTPUT_HEATMAP)
print(f"熱力圖已儲存至: {OUTPUT_HEATMAP}")

# 找出相關性最高的 Top 10 組合
print("\n--- Top 5 正相關組合 (Positive Correlation) ---")
# 將矩陣拉平，排除對角線 (自己對自己)
corr_unstack = corr_matrix.unstack().sort_values(ascending=False)
corr_unstack = corr_unstack[corr_unstack < 1.0] # 排除 1.0
# 排除重複組合 (A-B 和 B-A 是一樣的，這裡簡單取前幾名通常會看到重複的，但沒關係)
print(corr_unstack.head(10))

# ==========================================
# 3. Apriori 關聯規則 (Association Rules)
# ==========================================
print("\nStep 3: 執行 Apriori 演算法挖掘關聯規則...")

# 1. 找出頻繁項目集 (Frequent Itemsets)
# min_support: 最小支持度。
# 設定 0.01 代表該組合至少要在 1% 的客戶中出現才算數。
# 銀行產品有些很冷門，設太高會找不到東西，設太低會跑很久。
MIN_SUPPORT = 0.01 

frequent_itemsets = apriori(product_matrix.astype(bool), min_support=MIN_SUPPORT, use_colnames=True)

print(f"找到 {len(frequent_itemsets)} 個頻繁項目集 (Support > {MIN_SUPPORT})")

if len(frequent_itemsets) > 0:
    # 2. 產生關聯規則
    # metric="lift": 我們用提升度 (Lift) 來篩選
    # Lift > 1 代表兩者有正向關係，Lift >> 1 代表強力推薦
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.1)
    
    # 排序：優先看 Lift 高的 (強關聯)，或者 Confidence 高的 (高機率)
    rules.sort_values(by='lift', ascending=False, inplace=True)
    
    # 整理輸出欄位
    output_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    
    print(f"挖掘出 {len(output_rules)} 條強關聯規則 (Lift > 1.1)")
    
    # 儲存
    output_rules.to_csv(OUTPUT_RULES, index=False)
    print(f"完整關聯規則已儲存至: {OUTPUT_RULES}")
    
    # 顯示前 5 條最強規則
    print("\n--- 最強關聯規則 Top 5 (依 Lift 排序) ---")
    for index, row in output_rules.head(5).iterrows():
        ant = list(row['antecedents'])[0]
        con = list(row['consequents'])[0]
        print(f"規則: 若買 [{ant}] -> 則買 [{con}]")
        print(f"      Support: {row['support']:.4f} (同時出現機率)")
        print(f"      Confidence: {row['confidence']:.4f} (買A後買B的機率)")
        print(f"      Lift: {row['lift']:.4f} (比隨機高幾倍)\n")
        
else:
    print("未找到符合條件的頻繁項目集，請嘗試降低 min_support。")

print("Step 2 商品關聯分析完成！")