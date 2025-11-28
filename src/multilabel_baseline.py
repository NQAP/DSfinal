import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 引入模型
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# ==========================================
# 1. 資料準備 (Raw Data Only)
# ==========================================
# INPUT_FILE = './data/ready_for_pca_6to9.csv'
# print("Step 1: 讀取原始清洗資料 (Baseline)...")
# df = pd.read_csv(INPUT_FILE)

# # 確保 ncodpers 是整數
# df['ncodpers'] = df['ncodpers'].astype(int)

# # --- A. 準備特徵 (X) ---
# # 只取 8 月份資料
# print("   - 鎖定 2015-08-28 作為特徵基準...")
# X_df = df[df['fecha_dato'] == '2015-08-28'].copy()

# # 移除日期
# X_df.drop(columns=['fecha_dato'], inplace=True, errors='ignore')

# # 找出產品欄位 (這些也是特徵，因為"現在有什麼"會影響"未來買什麼")
# product_cols = [c for c in df.columns if 'ind_' in c and 'ult1' in c]

# # 找出人口特徵 (除了產品、ID以外的)
# demo_cols = [c for c in X_df.columns if c not in product_cols and c != 'ncodpers']

# # 定義 Baseline 特徵集: 人口 + 當前產品
# # 注意：這裡完全沒有 rule_... 或 cluster_...
# baseline_features = demo_cols + product_cols
# print(f"   - Baseline 使用特徵數: {len(baseline_features)}")

# # --- B. 準備目標 (Y) ---
# print("   - 計算 2015-09-28 的新增購買行為...")
# Y_df_sep = df[df['fecha_dato'] == '2015-09-28'][['ncodpers'] + product_cols].copy()
# Y_df_sep.columns = ['ncodpers'] + [f"{c}_target" for c in product_cols]

# # 合併 8月 和 9月
# merged = pd.merge(X_df[['ncodpers'] + product_cols], Y_df_sep, on='ncodpers', how='left')

# # 建立 Label 字典 (Product -> Series)
# labels = {}
# for prod in product_cols:
#     target_col = f"{prod}_target"
#     merged[target_col] = merged[target_col].fillna(0)
#     # 邏輯: 9月有 (1) & 8月沒有 (0)
#     labels[prod] = ((merged[target_col] == 1) & (merged[prod] == 0)).astype(int)

# # 準備訓練矩陣
# X = X_df[baseline_features] # 確保只用原始特徵
# # ID 不放進去訓練，但可以用來對齊(這裡直接用index對齊即可)

# # 切分訓練/測試 (固定 random_state 以確保跟實驗組切的一樣)
# X_train, X_test, _, _ = train_test_split(X, X, test_size=0.2, random_state=42)

# print(f"   - 訓練集大小: {len(X_train)}")

# # ==========================================
# # 2. 定義模型 (與實驗組保持完全一致)
# # ==========================================
# def get_models(pos_ratio):
#     models = {}
#     models['DecisionTree'] = DecisionTreeClassifier(
#         max_depth=8, class_weight='balanced', random_state=42
#     )
#     models['RandomForest'] = RandomForestClassifier(
#         n_estimators=50, max_depth=8, class_weight='balanced', n_jobs=-1, random_state=42
#     )
#     models['XGBoost'] = xgb.XGBClassifier(
#         n_estimators=100, max_depth=5, learning_rate=0.1, objective='binary:logistic',
#         scale_pos_weight=pos_ratio, n_jobs=-1, eval_metric='auc', tree_method='hist'
#     )
#     models['LightGBM'] = lgb.LGBMClassifier(
#         n_estimators=100, num_leaves=31, learning_rate=0.05, objective='binary',
#         class_weight='balanced', n_jobs=-1, verbose=-1
#     )
#     return models

# # ==========================================
# # 3. 訓練迴圈 (Baseline)
# # ==========================================
# print("\nStep 2: 開始 Baseline 全面評測...")
# results_data = []
# start_global = time.time()

# for prod in product_cols:
#     product_name = prod
    
#     # 取得該產品的 Y (並切分)
#     y_series = labels[prod]
#     y_train, y_test = train_test_split(y_series, test_size=0.2, random_state=42)
    
#     # 檢查樣本
#     n_pos = y_train.sum()
#     if n_pos < 10:
#         continue # 跳過樣本過少的產品
        
#     ratio = (len(y_train) - n_pos) / n_pos
#     current_models = get_models(ratio)
    
#     print(f"Baseline Eval: [{product_name}] (Pos: {n_pos})...")
    
#     for model_name, clf in current_models.items():
#         try:
#             clf.fit(X_train, y_train)
#             y_pred_prob = clf.predict_proba(X_test)[:, 1]
#             score = roc_auc_score(y_test, y_pred_prob)
#         except:
#             score = 0.5
            
#         results_data.append({
#             'Product': product_name,
#             'Model': model_name,
#             'Baseline_AUC': score
#         })

# print(f"\nBaseline 訓練結束！耗時: {time.time() - start_global:.2f} 秒")

# # ==========================================
# # 4. 輸出與比較準備
# # ==========================================
# df_res = pd.DataFrame(results_data)
# df_res.to_csv('./data/baseline_multi_model_results.csv', index=False)

# 畫 Baseline 熱力圖
df_res = pd.read_csv('./data/baseline_multi_model_results.csv')
pivot_table = df_res.pivot(index='Product', columns='Model', values='Baseline_AUC')
plt.figure(figsize=(10, 12))
sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', fmt='.3f', center=0.7)
plt.title('Baseline Model Performance (Raw Data Only)')
plt.tight_layout()
plt.savefig('./data/baseline_heatmap.png')
print("已儲存 Baseline 熱力圖: ./data/baseline_heatmap.png")