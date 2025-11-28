import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os

# ==========================================
# 1. 設定與讀取
# ==========================================
INPUT_FILE = './data/train_set_for_step3.csv'
OUTPUT_SUMMARY = './data/all_product_feature_importance_summary.csv'
IMG_DIR = './data/feature_plots/'

if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

print("Step 1: 讀取資料...")
df = pd.read_csv(INPUT_FILE)

# 準備特徵與標籤列表
label_cols = [c for c in df.columns if c.startswith('label_')]
feature_cols = [c for c in df.columns if c not in label_cols and c != 'ncodpers' and not c.endswith('_target')]

X = df[feature_cols]

print(f"總共將分析 {len(label_cols)} 個產品模型。")

# ==========================================
# 2. 迴圈訓練與提取重要性
# ==========================================
print("\nStep 2: 開始批量訓練與分析...")

summary_data = []

for target_col in label_cols:
    product_name = target_col.replace('label_', '')
    y = df[target_col]
    
    # 檢查樣本數 (少於 10 個正樣本就不練了)
    if y.sum() < 10:
        continue
        
    # 簡單切分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 計算權重
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    
    # 快速訓練 (為了分析特徵，樹不用太多，50棵夠了)
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        tree_method='hist',
        importance_type='gain'
    )
    
    model.fit(X_train, y_train)
    
    # 提取重要性
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1] # 降序排列
    
    # 紀錄 Top 3 特徵
    top_3_feats = []
    top_3_scores = []
    
    for i in range(3):
        if i < len(indices):
            idx = indices[i]
            top_3_feats.append(feature_cols[idx])
            top_3_scores.append(importances[idx])
    
    # 加入總表數據
    summary_data.append({
        'Product': product_name,
        'Top_1_Feature': top_3_feats[0],
        'Top_1_Score': f"{top_3_scores[0]:.4f}",
        'Top_2_Feature': top_3_feats[1],
        'Top_2_Score': f"{top_3_scores[1]:.4f}",
        'Top_3_Feature': top_3_feats[2],
        'Top_3_Score': f"{top_3_scores[2]:.4f}"
    })
    
    print(f"   -> {product_name}: Top 1 is [{top_3_feats[0]}]")

# ==========================================
# 3. 產出報告
# ==========================================
print("\nStep 3: 生成總結報告...")
summary_df = pd.DataFrame(summary_data)

# 存檔
summary_df.to_csv(OUTPUT_SUMMARY, index=False)
print(f"特徵重要性總表已儲存至: {OUTPUT_SUMMARY}")

# 顯示結果
print("\n[所有產品的關鍵特徵 Top 3]")
# 調整顯示格式
pd.set_option('display.max_columns', None)
