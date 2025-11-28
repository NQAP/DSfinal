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
# 1. 資料準備
# ==========================================
INPUT_FILE = './data/train_set_for_step3.csv'
print("Step 1: 讀取資料...")
df = pd.read_csv(INPUT_FILE)

# 找出 label 和 feature
label_cols = [c for c in df.columns if c.startswith('label_')]
feature_cols = [c for c in df.columns if c not in label_cols and c != 'ncodpers' and not c.endswith('_target')]

# 這次我們使用全部資料 (包含沒買的人)，因為我們要訓練二元分類器
# 這是為了讓模型學習 "不買" 的特徵
X = df[feature_cols]
y_all = df[label_cols]

# 切分
X_train, X_test, y_train_all, y_test_all = train_test_split(X, y_all, test_size=0.2, random_state=42)

print(f"特徵數: {len(feature_cols)}, 待預測產品數: {len(label_cols)}")
print(f"訓練樣本數: {len(X_train)}")

# ==========================================
# 2. 定義模型工廠 (Model Factory)
# ==========================================
# 因為針對不同產品，資料不平衡比例不同，有些參數需動態調整
# 這裡定義基本架構

def get_models(pos_ratio):
    """
    根據正樣本比例，回傳四個設定好的模型
    pos_ratio: 正樣本權重 (scale_pos_weight)
    """
    models = {}
    
    # 1. Decision Tree
    models['DecisionTree'] = DecisionTreeClassifier(
        max_depth=8,
        class_weight='balanced', # 自動平衡
        random_state=42
    )
    
    # 2. Random Forest
    models['RandomForest'] = RandomForestClassifier(
        n_estimators=50, # 設小一點以節省時間
        max_depth=8,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    
    # 3. XGBoost
    models['XGBoost'] = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='binary:logistic',
        scale_pos_weight=pos_ratio, # 手動設定平衡
        n_jobs=-1,
        eval_metric='auc',
        tree_method='hist' # 加速
    )
    
    # 4. LightGBM
    models['LightGBM'] = lgb.LGBMClassifier(
        n_estimators=100,
        num_leaves=31,
        learning_rate=0.05,
        objective='binary',
        class_weight='balanced', # 自動平衡
        n_jobs=-1,
        verbose=-1
    )
    
    return models

# ==========================================
# 3. 開始大亂鬥 (Training Loop)
# ==========================================
print("\nStep 2: 開始 4 模型 x 24 產品 的全面評測...")
print("注意：這可能需要幾分鐘時間，請耐心等候...\n")

results_data = [] # 存結果: [Product, Model, AUC]

start_global = time.time()

for target_col in label_cols:
    product_name = target_col.replace('label_', '')
    
    # 取出該產品的 Y
    y_train = y_train_all[target_col]
    y_test = y_test_all[target_col]
    
    # 檢查樣本數 (如果正樣本太少，無法訓練)
    n_pos = y_train.sum()
    if n_pos < 10:
        print(f"Skipping {product_name}: 正樣本不足 ({n_pos})")
        continue
        
    # 計算不平衡比例 (給 XGBoost 用)
    ratio = (len(y_train) - n_pos) / n_pos
    
    # 取得四個模型
    current_models = get_models(ratio)
    
    print(f"Evaluating [{product_name}] (Positive: {n_pos})...")
    
    for model_name, clf in current_models.items():
        try:
            # 訓練
            clf.fit(X_train, y_train)
            
            # 預測機率
            y_pred_prob = clf.predict_proba(X_test)[:, 1]
            
            # 計算 AUC
            score = roc_auc_score(y_test, y_pred_prob)
            
        except Exception as e:
            print(f"   Error in {model_name}: {e}")
            score = 0.5 # 失敗視為隨機
            
        results_data.append({
            'Product': product_name,
            'Model': model_name,
            'AUC': score
        })

print(f"\n全流程結束！總耗時: {time.time() - start_global:.2f} 秒")

# ==========================================
# 4. 視覺化比較 (Heatmap)
# ==========================================
print("Step 3: 產出比較報告與熱力圖...")

df_res = pd.DataFrame(results_data)

# 儲存詳細數據
df_res.to_csv('./data/multi_model_auc_results.csv', index=False)

# 轉成矩陣格式 (Pivot) 以畫熱力圖
# Index=Product, Column=Model, Value=AUC
pivot_table = df_res.pivot(index='Product', columns='Model', values='AUC')

# 繪圖
plt.figure(figsize=(10, 12))
sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', fmt='.3f', center=0.7)
plt.title('Model Comparison by Product (AUC Score)')
plt.tight_layout()
plt.savefig('./data/multi_model_comparison_heatmap.png')

print("已儲存熱力圖至: ./data/multi_model_comparison_heatmap.png")

# 計算每個模型的平均 AUC
avg_scores = df_res.groupby('Model')['AUC'].mean().sort_values(ascending=False)
print("\n[模型平均表現排名]:")
print(avg_scores)

best_model = avg_scores.index[0]
print(f"\n結論：整體表現最好的模型是 {best_model}。")