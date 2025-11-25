import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 引入四種模型
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# ==========================================
# 1. 資料準備 (Data Preparation)
# ==========================================
INPUT_FILE = './data/train_set_for_step3.csv'
print("Step 1: 讀取訓練資料...")
df = pd.read_csv(INPUT_FILE)

# 找出所有的 label 欄位 (預測目標)
label_cols = [c for c in df.columns if c.startswith('label_')]

# 找出所有的 feature 欄位 (排除 label 和 ID)
# 我們也要排除那些 '_target' 結尾的暫存欄位(如果有留下的話)
feature_cols = [c for c in df.columns if c not in label_cols and c != 'ncodpers' and not c.endswith('_target')]

print(f"特徵數量: {len(feature_cols)}")
print(f"特徵範例: {feature_cols[:5]} ... {feature_cols[-5:]}")

# --- [關鍵步驟] 轉換目標變數 (Multi-label to Multi-class) ---
# 我們的目標是預測 "買了什麼"。
# 為了簡化比較，我們只取 "有發生購買行為" (Row sum > 0) 的資料來訓練
# 這樣模型專注於學習 "產品之間的區別"

# 1. 篩選有購買的資料
df['total_new_products'] = df[label_cols].sum(axis=1)
df_active = df[df['total_new_products'] > 0].copy()

print(f"原始資料筆數: {len(df)}")
print(f"有新增購買行為的資料筆數 (用於訓練): {len(df_active)}")

# 2. 將 One-Hot Label 轉為 Single Column (0, 1, 2...)
# 如果一個人買多個，我們只取第一個 (為了簡化成多分類問題)
y_raw = df_active[label_cols].idxmax(axis=1)

# 使用 LabelEncoder 將 'label_ind_cco_...' 字串轉為數字 0, 1, 2
le_target = LabelEncoder()
y = le_target.fit_transform(y_raw)
X = df_active[feature_cols]

# 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"訓練集大小: {X_train.shape}, 測試集大小: {X_test.shape}")
print(f"預測目標類別數: {len(le_target.classes_)}")

# ==========================================
# 2. 模型定義與訓練 (Model Training)
# ==========================================
print("\nStep 2: 開始訓練四種模型...")

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1, 
        objective='multi:softmax', 
        num_class=len(le_target.classes_),
        n_jobs=-1,
        eval_metric='mlogloss'
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=200,      # 增加樹的數量
        num_leaves=31,         # LightGBM 的核心參數 (預設 31)
        # max_depth=-1,        # 建議不限制深度，讓它自由生長
        learning_rate=0.05,    # 降低學習率以獲得更穩定的結果
        objective='multiclass',
        num_class=len(le_target.classes_),
        class_weight='balanced', # 【關鍵】處理類別不平衡
        min_child_samples=20,  # 防止在小資料上過擬合
        n_jobs=-1,
        verbose=-1
    )
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    start_time = time.time()
    
    # 訓練
    model.fit(X_train, y_train)
    
    # 預測
    y_pred = model.predict(X_test)
    
    # 評估
    acc = accuracy_score(y_test, y_pred)
    elapsed_time = time.time() - start_time
    
    print(f"   -> Accuracy: {acc:.4f}")
    print(f"   -> Time: {elapsed_time:.2f} seconds")
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Time': elapsed_time,
        'Model_Obj': model # 存起來為了畫 Feature Importance
    })

# ==========================================
# 3. 結果比較與特徵重要性 (Evaluation)
# ==========================================
print("\nStep 3: 產出比較報告...")

# A. 準確度比較圖
results_df = pd.DataFrame(results)
plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x='Model', y='Accuracy', palette='viridis')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1.0)
plt.savefig('./data/model_comparison_accuracy.png')
print("已儲存準確度比較圖: ./data/model_comparison_accuracy.png")

# B. 特徵重要性 (以 XGBoost 為例，因為它通常最強)
print("\n檢查 XGBoost 的特徵重要性 (Top 15)...")
xgb_model = results[2]['Model_Obj'] # 取出 XGBoost
importances = xgb_model.feature_importances_
feature_names = X_train.columns

feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(10, 8))
sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='magma')
plt.title('Top 15 Feature Importance (XGBoost)')
plt.tight_layout()
plt.savefig('./data/feature_importance_xgb.png')
print("已儲存特徵重要性圖: ./data/feature_importance_xgb.png")

# 列出文字版
print(feat_imp_df)

print("\n流程結束！")