import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 引入四種模型
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
# from catboost import CatBoostClassifier # 備用，如果您之前有換這個的話

# ==========================================
# 1. 資料準備 (Baseline Data Preparation)
# ==========================================
# 直接使用清洗過但"未經特徵工程"的檔案
INPUT_FILE = './data/ready_for_pca_6to9.csv'
print("Step 1: 讀取基礎資料 (Control Group)...")
df = pd.read_csv(INPUT_FILE)

# 確保 ncodpers 是整數
df['ncodpers'] = df['ncodpers'].astype(int)

# -------------------------------------------------------
# 【關鍵差異】我們不做任何 Rule Merge 或 Cluster Merge
# -------------------------------------------------------
print("Step 2: 建立基礎訓練集 (僅含人口特徵 + 8月產品狀態)...")

# 以 8 月為基準
train_df = df[df['fecha_dato'] == '2015-08-28'].copy()

# 移除日期 (只保留當下狀態)
train_df.drop(columns=['fecha_dato'], inplace=True, errors='ignore')

# -------------------------------------------------------
# Step 3: 生成預測目標 (與實驗組完全相同)
# -------------------------------------------------------
print("Step 3: 生成預測目標 Label (Target: 9月新增購買)...")

cols_to_predict = [c for c in df.columns if 'ind_' in c and 'ult1' in c]

# 取出 9 月資料
df_sep = df[df['fecha_dato'] == '2015-09-28'][['ncodpers'] + cols_to_predict].copy()
df_sep.columns = ['ncodpers'] + [f"{col}_target" for col in cols_to_predict]

train_df = pd.merge(train_df, df_sep, on='ncodpers', how='left')

# 計算 Label
for col in cols_to_predict:
    label_col = f"label_{col}"
    target_col = f"{col}_target"
    
    train_df[target_col] = train_df[target_col].fillna(0)
    
    # 邏輯: 9月有 (1) 且 8月沒有 (0) -> 新增購買
    train_df[label_col] = ((train_df[target_col] == 1) & (train_df[col] == 0)).astype(int)
    
    train_df.drop(columns=[target_col], inplace=True)

# -------------------------------------------------------
# Step 4: 轉換為多分類問題 (Model Training)
# -------------------------------------------------------
print("Step 4: 準備訓練與測試資料...")

# 找出 Label 和 Feature
label_cols = [c for c in train_df.columns if c.startswith('label_')]
# 排除 label 和 ID
feature_cols = [c for c in train_df.columns if c not in label_cols and c != 'ncodpers']

print(f"使用的基礎特徵數 (Baseline Features): {len(feature_cols)}")
# 這裡應該只會有約 40-50 個基礎特徵 (人口統計 + 24個產品現狀)

# 篩選有購買行為的資料
train_df['total_new'] = train_df[label_cols].sum(axis=1)
df_active = train_df[train_df['total_new'] > 0].copy()

# 轉換 y
y_raw = df_active[label_cols].idxmax(axis=1)
le_target = LabelEncoder()
y = le_target.fit_transform(y_raw)
X = df_active[feature_cols]

# 切分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"訓練集大小: {X_train.shape}")

# ==========================================
# 5. 定義四種模型 (參數需與實驗組一致以示公平)
# ==========================================
print("\nStep 5: 開始訓練四種 Baseline 模型...")

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
    
    # 這裡使用優化過的 LightGBM 參數 (如果您之前有修過的話)
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.05,
        objective='multiclass',
        num_class=len(le_target.classes_),
        class_weight='balanced', 
        n_jobs=-1,
        verbose=-1
    )
}

results = []

for name, model in models.items():
    print(f"\nTraining Baseline: {name}...")
    start_time = time.time()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    elapsed_time = time.time() - start_time
    
    print(f"   -> Accuracy: {acc:.4f}")
    
    results.append({
        'Model': name,
        'Baseline_Accuracy': acc,
        'Time': elapsed_time
    })

# ==========================================
# 6. 結果視覺化
# ==========================================
print("\nStep 6: 產出 Baseline 報告...")

results_df = pd.DataFrame(results)

# 存檔以便後續合併比較
results_df.to_csv('./data/baseline_model_results.csv', index=False)

plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x='Model', y='Baseline_Accuracy', palette='gray')
plt.title('Baseline Model Accuracy (Raw Features Only)')
plt.ylim(0, 1.0)
for index, row in results_df.iterrows():
    plt.text(index, row.Baseline_Accuracy + 0.02, f"{row.Baseline_Accuracy:.4f}", color='black', ha="center")

plt.savefig('./data/baseline_accuracy_chart.png')
print("已儲存 Baseline 準確度圖表: ./data/baseline_accuracy_chart.png")
print("Baseline 訓練完成！")