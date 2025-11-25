import pandas as pd
import numpy as np

# ==========================================
# 1. 設定與讀取
# ==========================================
INPUT_FILE = './data/ready_for_pca_6to9.csv' 
CLUSTER_FILE = './data/august_data_with_clusters.csv'
OUTPUT_TRAIN_FILE = './data/train_set_for_step3.csv'

print("Step 1: 讀取原始資料與分群標籤...")
df = pd.read_csv(INPUT_FILE)
df_clusters = pd.read_csv(CLUSTER_FILE)

# 確保 Key 是整數
df['ncodpers'] = df['ncodpers'].astype(int)
df_clusters['ncodpers'] = df_clusters['ncodpers'].astype(int)

# ==========================================
# 2. 建立訓練主表 (以 8 月為基準)
# ==========================================
print("Step 2: 建立訓練主表 (Base: 2015-08-28)...")
train_df = df[df['fecha_dato'] == '2015-08-28'].copy()

# ==========================================
# 3. 合併 Cluster Label (來自 Step 1)
# ==========================================
print("Step 3: 合併客戶分群標籤...")
train_df = pd.merge(train_df, df_clusters[['ncodpers', 'cluster_label']], on='ncodpers', how='left')
# 填補未分群者 (如新戶)
train_df['cluster_label'] = train_df['cluster_label'].fillna(-1).astype(int)

# ==========================================
# 4. 建立組合特徵 (來自 Step 2 洞察)
# ==========================================
print("Step 4: 建立商品組合特徵 (Bundle Features)...")

# [自動生成的規則特徵 - 去重整理版]

# --- 針對 薪資帳戶 (cno) ---
# 強力推薦: 有薪資/退休金 -> 推薦薪資戶
train_df['rule_nomina_nom_pens_TO_cno'] = ((train_df['ind_nomina_ult1'] == 1) & (train_df['ind_nom_pens_ult1'] == 1) & (train_df['ind_cno_fin_ult1'] == 0)).astype(int)
train_df['rule_nomina_TO_cno'] = ((train_df['ind_nomina_ult1'] == 1) & (train_df['ind_cno_fin_ult1'] == 0)).astype(int)
train_df['rule_nom_pens_TO_cno'] = ((train_df['ind_nom_pens_ult1'] == 1) & (train_df['ind_cno_fin_ult1'] == 0)).astype(int)

# --- 針對 退休金 (nom_pens) ---
train_df['rule_recibo_cno_TO_nom_pens'] = ((train_df['ind_recibo_ult1'] == 1) & (train_df['ind_cno_fin_ult1'] == 1) & (train_df['ind_nom_pens_ult1'] == 0)).astype(int)
train_df['rule_reca_recibo_cno_TO_nom_pens'] = ((train_df['ind_reca_fin_ult1'] == 1) & (train_df['ind_recibo_ult1'] == 1) & (train_df['ind_cno_fin_ult1'] == 1) & (train_df['ind_nom_pens_ult1'] == 0)).astype(int)

# --- 針對 薪資 (nomina) ---
train_df['rule_recibo_cno_TO_nomina'] = ((train_df['ind_recibo_ult1'] == 1) & (train_df['ind_cno_fin_ult1'] == 1) & (train_df['ind_nomina_ult1'] == 0)).astype(int)

# --- 針對 信用卡 (tjcr) ---
train_df['rule_recibo_ecue_cno_TO_tjcr'] = ((train_df['ind_recibo_ult1'] == 1) & (train_df['ind_ecue_fin_ult1'] == 1) & (train_df['ind_cno_fin_ult1'] == 1) & (train_df['ind_tjcr_fin_ult1'] == 0)).astype(int)
train_df['rule_nom_pens_ecue_TO_tjcr'] = ((train_df['ind_nom_pens_ult1'] == 1) & (train_df['ind_ecue_fin_ult1'] == 1) & (train_df['ind_tjcr_fin_ult1'] == 0)).astype(int)
train_df['rule_ecue_cno_TO_tjcr'] = ((train_df['ind_ecue_fin_ult1'] == 1) & (train_df['ind_cno_fin_ult1'] == 1) & (train_df['ind_tjcr_fin_ult1'] == 0)).astype(int)

# --- 針對 帳單代扣 (recibo) ---
train_df['rule_reca_nomina_cno_TO_recibo'] = ((train_df['ind_reca_fin_ult1'] == 1) & (train_df['ind_nomina_ult1'] == 1) & (train_df['ind_cno_fin_ult1'] == 1) & (train_df['ind_recibo_ult1'] == 0)).astype(int)
train_df['rule_reca_nomina_nom_pens_cno_TO_recibo'] = ((train_df['ind_reca_fin_ult1'] == 1) & (train_df['ind_nomina_ult1'] == 1) & (train_df['ind_nom_pens_ult1'] == 1) & (train_df['ind_cno_fin_ult1'] == 1) & (train_df['ind_recibo_ult1'] == 0)).astype(int)

# --- 針對 定期存款 (ctpp) ---
train_df['rule_recibo_cno_TO_ctpp'] = ((train_df['ind_recibo_ult1'] == 1) & (train_df['ind_cno_fin_ult1'] == 1) & (train_df['ind_ctpp_fin_ult1'] == 0)).astype(int)
train_df['rule_nom_pens_TO_ctpp'] = ((train_df['ind_nom_pens_ult1'] == 1) & (train_df['ind_ctpp_fin_ult1'] == 0)).astype(int)

# --- 針對 電子帳戶 (ecue) ---
train_df['rule_tjcr_nom_pens_TO_ecue'] = ((train_df['ind_tjcr_fin_ult1'] == 1) & (train_df['ind_nom_pens_ult1'] == 1) & (train_df['ind_ecue_fin_ult1'] == 0)).astype(int)
train_df['rule_tjcr_recibo_cno_TO_ecue'] = ((train_df['ind_tjcr_fin_ult1'] == 1) & (train_df['ind_recibo_ult1'] == 1) & (train_df['ind_cno_fin_ult1'] == 1) & (train_df['ind_ecue_fin_ult1'] == 0)).astype(int)
train_df['rule_tjcr_recibo_TO_ecue'] = ((train_df['ind_tjcr_fin_ult1'] == 1) & (train_df['ind_recibo_ult1'] == 1) & (train_df['ind_ecue_fin_ult1'] == 0)).astype(int)

# --- 針對 中期存款 (ctop) ---
train_df['rule_tjcr_TO_ctop'] = ((train_df['ind_tjcr_fin_ult1'] == 1) & (train_df['ind_ctop_fin_ult1'] == 0)).astype(int)
train_df['rule_dela_TO_ctop'] = ((train_df['ind_dela_fin_ult1'] == 1) & (train_df['ind_ctop_fin_ult1'] == 0)).astype(int)
train_df['rule_reca_TO_ctop'] = ((train_df['ind_reca_fin_ult1'] == 1) & (train_df['ind_ctop_fin_ult1'] == 0)).astype(int)

# --- 針對 稅務 (reca) ---
train_df['rule_tjcr_cno_TO_reca'] = ((train_df['ind_tjcr_fin_ult1'] == 1) & (train_df['ind_cno_fin_ult1'] == 1) & (train_df['ind_reca_fin_ult1'] == 0)).astype(int)
train_df['rule_tjcr_recibo_TO_reca'] = ((train_df['ind_tjcr_fin_ult1'] == 1) & (train_df['ind_recibo_ult1'] == 1) & (train_df['ind_reca_fin_ult1'] == 0)).astype(int)
train_df['rule_cno_TO_reca'] = ((train_df['ind_cno_fin_ult1'] == 1) & (train_df['ind_reca_fin_ult1'] == 0)).astype(int)

# --- 其他 (dela, cco, fond, valo) ---
train_df['rule_ecue_TO_dela'] = ((train_df['ind_ecue_fin_ult1'] == 1) & (train_df['ind_dela_fin_ult1'] == 0)).astype(int)
train_df['rule_dela_TO_cco'] = ((train_df['ind_dela_fin_ult1'] == 1) & (train_df['ind_cco_fin_ult1'] == 0)).astype(int)
train_df['rule_recibo_TO_dela'] = ((train_df['ind_recibo_ult1'] == 1) & (train_df['ind_dela_fin_ult1'] == 0)).astype(int)
train_df['rule_ctop_TO_dela'] = ((train_df['ind_ctop_fin_ult1'] == 1) & (train_df['ind_dela_fin_ult1'] == 0)).astype(int)
train_df['rule_valo_TO_cco'] = ((train_df['ind_valo_fin_ult1'] == 1) & (train_df['ind_cco_fin_ult1'] == 0)).astype(int)
train_df['rule_recibo_TO_cco'] = ((train_df['ind_recibo_ult1'] == 1) & (train_df['ind_cco_fin_ult1'] == 0)).astype(int)
train_df['rule_cco_TO_fond'] = ((train_df['ind_cco_fin_ult1'] == 1) & (train_df['ind_fond_fin_ult1'] == 0)).astype(int)
train_df['rule_cco_TO_valo'] = ((train_df['ind_cco_fin_ult1'] == 1) & (train_df['ind_valo_fin_ult1'] == 0)).astype(int)

# ==========================================
# 5. 生成 Lag 特徵 (6月, 7月歷史)
# ==========================================
print("Step 5: 生成 Lag 特徵 (Merge 6月, 7月數據)...")
cols_to_lag = [c for c in df.columns if 'ind_' in c and 'ult1' in c]

for lag_month, lag_name in [('2015-07-28', 'lag2'), ('2015-06-28', 'lag3')]:
    lag_df = df[df['fecha_dato'] == lag_month][['ncodpers'] + cols_to_lag].copy()
    lag_df.columns = ['ncodpers'] + [f"{col}_{lag_name}" for col in cols_to_lag]
    
    train_df = pd.merge(train_df, lag_df, on='ncodpers', how='left')

# 填補 Lag 缺失 (沒 Merge 到代表當時沒資料，視為 0)
train_df.fillna(0, inplace=True)

# ==========================================
# 6. 生成預測目標 Label (9月新增購買)
# ==========================================
print("Step 6: 生成預測目標 Label (Target: 2015-09-28)...")

# 取出 9 月資料
df_sep = df[df['fecha_dato'] == '2015-09-28'][['ncodpers'] + cols_to_lag].copy()
df_sep.columns = ['ncodpers'] + [f"{col}_target" for col in cols_to_lag]

train_df = pd.merge(train_df, df_sep, on='ncodpers', how='left')

# 計算 Label
for col in cols_to_lag:
    label_col = f"label_{col}"
    target_col = f"{col}_target"
    
    # 9月缺失視為 0
    train_df[target_col] = train_df[target_col].fillna(0)
    
    # 新增購買邏輯: 9月有 (1) 且 8月沒有 (0)
    train_df[label_col] = ((train_df[target_col] == 1) & (train_df[col] == 0)).astype(int)
    
    # 刪除暫時欄位
    train_df.drop(columns=[target_col], inplace=True)

# 移除日期欄位
train_df.drop(columns=['fecha_dato'], inplace=True, errors='ignore')

# ==========================================
# 7. 存檔
# ==========================================
print(f"處理完成！最終訓練集維度: {train_df.shape}")
train_df.to_csv(OUTPUT_TRAIN_FILE, index=False)
print(f"檔案已儲存至: {OUTPUT_TRAIN_FILE}")