import pandas as pd
import numpy as np
import gc  # 用於手動釋放記憶體

# ==========================================
# 1. 設定與參數
# ==========================================
FILE_PATH = './prime/train_ver2.csv'

# 指定需要的日期：6, 7, 8月做特徵，9月做預測目標
DATES_NEEDED = ['2015-06-28', '2015-07-28', '2015-08-28', '2015-09-28']

# 產品欄位列表 (24個)
PRODUCT_COLS = [
    'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
    'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
    'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
    'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
    'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
    'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
    'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
    'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1'
]

# 記憶體優化設定
DTYPE_DICT = {
    'ind_nuevo': 'float32', # 有NaN所以用float
    'indrel': 'float32',
    'age': 'object',       # 先讀成字串處理髒數據
    'antiguedad': 'object',
    'renta': 'float32',
    'ind_actividad_cliente': 'float32',
    'sexo': 'category',
    'segmento': 'category',
    'pais_residencia': 'category'
}
# 將產品欄位全部設為 int8 (或 float16 以防有 NaN)
for col in PRODUCT_COLS:
    DTYPE_DICT[col] = 'float32'

# ==========================================
# 2. 精準讀取資料 (Filtering)
# ==========================================
print("Step 1: 開始讀取 6~9 月的資料...")
chunks = []
# 使用 chunksize 分批讀取
for chunk in pd.read_csv(FILE_PATH, chunksize=100000, dtype=DTYPE_DICT, low_memory=False):
    # 篩選日期
    mask = chunk['fecha_dato'].isin(DATES_NEEDED)
    if mask.any():
        chunks.append(chunk[mask])

df = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

print(f"資料讀取完成，總筆數: {len(df)}")
print(f"包含日期: {df['fecha_dato'].unique()}")

# ==========================================
# 3. 資料清洗 (Data Cleaning)
# ==========================================
print("Step 2: 執行資料清洗與缺失值填補...")

# 3.1 刪除廢棄欄位
cols_to_drop = ['conyuemp', 'ult_fec_cli_1t', 'tipodom', 'cod_prov']
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# 3.2 刪除「幽靈客戶」 (9個關鍵特徵同時缺失)
cols_ghost = [
    'ind_empleado', 'fecha_alta', 'pais_residencia', 'indrel', 
    'ind_nuevo', 'indresi', 'indext', 'indfall', 'ind_actividad_cliente'
]
original_len = len(df)
df.dropna(subset=cols_ghost, how='all', inplace=True)
print(f"已刪除幽靈客戶: {original_len - len(df)} 筆")

# 3.3 修復 antiguedad (年資)
# 轉換日期格式
df['fecha_alta'] = pd.to_datetime(df['fecha_alta'], errors='coerce')
df['fecha_dato'] = pd.to_datetime(df['fecha_dato'], errors='coerce')

# 處理 antiguedad 的髒數據 (轉數值)
df['antiguedad'] = pd.to_numeric(df['antiguedad'], errors='coerce')

# 找出需要修復的行 (NaN 或 < 0)
mask_fix_anti = (df['antiguedad'].isnull()) | (df['antiguedad'] < 0)
# 計算月份差來填補
if mask_fix_anti.any():
    months_diff = (df['fecha_dato'].dt.year - df['fecha_alta'].dt.year) * 12 + \
                  (df['fecha_dato'].dt.month - df['fecha_alta'].dt.month)
    df.loc[mask_fix_anti, 'antiguedad'] = months_diff.loc[mask_fix_anti]

df['antiguedad'] = df['antiguedad'].fillna(0) # 還有空的就補0

# 3.4 提取加入月份後，刪除 fecha_alta
df['month_joined'] = df['fecha_alta'].dt.month.fillna(-1).astype(int)
df.drop(columns=['fecha_alta'], inplace=True)

# 3.5 填補 Renta (收入) - 依 nomprov 分組
# 先把 age 轉好，可能有助於進階填補 (這邊先依地區)
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['age'] = df['age'].fillna(df['age'].median())

print("正在依地區填補收入...")
df['renta'] = df['renta'].fillna(df.groupby('nomprov')['renta'].transform('median'))
# 兜底：如果該地區也沒資料，用全體中位數
df['renta'] = df['renta'].fillna(df['renta'].median())

# 3.6 填補其他欄位 (Mode 或 0)
# 薪資與退休金 -> 補 0
df['ind_nomina_ult1'] = df['ind_nomina_ult1'].fillna(0)
df['ind_nom_pens_ult1'] = df['ind_nom_pens_ult1'].fillna(0)

# 類別欄位 -> 補眾數
categorical_cols_fill_mode = ['segmento', 'indrel_1mes', 'tiprel_1mes', 'canal_entrada', 'sexo']
for col in categorical_cols_fill_mode:
    if col in df.columns:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        
        # 修正 indrel_1mes 格式問題 (1.0 vs 1)
        if col == 'indrel_1mes':
            df[col] = df[col].astype(str).str.replace('.0', '', regex=False)

df['nomprov'] = df['nomprov'].fillna('UNKNOWN')

# 3.7 刪除 indrel (如果確認全都是 1)
if 'indrel' in df.columns and df['indrel'].nunique() <= 1:
    df.drop(columns=['indrel'], inplace=True)

# ==========================================
# 3.5 極端值處理 (Outlier Capping)
# ==========================================
print("Step 2.5: 執行極端值處理...")

# 只針對連續數值型欄位
outlier_cols = ['renta', 'age', 'antiguedad']

for col in outlier_cols:
    if col in df.columns:
        # 計算統計數據
        median_val = df[col].median()
        std_val = df[col].std()
        p95 = df[col].quantile(0.95)
        p99 = df[col].quantile(0.99) # 建議改用 99%
        
        # 您的原始邏輯 (若是堅持要用，我幫您寫出來)
        # limit_user = min(median_val + 10 * std_val, p95)
        
        # 【推薦邏輯】：直接使用 99% 分位數做 Cap
        # 這樣能保留 Top 1% ~ Top 5% 之間的差異
        upper_limit = p99
        
        # 針對 age 做防呆 (例如不超過 100歲)
        if col == 'age':
            upper_limit = min(upper_limit, 100) 

        print(f"   - {col}: 上限設為 {upper_limit:.2f} (99th Percentile)")
        
        # 執行蓋帽 (Clip)
        # 只有大於 upper_limit 的值會被換掉，小於的保持原樣
        df[col] = df[col].clip(upper=upper_limit)

        # 下限處理 (Optional): 防止負數 (例如 age < 0)
        # df[col] = df[col].clip(lower=0)

# ==========================================
# 4. 資料重組 (Lag Feature Engineering)
# ==========================================
print("Step 3: 建立滯後特徵 (Lags) 與 預測目標 (Label)...")

# 4.1 切分月份
# 我們的目標是：用 6,7,8月 預測 9月
# Base (Lag 1): 2015-08-28 (這是我們的主要客戶狀態)
# History (Lag 2): 2015-07-28
# History (Lag 3): 2015-06-28
# Target (Label): 2015-09-28

def get_product_data(date_str, suffix):
    """只提取該月份的 ID 和 產品欄位"""
    temp = df[df['fecha_dato'] == date_str][['ncodpers'] + PRODUCT_COLS].copy()
    temp.columns = ['ncodpers'] + [col + suffix for col in PRODUCT_COLS]
    return temp

# 提取資料
df_lag3 = get_product_data('2015-06-28', '_lag3')
df_lag2 = get_product_data('2015-07-28', '_lag2')
df_target_products = get_product_data('2015-09-28', '_target')

# 主訓練集：取 8 月的所有資料 (包含客戶特徵 + 8月產品)
train_df = df[df['fecha_dato'] == '2015-08-28'].copy()

# 釋放原始大表
del df
gc.collect()

# 4.2 合併 (Merge)
print("正在合併 6月 與 7月 的歷史產品資訊...")
train_df = pd.merge(train_df, df_lag2, on='ncodpers', how='left')
train_df = pd.merge(train_df, df_lag3, on='ncodpers', how='left')

# 填補合併後的產品缺失 (沒合併到代表當時沒產品，補0)
for col in train_df.columns:
    if '_lag' in col:
        train_df[col] = train_df[col].fillna(0).astype('int8')

# 4.3 製作 Label (9月新增購買)
print("正在計算 9月 的新增購買 (Label)...")
train_df = pd.merge(train_df, df_target_products, on='ncodpers', how='left')

# 迴圈計算每個產品的 Label
for col in PRODUCT_COLS:
    target_col = col + '_target'
    
    # 如果 9 月沒有資料 (NaN)，視為 0 (沒買/流失)
    train_df[target_col] = train_df[target_col].fillna(0)
    
    # 定義新增購買: 9月有 (1) 且 8月沒有 (0)
    # 這裡會產生一個新的欄位 'label_ind_...'
    train_df[f'label_{col}'] = ((train_df[target_col] == 1) & (train_df[col] == 0)).astype('int8')
    
    # 刪除暫時的 target 欄位
    train_df.drop(columns=[target_col], inplace=True)

# ==========================================
# 5. 最終處理與存檔
# ==========================================
# 刪除不需要的日期欄位 (都已經是 8月了)
train_df.drop(columns=['fecha_dato'], inplace=True)

print("\n處理完成！")
print(f"訓練集維度: {train_df.shape}")
print("包含特徵範例: age, renta, month_joined...")
print("包含 Lag 範例: ind_cco_fin_ult1_lag2, ind_cco_fin_ult1_lag3...")
print("包含 Label 範例: label_ind_cco_fin_ult1, label_ind_cder_fin_ult1...")

total_nan = train_df.isnull().sum().sum()
print(f"\n所有缺失值處理完畢！目前資料集總缺失值數量: {total_nan}")

# 如果還有 NaN，列出是哪個欄位 (通常這時候應該是 0 了)
if total_nan > 0:
    print(train_df.columns[train_df.isnull().any()])

# 若要儲存
train_df.to_csv('./data/seasoning6to9.csv', index=False)