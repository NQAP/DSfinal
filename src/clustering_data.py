import pandas as pd
import numpy as np
import gc
import json  # 新增：用於儲存映射檔案
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ==========================================
# 1. 設定與讀取
# ==========================================
INPUT_FILE = './prime/train_ver2.csv'
OUTPUT_FILE = './data/ready_for_pca_6to9.csv'
MAPPING_FILE = './data/encoding_mapping.json' # 新增：LabelEncoder 映射檔
SCALER_FILE = './data/scaler_params.json'     # 新增：Scaler 參數檔

# 指定日期
DATES_NEEDED = ['2015-06-28', '2015-07-28', '2015-08-28', '2015-09-28']

# 基礎型態設定 (與之前相同)
dtype_dict = {
    'ind_nuevo': 'float32', 'indrel': 'float32', 'age': 'object', 'antiguedad': 'object',
    'renta': 'float32', 'ind_actividad_cliente': 'float32', 'sexo': 'category',
    'segmento': 'category', 'pais_residencia': 'category', 'nomprov': 'object',
    'canal_entrada': 'category'
}
product_cols = [
    'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
    'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
    'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
    'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
    'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
    'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1'
]
for col in product_cols:
    dtype_dict[col] = 'float32'

print("Step 1: 讀取 6~9 月資料...")
chunks = []
for chunk in pd.read_csv(INPUT_FILE, chunksize=100000, dtype=dtype_dict, low_memory=False):
    mask = chunk['fecha_dato'].isin(DATES_NEEDED)
    if mask.any():
        chunks.append(chunk[mask])

df = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

# ==========================================
# 2. 資料清洗 (Cleaning)
# ==========================================
print("Step 2: 執行資料清洗...")
# (清洗邏輯與之前完全相同，為節省篇幅簡寫，實際執行請保留完整的清洗代碼)
df.drop(columns=['conyuemp', 'ult_fec_cli_1t', 'tipodom', 'cod_prov'], errors='ignore', inplace=True)
cols_ghost = ['ind_empleado', 'fecha_alta', 'pais_residencia', 'indrel', 'ind_nuevo', 'indresi', 'indext', 'indfall', 'ind_actividad_cliente']
df.dropna(subset=cols_ghost, how='all', inplace=True)

df['fecha_alta'] = pd.to_datetime(df['fecha_alta'], errors='coerce')
df['fecha_dato'] = pd.to_datetime(df['fecha_dato'], errors='coerce')
df['antiguedad'] = pd.to_numeric(df['antiguedad'], errors='coerce')
mask_fix = (df['antiguedad'].isnull()) | (df['antiguedad'] < 0)
if mask_fix.any():
    months_diff = (df['fecha_dato'].dt.year - df['fecha_alta'].dt.year) * 12 + (df['fecha_dato'].dt.month - df['fecha_alta'].dt.month)
    df.loc[mask_fix, 'antiguedad'] = months_diff.loc[mask_fix]
df['antiguedad'] = df['antiguedad'].fillna(0)

df['month_joined'] = df['fecha_alta'].dt.month.fillna(-1).astype(int)
df.drop(columns=['fecha_alta'], inplace=True)
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['age'] = df['age'].fillna(df['age'].median())
print(df['age'].describe())
df['nomprov'] = df['nomprov'].fillna('UNKNOWN')
df['renta'] = df['renta'].fillna(df.groupby('nomprov')['renta'].transform('median'))
df['renta'] = df['renta'].fillna(df['renta'].median())
df['ind_nomina_ult1'] = df['ind_nomina_ult1'].fillna(0)
df['ind_nom_pens_ult1'] = df['ind_nom_pens_ult1'].fillna(0)

fill_mode_cols = ['segmento', 'indrel_1mes', 'tiprel_1mes', 'canal_entrada', 'sexo']
for col in fill_mode_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])
        if col == 'indrel_1mes':
            df[col] = df[col].astype(str).str.replace('.0', '', regex=False)

if 'indrel' in df.columns and df['indrel'].nunique() <= 1:
    df.drop(columns=['indrel'], inplace=True)

# ==========================================
# 2.5 極端值處理 (Outlier Capping)
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
# 3. 特徵預處理與儲存映射 (Encoding & Saving)
# ==========================================
print("Step 3: 執行 Label Encoding 與 Scaling 並儲存 JSON...")

cat_cols = ['sexo', 'ind_empleado', 'pais_residencia', 'nomprov', 'segmento', 'canal_entrada', 'indrel_1mes', 'tiprel_1mes']
num_cols = ['age', 'renta', 'antiguedad', 'month_joined']

# 確保欄位存在
cat_cols = [c for c in cat_cols if c in df.columns]
num_cols = [c for c in num_cols if c in df.columns]

# --- A. 處理 Label Encoding 並儲存 JSON ---
encoding_mapping = {}

for col in cat_cols:
    le = LabelEncoder()
    # 轉字串
    df[col] = df[col].astype(str)
    # 擬合並轉換
    df[col] = le.fit_transform(df[col])
    
    # 將 class 與 int 的對應關係存入字典
    # 注意：json 不支援 numpy int，需轉為 Python int
    # 結構: {'Madrid': 28, 'Barcelona': 8, ...}
    mapping = {str(label): int(idx) for idx, label in enumerate(le.classes_)}
    encoding_mapping[col] = mapping

# 儲存 Encoding JSON
with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
    json.dump(encoding_mapping, f, ensure_ascii=False, indent=4)
print(f"   - Label Encoding 映射已存至: {MAPPING_FILE}")


# --- B. 處理 Scaler 並儲存參數 ---
scaler = StandardScaler()
# 擬合並轉換
df[num_cols] = scaler.fit_transform(df[num_cols])

# 提取 Mean 和 Std (Scale)
scaler_params = {}
for i, col in enumerate(num_cols):
    scaler_params[col] = {
        'mean': float(scaler.mean_[i]),   # 轉為 python float 以利 json 儲存
        'std': float(scaler.scale_[i])    # scale_ 屬性即為標準差
    }

# 儲存 Scaler JSON
with open(SCALER_FILE, 'w', encoding='utf-8') as f:
    json.dump(scaler_params, f, ensure_ascii=False, indent=4)
print(f"   - Scaler 參數已存至: {SCALER_FILE}")


# ==========================================
# 4. 存檔
# ==========================================
print("Step 4: 儲存全數值化檔案...")
df.to_csv(OUTPUT_FILE, index=False)
print(f"最終處理好的資料已存至: {OUTPUT_FILE}")