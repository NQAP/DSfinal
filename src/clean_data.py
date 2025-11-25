import pandas as pd
import numpy as np

# 1. 讀取數據 (這裡只讀取部分欄位作為示範，實際需讀取更多)
# 使用 dtype 優化記憶體
dtype_dict = {
    'ncodpers': 'int32', 
    'age': 'str',  # 先讀成 str 因為可能有髒數據
    'ind_ahor_fin_ult1': 'int8', # 產品欄位省記憶體
    'renta': 'float32'
}

# 假設讀取 train_ver2.csv
df = pd.read_csv('./prime/train_ver2.csv', dtype=dtype_dict, nrows=100000) # 開發時先讀 10萬筆測試

# 2. 處理異常值與轉換型態
# Age: 轉數值，無法轉的變 NaN，然後填補
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['age'] = df['age'].fillna(df['age'].median()).astype('int16')

# Antiguedad (年資): 處理類似
df['antiguedad'] = pd.to_numeric(df['antiguedad'], errors='coerce')
df['antiguedad'] = df['antiguedad'].fillna(0)
df.loc[df['antiguedad'] < 0, 'antiguedad'] = 0 # 修正負數

# 3. 處理缺失值 (Renta)
# 簡單版：用全體中位數填補
df['renta'] = df['renta'].fillna(df['renta'].median())
# (進階版需用到 groupby('nomprov')['renta'].transform('median'))

# 4. 日期處理
df['fecha_dato'] = pd.to_datetime(df['fecha_dato'])
df['month'] = df['fecha_dato'].dt.month

# 5. 類別變數編碼 (Label Encoding 範例)
# 將性別轉為數字
df['sexo'] = df['sexo'].map({'H': 0, 'V': 1}).fillna(-1).astype('int8')

# 6. 產出 Dataset A (給分群用 - 取最後一個月)
last_month = df['fecha_dato'].max()
df_clustering = df[df['fecha_dato'] == last_month].copy()

# 儲存處理好的檔案
df_clustering.to_csv('./data/clean_data_for_clustering.csv', index=False)
df.to_csv('./data/clean_data_full.csv', index=False)

print("Step 0 資料處理完成！")
print(f"分群用資料集大小: {df_clustering.shape}")