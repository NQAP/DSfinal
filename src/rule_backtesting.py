import pandas as pd
import numpy as np
import ast # 用於安全地解析字串格式的集合

# ==========================================
# 1. 設定與讀取
# ==========================================
INPUT_DATA = './data/ready_for_pca_6to9.csv' 
INPUT_RULES = './data/association_rules.csv'
OUTPUT_VALIDATED = './data/validated_rules_jul_to_aug.csv'

print("1. 讀取資料中...")
try:
    df = pd.read_csv(INPUT_DATA)
    rules_df = pd.read_csv(INPUT_RULES)
    print(f"   - 資料集筆數: {len(df)}")
    print(f"   - 待驗證規則數: {len(rules_df)}")
except FileNotFoundError:
    print("錯誤：找不到輸入檔案，請確認路徑或先執行之前的步驟。")
    exit()

# 準備 7 月和 8 月的資料切片
# 為了加速運算，我們先把這兩個月的資料切出來
df_jul = df[df['fecha_dato'] == '2015-07-28'].set_index('ncodpers')
df_aug = df[df['fecha_dato'] == '2015-08-28'].set_index('ncodpers')

# 找出共同存在的客戶 (因為我們只能驗證那些兩個月都在的人)
valid_users = df_jul.index.intersection(df_aug.index)
df_jul = df_jul.loc[valid_users]
df_aug = df_aug.loc[valid_users]

print(f"   - 可用於回測的有效客戶數 (兩個月都在): {len(valid_users)}")

# ==========================================
# 2. 定義解析與計算函數
# ==========================================

def parse_frozenset(s):
    """
    將 CSV 中的 "frozenset({'a', 'b'})" 字串轉換為 Python list ['a', 'b']
    """
    try:
        # 簡單的字串處理
        s = s.replace("frozenset({", "").replace("})", "").replace("'", "")
        return [item.strip() for item in s.split(",")]
    except:
        return []

# 預先計算每個產品的 Baseline 轉換率 (全體平均購買率)
# 這樣不用在迴圈裡重複算，速度會快很多
print("2. 預先計算所有產品的 Baseline 轉化率...")
all_products = [c for c in df.columns if 'ind_' in c and 'ult1' in c]
baseline_rates = {}

for prod in all_products:
    # 分母：7月沒有該產品的人
    no_prod_jul = df_jul[df_jul[prod] == 0]
    if len(no_prod_jul) > 0:
        # 分子：這些人 8 月有了該產品
        # 使用 index 對齊來找 8 月狀態
        new_buyers = df_aug.loc[no_prod_jul.index][prod]
        conversion = new_buyers.sum()
        rate = conversion / len(no_prod_jul)
        baseline_rates[prod] = rate
    else:
        baseline_rates[prod] = 0.0

print("   - Baseline 計算完成。")

# ==========================================
# 3. 批量回測迴圈
# ==========================================
print("3. 開始批量回測每一條規則...")

results = []

for idx, row in rules_df.iterrows():
    # 解析規則
    antecedents = parse_frozenset(row['antecedents'])
    consequents = parse_frozenset(row['consequents'])
    
    # 為了簡化，我們主要驗證 "單一產品預測" (這是最常見的推薦場景)
    # 如果後項有多個產品，我們只取第一個來驗證，或者您可以視需求修改邏輯
    target_product = consequents[0] 
    
    # A. 鎖定目標群體 (7月有 Antecedents 且 沒有 Target Product)
    # 開始篩選
    mask = pd.Series(True, index=df_jul.index)
    
    # 必須擁有所有前項 (Antecedents)
    for ant in antecedents:
        if ant in df_jul.columns:
            mask = mask & (df_jul[ant] == 1)
            
    # 必須還沒擁有後項 (Target Product)
    if target_product in df_jul.columns:
        mask = mask & (df_jul[target_product] == 0)
    
    target_group = df_jul[mask]
    n_target = len(target_group)
    
    # 如果樣本太少，統計無意義，跳過
    if n_target < 50:
        continue
        
    # B. 計算實際轉化 (Real Conversion)
    # 看這群人 8 月有沒有買 Target Product
    actual_buyers = df_aug.loc[target_group.index][target_product].sum()
    real_rate = actual_buyers / n_target
    
    # C. 取得 Baseline
    base_rate = baseline_rates.get(target_product, 0)
    
    # D. 計算未來提升度 (Future Lift)
    future_lift = real_rate / base_rate if base_rate > 0 else 0
    
    # 記錄結果
    results.append({
        'rule_id': idx,
        'antecedents': str(antecedents),
        'consequent': target_product,
        'support_jul': n_target,           # 潛在客戶基數
        'converted_aug': actual_buyers,    # 實際購買人數
        'real_conversion_rate': real_rate, # 真實轉化率
        'baseline_rate': base_rate,        # 隨機轉化率
        'future_lift': future_lift,        # 未來提升度 (最重要指標)
        'original_lift': row['lift']       # 當初 Apriori 算的靜態 Lift
    })

# ==========================================
# 4. 整理與存檔
# ==========================================
results_df = pd.DataFrame(results)

# 排序：按照 "Future Lift" 由高到低排，找出預測力最強的規則
results_df.sort_values(by='future_lift', ascending=False, inplace=True)

print(f"\n回測完成！共驗證了 {len(results_df)} 條有效規則。")

# 存檔
results_df.to_csv(OUTPUT_VALIDATED, index=False)
print(f"完整驗證報告已儲存至: {OUTPUT_VALIDATED}")

# 顯示 Top 10 強力規則
print("\n=== Top 10 經時間驗證的強力規則 (Future Lift > 1) ===")
cols_to_show = ['antecedents', 'consequent', 'real_conversion_rate', 'baseline_rate', 'future_lift']
print(results_df[cols_to_show].head(10).to_string(index=False))

# 簡單洞察
top_rule = results_df.iloc[0]
print(f"\n[最佳規則洞察]:")
print(f"若客戶持有 {top_rule['antecedents']}，")
print(f"下個月購買 {top_rule['consequent']} 的機率是常人的 {top_rule['future_lift']:.2f} 倍！")