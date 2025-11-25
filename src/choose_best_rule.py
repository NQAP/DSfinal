import pandas as pd

# ==========================================
# 1. 讀取驗證後的規則
# ==========================================
INPUT_VALIDATED_RULES = './data/validated_rules_jul_to_aug.csv'
OUTPUT_DIVERSE_RULES = './data/selected_diverse_rules.csv'

df_rules = pd.read_csv(INPUT_VALIDATED_RULES)

print(f"原始規則總數: {len(df_rules)}")

# ==========================================
# 2. 多樣化篩選策略 (Diversity Strategy)
# ==========================================
# 我們不想要前 20 名都是同一個產品。
# 策略：針對每一個 consequent (預測目標產品)，取出 Lift 最高的 Top 3 規則。

# 門檻設定 (稍微放寬，以免冷門產品選不到規則)
# Future Lift > 1.05 (只要比隨機準一點點就算有效)
# Support > 30 (只要有30個潛在客戶就算數)
filtered_rules = df_rules[
    (df_rules['future_lift'] > 1.05) & 
    (df_rules['support_jul'] > 30)
]

# 【關鍵動作】Group By Consequent + Head(3)
# 這會強迫每個產品都貢獻出它最強的規則
top_rules_per_product = filtered_rules.sort_values(
    by='future_lift', ascending=False
).groupby('consequent').head(3)

print(f"\n篩選後的規則總數: {len(top_rules_per_product)}")
print("包含的產品種類:")
print(top_rules_per_product['consequent'].value_counts())

# ==========================================
# 3. 自動生成 Step 3 特徵代碼
# ==========================================
print("\n=== 請將以下代碼貼入 05_feature_engineering.py 的 Step 4 區域 ===")
print("# [自動生成的 多樣化 規則特徵]")

for idx, row in top_rules_per_product.iterrows():
    import ast
    ants = ast.literal_eval(row['antecedents'])
    consequent = row['consequent']
    
    # 簡化名稱函數
    def simplify_name(col_name):
        return col_name.replace('ind_', '').replace('_ult1', '').replace('_fin', '')
    
    ant_names = "_".join([simplify_name(a) for a in ants])
    cons_name = simplify_name(consequent)
    feature_name = f"rule_{ant_names}_TO_{cons_name}"
    
    # 生成邏輯
    conditions = [f"(train_df['{ant}'] == 1)" for ant in ants]
    conditions.append(f"(train_df['{consequent}'] == 0)") # 關鍵：缺口分析
    condition_str = " & ".join(conditions)
    
    print(f"# Rule for {cons_name}: Lift={row['future_lift']:.2f}")
    print(f"train_df['{feature_name}'] = ({condition_str}).astype(int)")

# 存檔
top_rules_per_product.to_csv(OUTPUT_DIVERSE_RULES, index=False)