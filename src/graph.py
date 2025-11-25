import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# ==========================================
# 1. 讀取與處理
# ==========================================
INPUT_RULES = './data/association_rules.csv'

try:
    rules = pd.read_csv(INPUT_RULES)
    print(f"成功讀取 {len(rules)} 條規則。")
except FileNotFoundError:
    print("錯誤：找不到 association_rules.csv，請先執行上一步的分析程式。")
    exit()

# 由於 csv 讀進來後，antecedents 變成了字串 "frozenset({'Product_A'})"
# 我們需要把它清乾淨，變成單純的 "Product_A"
def clean_product_name(text):
    # 去除 frozenset({' 和 '}) 這些符號
    return text.replace("frozenset({'", "").replace("'})", "").replace("', '", "\n")

rules['antecedents_str'] = rules['antecedents'].apply(clean_product_name)
rules['consequents_str'] = rules['consequents'].apply(clean_product_name)

# ==========================================
# 2. 規則品質散佈圖 (Scatter Plot)
# ==========================================
print("繪製規則品質散佈圖...")
plt.figure(figsize=(12, 8))

# X=Support, Y=Confidence, Color=Lift, Size=Lift
sns.scatterplot(
    data=rules, 
    x="support", 
    y="confidence", 
    hue="lift", 
    size="lift",
    palette="viridis",
    sizes=(20, 200),
    alpha=0.7
)
plt.title("Association Rules: Support vs Confidence (Color=Lift)", fontsize=16)
plt.xlabel("Support (Popularity)", fontsize=12)
plt.ylabel("Confidence (Predictability)", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # Legend 放旁邊
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('./data/rules_scatter_plot.png')
print("散佈圖已儲存至 ./data/rules_scatter_plot.png")

# ==========================================
# 3. 網絡關聯圖 (Network Graph)
# ==========================================
print("繪製網絡關聯圖...")

# 過濾：如果規則太多，圖會變成一團毛球。我們只畫 Lift 最高的 Top 20 條規則
TOP_N = 20
top_rules = rules.sort_values(by='lift', ascending=False).head(TOP_N)

# 建立有向圖 (Directed Graph)
G = nx.DiGraph()

# 添加節點與邊
for i, row in top_rules.iterrows():
    src = row['antecedents_str']
    dst = row['consequents_str']
    weight = row['lift']
    
    # 加入邊，並把 Lift 當作權重
    G.add_edge(src, dst, weight=weight)

# 設定畫圖佈局
plt.figure(figsize=(14, 10))
# spring_layout 會自動把節點撐開，k 值越大分越開
pos = nx.spring_layout(G, k=2.0, seed=42) 

# 畫節點 (產品)
nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', alpha=0.9)

# 畫邊 (箭頭)，線條粗細代表 Lift 大小
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
# 正規化粗細，讓線條不要太粗
width_normalized = [(w / max(weights)) * 5 for w in weights]

nx.draw_networkx_edges(
    G, pos, 
    width=width_normalized, 
    edge_color='grey', 
    arrowstyle='->', 
    arrowsize=20, 
    connectionstyle="arc3,rad=0.1" # 讓線有點弧度，避免重疊
)

# 畫標籤 (產品名稱)
# 為了怕字疊在一起，稍微調整位置
nx.draw_networkx_labels(G, pos, font_size=9, font_family='sans-serif', font_weight='bold')

# 顯示 Lift 數值在線上 (選擇性)
edge_labels = {(u, v): f"Lift: {d['weight']:.2f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title(f"Top {TOP_N} Strongest Association Rules (Network Graph)", fontsize=16)
plt.axis('off') # 關閉座標軸
plt.tight_layout()
plt.savefig('./data/rules_network_graph.png')
print("網絡圖已儲存至 ./data/rules_network_graph.png")