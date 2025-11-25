import pandas as pd

df = pd.read_csv("./data/test_ver2.csv")

row_count = 1000000
for chunk in pd.read_csv("./data/test_ver2.csv", chunksize=row_count): 
    chunk.info() # 處理它