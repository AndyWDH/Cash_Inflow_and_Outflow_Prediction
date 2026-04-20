import pandas as pd

# 读取CSV文件
df = pd.read_csv('user_balance_table.csv')

# 显示所有列名
print("Columns:")
print(df.columns.tolist())
print()
print("First 5 rows:")
print(df.head(5))