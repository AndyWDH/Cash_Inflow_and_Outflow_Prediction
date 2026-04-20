import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 读取CSV文件
df = pd.read_csv('user_balance_table.csv')

# 将report_date转换为日期格式
df['report_date'] = pd.to_datetime(df['report_date'], format='%Y%m%d')

# 筛选2014-03-01到2014-08-31的数据
start_date = datetime(2014, 3, 1)
end_date = datetime(2014, 8, 31)
mask = (df['report_date'] >= start_date) & (df['report_date'] <= end_date)
filtered_df = df.loc[mask]

# 按日期汇总total_purchase_amt和total_redeem_amt
daily_summary = filtered_df.groupby('report_date')[['total_purchase_amt', 'total_redeem_amt']].sum().reset_index()

# 添加weekday和day列
daily_summary['weekday'] = daily_summary['report_date'].dt.weekday  # 0=Monday, 6=Sunday
daily_summary['day'] = daily_summary['report_date'].dt.day

# 计算基础平均值（去除空值）
base_purchase_mean = daily_summary['total_purchase_amt'].mean()
base_redeem_mean = daily_summary['total_redeem_amt'].mean()

# 计算weekday周期因子
weekday_purchase_factors = daily_summary.groupby('weekday')['total_purchase_amt'].mean() / base_purchase_mean
weekday_redeem_factors = daily_summary.groupby('weekday')['total_redeem_amt'].mean() / base_redeem_mean

# 计算day周期因子
day_purchase_factors = daily_summary.groupby('day')['total_purchase_amt'].mean() / base_purchase_mean
day_redeem_factors = daily_summary.groupby('day')['total_redeem_amt'].mean() / base_redeem_mean

# 生成未来30天的日期序列
future_dates = pd.date_range(start=end_date + timedelta(days=1), periods=30, freq='D')

# 创建结果DataFrame
result_df = pd.DataFrame({
    'report_date': future_dates.strftime('%Y%m%d'),
    'weekday': future_dates.weekday,
    'day': future_dates.day,
    'total_purchase_amt': 0.0,
    'total_redeem_amt': 0.0
})

# 应用周期因子进行预测
for idx, row in result_df.iterrows():
    # 获取对应的weekday和day
    weekday = row['weekday']
    day = row['day']
    
    # 计算周期因子（使用平均值作为基础预测）
    weekday_purchase_factor = weekday_purchase_factors.get(weekday, 1.0)
    day_purchase_factor = day_purchase_factors.get(day, 1.0)
    weekday_redeem_factor = weekday_redeem_factors.get(weekday, 1.0)
    day_redeem_factor = day_redeem_factors.get(day, 1.0)
    
    # 预测值 = 基础平均值 × weekday因子 × day因子
    purchase_pred = base_purchase_mean * weekday_purchase_factor * day_purchase_factor
    redeem_pred = base_redeem_mean * weekday_redeem_factor * day_redeem_factor
    
    result_df.at[idx, 'total_purchase_amt'] = purchase_pred
    result_df.at[idx, 'total_redeem_amt'] = redeem_pred

# 只保留需要的列
result_df = result_df[['report_date', 'total_purchase_amt', 'total_redeem_amt']]

# 保存结果到CSV文件
result_df.to_csv('result2.csv', index=False)

print("周期因子预测结果已保存到 result2.csv")
print(result_df.head())