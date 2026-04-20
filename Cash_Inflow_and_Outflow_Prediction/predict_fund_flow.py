import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
# 构建文件的完整路径
file_path = os.path.join(script_dir, 'user_balance_table.csv')
df = pd.read_csv(file_path)
# 读取CSV文件
#df = pd.read_csv('user_balance_table.csv')

# 将report_date转换为日期格式
df['report_date'] = pd.to_datetime(df['report_date'], format='%Y%m%d')

# 筛选2014-03-01到2014-08-31的数据
start_date = datetime(2014, 3, 1)
end_date = datetime(2014, 8, 31)
mask = (df['report_date'] >= start_date) & (df['report_date'] <= end_date)
filtered_df = df.loc[mask]

# 按日期汇总total_purchase_amt和total_redeem_amt
daily_summary = filtered_df.groupby('report_date')[['total_purchase_amt', 'total_redeem_amt']].sum().reset_index()

# 为缺失的日期填充0
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
complete_df = pd.DataFrame({'report_date': date_range})
complete_df = pd.merge(complete_df, daily_summary, on='report_date', how='left')
complete_df = complete_df.fillna(0)

# 对total_purchase_amt和total_redeem_amt分别使用ARIMA(7,1,7)建模并预测
# 预测未来30天（2014-09-01到2014-09-30）
forecast_days = 30
future_dates = pd.date_range(start=end_date + timedelta(days=1), periods=forecast_days, freq='D')

result_data = {'report_date': future_dates, 'total_purchase_amt': [], 'total_redeem_amt': []}

# 对total_purchase_amt建模并预测
model_purchase = ARIMA(complete_df['total_purchase_amt'], order=(7,1,7))
fitted_purchase = model_purchase.fit()
forecast_purchase = fitted_purchase.forecast(steps=forecast_days)
result_data['total_purchase_amt'] = forecast_purchase.values

# 对total_redeem_amt建模并预测
model_redeem = ARIMA(complete_df['total_redeem_amt'], order=(7,1,7))
fitted_redeem = model_redeem.fit()
forecast_redeem = fitted_redeem.forecast(steps=forecast_days)
result_data['total_redeem_amt'] = forecast_redeem.values

# 创建结果DataFrame并保存为CSV，不包含表头
result_df = pd.DataFrame(result_data)
result_df['report_date'] = result_df['report_date'].dt.strftime('%Y%m%d')  # 转换日期格式为字符串
result_df.to_csv('result.csv', header=False, index=False)

print("预测结果已保存到 result.csv（不包含表头）")
print(result_df.head())