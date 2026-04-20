import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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

# 为缺失的日期填充0
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
complete_df = pd.DataFrame({'report_date': date_range})
complete_df = pd.merge(complete_df, daily_summary, on='report_date', how='left')
complete_df = complete_df.fillna(0)

# 准备LSTM模型的数据
def prepare_data_for_lstm(data, timesteps=60):
    X, y = [], []
    for i in range(timesteps, len(data)):
        X.append(data[i-timesteps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# 设置时间步长
timesteps = 60

# 对total_purchase_amt进行LSTM建模和预测
purchase_data = complete_df['total_purchase_amt'].values.reshape(-1, 1)

# 数据标准化
scaler_purchase = MinMaxScaler(feature_range=(0, 1))
scaled_purchase_data = scaler_purchase.fit_transform(purchase_data)

# 准备训练数据
X_purchase, y_purchase = prepare_data_for_lstm(scaled_purchase_data, timesteps)

# 重塑数据以适应LSTM输入格式
X_purchase = X_purchase.reshape((X_purchase.shape[0], X_purchase.shape[1], 1))

# 构建LSTM模型
model_purchase = Sequential()
model_purchase.add(LSTM(units=50, return_sequences=True, input_shape=(X_purchase.shape[1], 1)))
model_purchase.add(LSTM(units=50, return_sequences=False))
model_purchase.add(Dense(units=1))
model_purchase.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model_purchase.fit(X_purchase, y_purchase, epochs=50, batch_size=32, verbose=0)

# 对total_redeem_amt进行LSTM建模和预测
redeem_data = complete_df['total_redeem_amt'].values.reshape(-1, 1)

# 数据标准化
scaler_redeem = MinMaxScaler(feature_range=(0, 1))
scaled_redeem_data = scaler_redeem.fit_transform(redeem_data)

# 准备训练数据
X_redeem, y_redeem = prepare_data_for_lstm(scaled_redeem_data, timesteps)

# 重塑数据以适应LSTM输入格式
X_redeem = X_redeem.reshape((X_redeem.shape[0], X_redeem.shape[1], 1))

# 构建LSTM模型
model_redeem = Sequential()
model_redeem.add(LSTM(units=50, return_sequences=True, input_shape=(X_redeem.shape[1], 1)))
model_redeem.add(LSTM(units=50, return_sequences=False))
model_redeem.add(Dense(units=1))
model_redeem.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model_redeem.fit(X_redeem, y_redeem, epochs=50, batch_size=32, verbose=0)

# 预测未来30天的数据
def predict_future(model, last_sequence, scaler, steps, timesteps):
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(steps):
        # 预测下一个值
        next_pred = model.predict(current_sequence.reshape(1, timesteps, 1), verbose=0)
        predictions.append(next_pred[0, 0])
        
        # 更新序列：移除第一个元素，添加新预测值
        current_sequence = np.append(current_sequence[1:], next_pred[0, 0])
    
    # 反标准化
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    return predictions.flatten()

# 获取用于预测的最后序列
last_sequence_purchase = scaled_purchase_data[-timesteps:]
last_sequence_redeem = scaled_redeem_data[-timesteps:]

# 预测未来30天
forecast_days = 30
future_purchase = predict_future(model_purchase, last_sequence_purchase, scaler_purchase, forecast_days, timesteps)
future_redeem = predict_future(model_redeem, last_sequence_redeem, scaler_redeem, forecast_days, timesteps)

# 创建结果DataFrame
future_dates = pd.date_range(start=end_date + timedelta(days=1), periods=forecast_days, freq='D')
result_df = pd.DataFrame({
    'report_date': future_dates.strftime('%Y%m%d'),
    'total_purchase_amt': future_purchase,
    'total_redeem_amt': future_redeem
})

# 保存结果到CSV文件
result_df.to_csv('result.csv', index=False)

print("LSTM预测结果已保存到 result.csv")
print(result_df.head())