import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import os

def load_and_preprocess_data(filepath):
    """加载并预处理数据"""
    df = pd.read_csv(filepath)
    df['report_date'] = pd.to_datetime(df['report_date'], format='%Y%m%d')
    return df

def create_features(df):
    """创建特征工程"""
    df = df.copy()
    df['weekday'] = df['report_date'].dt.weekday
    df['day'] = df['report_date'].dt.day
    df['month'] = df['report_date'].dt.month
    
    # 创建周期性特征
    df['sin_weekday'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['cos_weekday'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['sin_day'] = np.sin(2 * np.pi * df['day'] / 31)
    df['cos_day'] = np.cos(2 * np.pi * df['day'] / 31)
    
    return df

def build_neural_network(input_dim):
    """构建神经网络模型"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(2)  # 输出purchase和redeem两个值
    ])
    
    model.compile(optimizer='adam', 
                  loss='mse', 
                  metrics=['mae'])
    return model

def prepare_training_data(df):
    """准备训练数据"""
    # 筛选训练时间段的数据
    start_date = datetime(2014, 3, 1)
    end_date = datetime(2014, 8, 31)
    mask = (df['report_date'] >= start_date) & (df['report_date'] <= end_date)
    filtered_df = df.loc[mask].copy()
    
    # 特征工程
    filtered_df = create_features(filtered_df)
    
    # 选择特征列
    feature_cols = ['sin_weekday', 'cos_weekday', 'sin_day', 'cos_day', 'month']
    X = filtered_df[feature_cols].values
    y = filtered_df[['total_purchase_amt', 'total_redeem_amt']].values
    
    # 标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    return X_scaled, y_scaled, scaler_X, scaler_y, feature_cols

def predict_future_values_nn(model, scaler_X, scaler_y, future_dates, feature_cols):
    """使用神经网络预测未来值"""
    # 创建未来日期的特征
    future_df = pd.DataFrame({
        'report_date': future_dates
    })
    future_df = create_features(future_df)
    
    X_future = future_df[feature_cols].values
    X_future_scaled = scaler_X.transform(X_future)
    
    # 预测
    predictions_scaled = model.predict(X_future_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled)
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'report_date': future_dates.strftime('%Y%m%d'),
        'total_purchase_amt': predictions[:, 0],
        'total_redeem_amt': predictions[:, 1]
    })
    
    return result_df

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'user_balance_table.csv')
    output_file = os.path.join(script_dir, 'nn_result.csv')

    try:
        # 加载数据
        df = load_and_preprocess_data(input_file)
        
        # 准备训练数据
        X, y, scaler_X, scaler_y, feature_cols = prepare_training_data(df)
        
        # 构建并训练模型
        model = build_neural_network(X.shape[1])
        print("开始训练神经网络...")
        history = model.fit(
            X, y,
            epochs=500,
            batch_size=256,
            validation_split=0.2,
            verbose=1
        )
        
        # 生成未来30天预测
        end_date = datetime(2014, 8, 31)
        future_dates = pd.date_range(start=end_date + timedelta(days=1), periods=30, freq='D')
        
        result_df = predict_future_values_nn(model, scaler_X, scaler_y, future_dates, feature_cols)
        
        # 保存结果
        result_df.to_csv(output_file, header=False, index=False)
        print(f"神经网络预测结果已保存到 {output_file}（不包含表头）")
        print(result_df.head())
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
    except Exception as e:
        print(f"处理过程中发生错误：{e}")

if __name__ == "__main__":
    main()