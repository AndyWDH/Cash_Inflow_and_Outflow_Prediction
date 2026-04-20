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

def create_advanced_features(df, include_target_cols=True):
    """创建更丰富的特征工程，模仿周期因子方法的核心思想"""
    df = df.copy()
    df['weekday'] = df['report_date'].dt.weekday
    df['day'] = df['report_date'].dt.day
    df['month'] = df['report_date'].dt.month
    df['quarter'] = df['report_date'].dt.quarter
    df['dayofyear'] = df['report_date'].dt.dayofyear
    
    # 创建更精确的周期性特征，采用类似周期因子的方式
    df['sin_weekday'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['cos_weekday'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['sin_day'] = np.sin(2 * np.pi * df['day'] / 31)
    df['cos_day'] = np.cos(2 * np.pi * df['day'] / 31)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 只有在训练数据时才计算移动平均特征，预测时跳过这一步
    if include_target_cols and 'total_purchase_amt' in df.columns and 'total_redeem_amt' in df.columns:
        # 添加移动平均特征
        df = df.sort_values('report_date').reset_index(drop=True)
        df['purchase_ma_7'] = df['total_purchase_amt'].rolling(window=7, min_periods=1).mean()
        df['redeem_ma_7'] = df['total_redeem_amt'].rolling(window=7, min_periods=1).mean()
        df['purchase_ma_14'] = df['total_purchase_amt'].rolling(window=14, min_periods=1).mean()
        df['redeem_ma_14'] = df['total_redeem_amt'].rolling(window=14, min_periods=1).mean()
        
        # 将移动平均也进行标准化处理
        scaler_ma = StandardScaler()
        df['purchase_ma_7_scaled'] = scaler_ma.fit_transform(df[['purchase_ma_7']]).flatten()
        df['redeem_ma_7_scaled'] = scaler_ma.fit_transform(df[['redeem_ma_7']]).flatten()
        df['purchase_ma_14_scaled'] = scaler_ma.fit_transform(df[['purchase_ma_14']]).flatten()
        df['redeem_ma_14_scaled'] = scaler_ma.fit_transform(df[['redeem_ma_14']]).flatten()
    else:
        # 当预测未来的日期时，我们用固定的值填充这些字段
        df['purchase_ma_7_scaled'] = 0
        df['redeem_ma_7_scaled'] = 0
        df['purchase_ma_14_scaled'] = 0
        df['redeem_ma_14_scaled'] = 0
    
    return df

def build_improved_neural_network(input_dim):
    """构建改进的神经网络模型"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(2)  # 输出purchase和redeem两个值
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='mse', 
                  metrics=['mae'])
    return model

def prepare_training_data(df):
    """准备训练数据，更加注重时间序列特性"""
    # 筛选训练时间段的数据
    start_date = datetime(2014, 3, 1)
    end_date = datetime(2014, 8, 31)
    mask = (df['report_date'] >= start_date) & (df['report_date'] <= end_date)
    filtered_df = df.loc[mask].copy()
    
    # 按日期汇总数据（因为原始数据可能是按用户汇总的）
    daily_summary = filtered_df.groupby('report_date')[['total_purchase_amt', 'total_redeem_amt']].sum().reset_index()
    
    # 合并特征
    daily_summary = daily_summary.merge(
        filtered_df[['report_date', 'sin_weekday', 'cos_weekday', 'sin_day', 'cos_day', 'sin_month', 'cos_month',
                     'purchase_ma_7_scaled', 'redeem_ma_7_scaled', 'purchase_ma_14_scaled', 'redeem_ma_14_scaled']].drop_duplicates(subset=['report_date']),
        on='report_date',
        how='left'
    )
    
    # 特征工程
    daily_summary = create_advanced_features(daily_summary)
    
    # 选择特征列
    feature_cols = [
        'sin_weekday', 'cos_weekday', 
        'sin_day', 'cos_day', 
        'sin_month', 'cos_month',
        'purchase_ma_7_scaled', 'redeem_ma_7_scaled',
        'purchase_ma_14_scaled', 'redeem_ma_14_scaled'
    ]
    
    # 确保没有缺失值
    daily_summary = daily_summary.dropna()
    
    X = daily_summary[feature_cols].values
    y = daily_summary[['total_purchase_amt', 'total_redeem_amt']].values
    
    # 标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    return X_scaled, y_scaled, scaler_X, scaler_y, feature_cols

def predict_future_values_improved_nn(model, scaler_X, scaler_y, future_dates, historical_df):
    """使用改进的神经网络预测未来值，基于历史周期模式"""
    # 创建未来日期的特征，参考周期因子模式
    future_df = pd.DataFrame({
        'report_date': future_dates
    })
    future_df = create_advanced_features(future_df)
    
    # 为未来的移动平均特征使用最近的历史值或者均值
    last_ma_7_purc = historical_df['purchase_ma_7_scaled'].iloc[-1] if 'purchase_ma_7_scaled' in historical_df.columns else 0
    last_ma_7_redeem = historical_df['redeem_ma_7_scaled'].iloc[-1] if 'redeem_ma_7_scaled' in historical_df.columns else 0
    last_ma_14_purc = historical_df['purchase_ma_14_scaled'].iloc[-1] if 'purchase_ma_14_scaled' in historical_df.columns else 0
    last_ma_14_redeem = historical_df['redeem_ma_14_scaled'].iloc[-1] if 'redeem_ma_14_scaled' in historical_df.columns else 0
    
    # 设置未来的移动平均值
    future_df['purchase_ma_7_scaled'] = last_ma_7_purc
    future_df['redeem_ma_7_scaled'] = last_ma_7_redeem
    future_df['purchase_ma_14_scaled'] = last_ma_14_purc
    future_df['redeem_ma_14_scaled'] = last_ma_14_redeem
    
    feature_cols = [
        'sin_weekday', 'cos_weekday', 
        'sin_day', 'cos_day', 
        'sin_month', 'cos_month',
        'purchase_ma_7_scaled', 'redeem_ma_7_scaled',
        'purchase_ma_14_scaled', 'redeem_ma_14_scaled'
    ]
    
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
    
    # 确保预测值为非负数
    result_df['total_purchase_amt'] = np.maximum(result_df['total_purchase_amt'], 0)
    result_df['total_redeem_amt'] = np.maximum(result_df['total_redeem_amt'], 0)
    
    return result_df

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'user_balance_table.csv')
    output_file = os.path.join(script_dir, 'improved_nn_result.csv')

    try:
        # 加载数据
        df = load_and_preprocess_data(input_file)
        
        # 按日期汇总数据
        df_daily = df.groupby('report_date')[['total_purchase_amt', 'total_redeem_amt']].sum().reset_index()
        
        # 对汇总后的数据进行特征提取
        df_daily = create_advanced_features(df_daily)
        
        # 准备训练数据
        X, y, scaler_X, scaler_y, feature_cols = prepare_training_data(df_daily)
        
        print(f"训练数据形状: X={X.shape}, y={y.shape}")
        
        # 构建并训练模型
        model = build_improved_neural_network(X.shape[1])
        print("开始训练改进的神经网络...")
        
        # 添加早停机制防止过拟合
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001)
        
        history = model.fit(
            X, y,
            epochs=500,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # 生成未来30天预测
        end_date = datetime(2014, 8, 31)
        future_dates = pd.date_range(start=end_date + timedelta(days=1), periods=30, freq='D')
        
        result_df = predict_future_values_improved_nn(model, scaler_X, scaler_y, future_dates, df_daily)
        
        # 保存结果
        result_df.to_csv(output_file, header=False, index=False)
        print(f"改进的神经网络预测结果已保存到 {output_file}（不包含表头）")
        print(result_df.head())
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
    except Exception as e:
        print(f"处理过程中发生错误：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()