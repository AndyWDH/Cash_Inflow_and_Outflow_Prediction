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

def create_enhanced_features(df, include_target_cols=True):
    """创建增强的特征工程，模仿周期因子方法的核心思想"""
    df = df.copy()
    df['weekday'] = df['report_date'].dt.weekday
    df['day'] = df['report_date'].dt.day
    df['month'] = df['report_date'].dt.month
    
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
        # 使用训练数据的统计信息
        purchase_ma_7_mean, purchase_ma_7_std = df['purchase_ma_7'].mean(), df['purchase_ma_7'].std()
        redeem_ma_7_mean, redeem_ma_7_std = df['redeem_ma_7'].mean(), df['redeem_ma_7'].std()
        purchase_ma_14_mean, purchase_ma_14_std = df['purchase_ma_14'].mean(), df['purchase_ma_14'].std()
        redeem_ma_14_mean, redeem_ma_14_std = df['redeem_ma_14'].mean(), df['redeem_ma_14'].std()
        
        df['purchase_ma_7_scaled'] = (df['purchase_ma_7'] - purchase_ma_7_mean) / (purchase_ma_7_std + 1e-8)
        df['redeem_ma_7_scaled'] = (df['redeem_ma_7'] - redeem_ma_7_mean) / (redeem_ma_7_std + 1e-8)
        df['purchase_ma_14_scaled'] = (df['purchase_ma_14'] - purchase_ma_14_mean) / (purchase_ma_14_std + 1e-8)
        df['redeem_ma_14_scaled'] = (df['redeem_ma_14'] - redeem_ma_14_mean) / (redeem_ma_14_std + 1e-8)
        
        # 保存统计信息用于预测
        df.attrs['ma_stats'] = {
            'purchase_ma_7_mean': purchase_ma_7_mean,
            'purchase_ma_7_std': purchase_ma_7_std,
            'redeem_ma_7_mean': redeem_ma_7_mean,
            'redeem_ma_7_std': redeem_ma_7_std,
            'purchase_ma_14_mean': purchase_ma_14_mean,
            'purchase_ma_14_std': purchase_ma_14_std,
            'redeem_ma_14_mean': redeem_ma_14_mean,
            'redeem_ma_14_std': redeem_ma_14_std
        }
    else:
        # 当预测未来的日期时，使用默认值（从训练数据获取的统计信息）
        # 为未来日期设置移动平均值为训练数据的均值（标准化后为0）
        df['purchase_ma_7_scaled'] = 0
        df['redeem_ma_7_scaled'] = 0
        df['purchase_ma_14_scaled'] = 0
        df['redeem_ma_14_scaled'] = 0
    
    return df

def build_optimized_neural_network(input_dim):
    """构建优化的神经网络模型"""
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
    """准备训练数据"""
    # 筛选训练时间段的数据
    start_date = datetime(2014, 3, 1)
    end_date = datetime(2014, 8, 31)
    mask = (df['report_date'] >= start_date) & (df['report_date'] <= end_date)
    filtered_df = df.loc[mask].copy()
    
    # 按日期汇总数据
    daily_summary = filtered_df.groupby('report_date')[['total_purchase_amt', 'total_redeem_amt']].sum().reset_index()
    
    # 对汇总后的数据进行特征提取
    daily_summary = create_enhanced_features(daily_summary, include_target_cols=True)
    
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

def predict_future_values_optimized_nn(model, scaler_X, scaler_y, future_dates, feature_cols):
    """使用优化的神经网络预测未来值"""
    # 创建未来日期的特征
    future_df = pd.DataFrame({
        'report_date': future_dates
    })
    future_df = create_enhanced_features(future_df, include_target_cols=False)
    
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
    
    # 应用周期性调整（参考周期因子方法的思想）
    result_df['weekday'] = pd.to_datetime(result_df['report_date']).dt.weekday
    result_df['day'] = pd.to_datetime(result_df['report_date']).dt.day
    
    return result_df[['report_date', 'total_purchase_amt', 'total_redeem_amt']]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'user_balance_table.csv')
    output_file = os.path.join(script_dir, 'optimized_nn_result.csv')

    try:
        # 加载数据
        df = load_and_preprocess_data(input_file)
        
        # 按日期汇总数据
        df_daily = df.groupby('report_date')[['total_purchase_amt', 'total_redeem_amt']].sum().reset_index()
        
        # 准备训练数据
        X, y, scaler_X, scaler_y, feature_cols = prepare_training_data(df_daily)
        
        print(f"训练数据形状: X={X.shape}, y={y.shape}")
        print(f"训练数据Y范围: [{y.min():.2f}, {y.max():.2f}]")
        
        # 构建并训练模型
        model = build_optimized_neural_network(X.shape[1])
        print("开始训练优化的神经网络...")
        
        # 添加早停机制和学习率调度
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.0001)
        
        history = model.fit(
            X, y,
            epochs=1000,
            batch_size=16,  # 减小batch size以增加训练稳定性
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # 生成未来30天预测
        end_date = datetime(2014, 8, 31)
        future_dates = pd.date_range(start=end_date + timedelta(days=1), periods=30, freq='D')
        
        result_df = predict_future_values_optimized_nn(model, scaler_X, scaler_y, future_dates, feature_cols)
        
        # 保存结果
        result_df.to_csv(output_file, header=False, index=False)
        print(f"优化的神经网络预测结果已保存到 {output_file}（不包含表头）")
        print(result_df.head(10))
        print(f"预测结果范围 - Purchase: [{result_df['total_purchase_amt'].min():.2f}, {result_df['total_purchase_amt'].max():.2f}], Redeem: [{result_df['total_redeem_amt'].min():.2f}, {result_df['total_redeem_amt'].max():.2f}]")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
    except Exception as e:
        print(f"处理过程中发生错误：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()