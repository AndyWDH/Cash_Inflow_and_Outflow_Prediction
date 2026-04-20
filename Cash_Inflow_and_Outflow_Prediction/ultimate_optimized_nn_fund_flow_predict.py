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

def create_periodic_features(df):
    """创建更强的周期性特征，直接模拟周期因子方法"""
    df = df.copy()
    df['weekday'] = df['report_date'].dt.weekday
    df['day'] = df['report_date'].dt.day
    df['month'] = df['report_date'].dt.month
    
    # sin/cos编码 - 用于模型学习周期模式
    df['sin_weekday'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['cos_weekday'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['sin_day'] = np.sin(2 * np.pi * df['day'] / 31)
    df['cos_day'] = np.cos(2 * np.pi * df['day'] / 31)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

def calculate_periodic_factors(daily_summary):
    """计算周期因子，用于特征增强"""
    base_purchase_mean = daily_summary['total_purchase_amt'].mean()
    base_redeem_mean = daily_summary['total_redeem_amt'].mean()

    # 计算周期因子（使用字典映射提高查找效率）
    wday_purc_factors = daily_summary.groupby('weekday')['total_purchase_amt'].mean() / base_purchase_mean
    wday_redeem_factors = daily_summary.groupby('weekday')['total_redeem_amt'].mean() / base_redeem_mean
    day_purc_factors = daily_summary.groupby('day')['total_purchase_amt'].mean() / base_purchase_mean
    day_redeem_factors = daily_summary.groupby('day')['total_redeem_amt'].mean() / base_redeem_mean

    return {
        'base_purchase': base_purchase_mean,
        'base_redeem': base_redeem_mean,
        'wday_purc': wday_purc_factors.to_dict(),
        'wday_redeem': wday_redeem_factors.to_dict(),
        'day_purc': day_purc_factors.to_dict(),
        'day_redeem': day_redeem_factors.to_dict()
    }

def create_features_with_periodic_factors(df, periodic_factors):
    """创建特征，结合周期因子"""
    df = create_periodic_features(df)
    
    # 添加周期因子特征
    df['weekday'] = df['report_date'].dt.weekday
    df['day'] = df['report_date'].dt.day
    
    # 使用周期因子作为特征
    df['wday_purc_factor'] = df['weekday'].map(periodic_factors['wday_purc']).fillna(1.0)
    df['wday_redeem_factor'] = df['weekday'].map(periodic_factors['wday_redeem']).fillna(1.0)
    df['day_purc_factor'] = df['day'].map(periodic_factors['day_purc']).fillna(1.0)
    df['day_redeem_factor'] = df['day'].map(periodic_factors['day_redeem']).fillna(1.0)
    
    return df

def build_improved_model(input_dim):
    """构建改进的模型，更好地利用周期性特征"""
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    
    # 主干网络
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    # 分支输出：分别预测purchase和redeem
    purchase_branch = tf.keras.layers.Dense(64, activation='relu', name='purchase_branch')(x)
    purchase_branch = tf.keras.layers.Dense(32, activation='relu')(purchase_branch)
    purchase_output = tf.keras.layers.Dense(1, name='purchase_output')(purchase_branch)
    
    redeem_branch = tf.keras.layers.Dense(64, activation='relu', name='redeem_branch')(x)
    redeem_branch = tf.keras.layers.Dense(32, activation='relu')(redeem_branch)
    redeem_output = tf.keras.layers.Dense(1, name='redeem_output')(redeem_branch)
    
    model = tf.keras.Model(inputs=inputs, outputs=[purchase_output, redeem_output])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={'purchase_output': 'mse', 'redeem_output': 'mse'},
        metrics={'purchase_output': 'mae', 'redeem_output': 'mae'},
        loss_weights={'purchase_output': 1.0, 'redeem_output': 1.0}
    )
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
    
    # 对汇总后的数据应用时间特征
    daily_summary = create_periodic_features(daily_summary)
    
    # 计算周期因子
    periodic_factors = calculate_periodic_factors(daily_summary)
    
    # 创建特征（包含周期因子）
    daily_summary = create_features_with_periodic_factors(daily_summary, periodic_factors)
    
    # 选择特征列
    feature_cols = [
        'sin_weekday', 'cos_weekday', 
        'sin_day', 'cos_day', 
        'sin_month', 'cos_month',
        'wday_purc_factor', 'wday_redeem_factor',
        'day_purc_factor', 'day_redeem_factor'
    ]
    
    # 确保没有缺失值
    daily_summary = daily_summary.dropna()
    
    X = daily_summary[feature_cols].values
    y_purchase = daily_summary[['total_purchase_amt']].values
    y_redeem = daily_summary[['total_redeem_amt']].values
    
    # 标准化
    scaler_X = StandardScaler()
    scaler_y_purchase = StandardScaler()
    scaler_y_redeem = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_purchase_scaled = scaler_y_purchase.fit_transform(y_purchase)
    y_redeem_scaled = scaler_y_redeem.fit_transform(y_redeem)
    
    return X_scaled, y_purchase_scaled, y_redeem_scaled, scaler_X, scaler_y_purchase, scaler_y_redeem, feature_cols, periodic_factors

def predict_future_values_improved(model, scaler_X, scaler_y_purchase, scaler_y_redeem, future_dates, feature_cols, periodic_factors):
    """使用改进的模型预测未来值"""
    # 创建未来日期的特征
    future_df = pd.DataFrame({
        'report_date': future_dates
    })
    
    # 使用默认周期因子
    future_df['weekday'] = future_df['report_date'].dt.weekday
    future_df['day'] = future_df['report_date'].dt.day
    future_df['month'] = future_df['report_date'].dt.month
    
    future_df['sin_weekday'] = np.sin(2 * np.pi * future_df['weekday'] / 7)
    future_df['cos_weekday'] = np.cos(2 * np.pi * future_df['weekday'] / 7)
    future_df['sin_day'] = np.sin(2 * np.pi * future_df['day'] / 31)
    future_df['cos_day'] = np.cos(2 * np.pi * future_df['day'] / 31)
    future_df['sin_month'] = np.sin(2 * np.pi * future_df['month'] / 12)
    future_df['cos_month'] = np.cos(2 * np.pi * future_df['month'] / 12)
    
    # 使用周期因子
    future_df['wday_purc_factor'] = future_df['weekday'].map(periodic_factors['wday_purc']).fillna(1.0)
    future_df['wday_redeem_factor'] = future_df['weekday'].map(periodic_factors['wday_redeem']).fillna(1.0)
    future_df['day_purc_factor'] = future_df['day'].map(periodic_factors['day_purc']).fillna(1.0)
    future_df['day_redeem_factor'] = future_df['day'].map(periodic_factors['day_redeem']).fillna(1.0)
    
    X_future = future_df[feature_cols].values
    X_future_scaled = scaler_X.transform(X_future)
    
    # 预测
    predictions_scaled = model.predict(X_future_scaled)
    purchase_predictions_scaled = predictions_scaled[0]
    redeem_predictions_scaled = predictions_scaled[1]
    
    purchase_predictions = scaler_y_purchase.inverse_transform(purchase_predictions_scaled)
    redeem_predictions = scaler_y_redeem.inverse_transform(redeem_predictions_scaled)
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'report_date': future_dates.strftime('%Y%m%d'),
        'total_purchase_amt': purchase_predictions.flatten(),
        'total_redeem_amt': redeem_predictions.flatten()
    })
    
    # 确保预测值为非负数
    result_df['total_purchase_amt'] = np.maximum(result_df['total_purchase_amt'], 0)
    result_df['total_redeem_amt'] = np.maximum(result_df['total_redeem_amt'], 0)
    
    return result_df[['report_date', 'total_purchase_amt', 'total_redeem_amt']]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'user_balance_table.csv')
    output_file = os.path.join(script_dir, 'ultimate_optimized_nn_result.csv')

    try:
        # 加载数据
        df = load_and_preprocess_data(input_file)
        
        # 按日期汇总数据
        df_daily = df.groupby('report_date')[['total_purchase_amt', 'total_redeem_amt']].sum().reset_index()
        
        # 准备训练数据
        X, y_purchase, y_redeem, scaler_X, scaler_y_purchase, scaler_y_redeem, feature_cols, periodic_factors = prepare_training_data(df_daily)
        
        print(f"训练数据形状: X={X.shape}, y_purchase={y_purchase.shape}, y_redeem={y_redeem.shape}")
        print(f"训练数据统计 - Purchase: [{y_purchase.min():.2f}, {y_purchase.max():.2f}], Mean: {y_purchase.mean():.2f}")
        print(f"训练数据统计 - Redeem: [{y_redeem.min():.2f}, {y_redeem.max():.2f}], Mean: {y_redeem.mean():.2f}")
        
        # 构建并训练模型
        model = build_improved_model(X.shape[1])
        print("开始训练改进的神经网络...")
        
        # 添加早停机制和学习率调度
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=15, min_lr=0.0001)
        
        history = model.fit(
            X, 
            {'purchase_output': y_purchase, 'redeem_output': y_redeem},
            epochs=500,
            batch_size=16,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # 生成未来30天预测
        end_date = datetime(2014, 8, 31)
        future_dates = pd.date_range(start=end_date + timedelta(days=1), periods=30, freq='D')
        
        result_df = predict_future_values_improved(model, scaler_X, scaler_y_purchase, scaler_y_redeem, future_dates, feature_cols, periodic_factors)
        
        # 输出预测结果统计信息
        print(f"预测结果统计 - Purchase: [{result_df['total_purchase_amt'].min():.2f}, {result_df['total_purchase_amt'].max():.2f}], Mean: {result_df['total_purchase_amt'].mean():.2f}")
        print(f"预测结果统计 - Redeem: [{result_df['total_redeem_amt'].min():.2f}, {result_df['total_redeem_amt'].max():.2f}], Mean: {result_df['total_redeem_amt'].mean():.2f}")
        
        # 保存结果
        result_df.to_csv(output_file, header=False, index=False)
        print(f"终极优化的神经网络预测结果已保存到 {output_file}（不包含表头）")
        print("预测结果前10行:")
        print(result_df.head(10))
        
        # 输出每周数据以检查周期性
        result_df['date'] = pd.to_datetime(result_df['report_date'], format='%Y%m%d')
        result_df['weekday'] = result_df['date'].dt.weekday
        print("按星期的预测结果（前3周）:")
        print(result_df.head(21)[['report_date', 'weekday', 'total_purchase_amt', 'total_redeem_amt']].to_string(index=False))
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
    except Exception as e:
        print(f"处理过程中发生错误：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()