import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def load_and_preprocess_data(filepath):
    """加载并预处理数据"""
    df = pd.read_csv(filepath)
    df['report_date'] = pd.to_datetime(df['report_date'], format='%Y%m%d')
    return df

def calculate_periodic_factors(daily_summary):
    """计算周期因子"""
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

def predict_future_values(factors, future_dates):
    """批量预测未来值"""
    result_df = pd.DataFrame({
        'report_date': future_dates.strftime('%Y%m%d'),
        'weekday': future_dates.weekday,
        'day': future_dates.day
    })

    # 向量化计算（使用 map 方法替代循环）
    wday_purc_map = [factors['wday_purc'].get(wd, 1.0) for wd in result_df['weekday']]
    day_purc_map = [factors['day_purc'].get(d, 1.0) for d in result_df['day']]
    wday_redeem_map = [factors['wday_redeem'].get(wd, 1.0) for wd in result_df['weekday']]
    day_redeem_map = [factors['day_redeem'].get(d, 1.0) for d in result_df['day']]

    result_df['total_purchase_amt'] = (
        factors['base_purchase'] * 
        pd.Series(wday_purc_map).values * 
        pd.Series(day_purc_map).values
    )
    result_df['total_redeem_amt'] = (
        factors['base_redeem'] * 
        pd.Series(wday_redeem_map).values * 
        pd.Series(day_redeem_map).values
    )

    return result_df[['report_date', 'total_purchase_amt', 'total_redeem_amt']]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'user_balance_table.csv')
    output_file = os.path.join(script_dir, 'result2.csv')

    # 数据筛选参数
    start_date, end_date = datetime(2014, 3, 1), datetime(2014, 8, 31)

    try:
        # 加载并筛选数据
        df = load_and_preprocess_data(input_file)
        mask = (df['report_date'] >= start_date) & (df['report_date'] <= end_date)
        filtered_df = df.loc[mask]

        # 按日期汇总
        daily_summary = filtered_df.groupby('report_date')[['total_purchase_amt', 'total_redeem_amt']].sum().reset_index()
        daily_summary['weekday'] = daily_summary['report_date'].dt.weekday
        daily_summary['day'] = daily_summary['report_date'].dt.day

        # 计算周期因子
        factors = calculate_periodic_factors(daily_summary)

        # 生成未来30天预测
        future_dates = pd.date_range(start=end_date + timedelta(days=1), periods=30, freq='D')
        result_df = predict_future_values(factors, future_dates)

        # 保存结果
        result_df.to_csv(output_file, header=False, index=False)
        print(f"周期因子预测结果已保存到 {output_file}（不包含表头）")
        print(result_df.head())

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
    except Exception as e:
        print(f"处理过程中发生错误：{e}")

if __name__ == "__main__":
    main()