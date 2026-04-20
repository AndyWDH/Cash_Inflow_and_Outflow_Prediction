import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
# 构建文件的完整路径
file_path = os.path.join(script_dir, 'user_balance_table.csv')
df = pd.read_csv(file_path)
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取CSV文件
#df = pd.read_csv('user_balance_table.csv')

# 将report_date转换为日期格式
df['report_date'] = pd.to_datetime(df['report_date'], format='%Y%m%d')

# 按日期分组并对total_purchase_amt和total_redeem_amt求和
daily_summary = df.groupby('report_date')[['total_purchase_amt', 'total_redeem_amt']].sum().reset_index()

# 创建图表
plt.figure(figsize=(15, 8))
plt.plot(daily_summary['report_date'], daily_summary['total_purchase_amt'], label='申购总金额', linewidth=1.5)
plt.plot(daily_summary['report_date'], daily_summary['total_redeem_amt'], label='赎回总金额', linewidth=1.5)

# 设置图表标题和坐标轴标签
plt.title('资金流入流出走势图', fontsize=16)
plt.xlabel('日期', fontsize=12)
plt.ylabel('金额', fontsize=12)

# 设置x轴日期格式
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # 每2个月显示一个标签
plt.xticks(rotation=45)

# 添加图例
plt.legend()

# 调整布局
plt.tight_layout()

# 保存图表到文件
plt.savefig('fund_flow_chart.png', dpi=300, bbox_inches='tight')
print("资金流入流出走势图已保存为 fund_flow_chart.png")