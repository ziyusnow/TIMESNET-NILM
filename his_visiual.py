import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# 读取CSV文件
file_path = 'UKDALE_1_train.csv'  # 确保文件路径正确
data = pd.read_csv(file_path)

# 显示数据的前几行，以便了解数据的结构
print(data.head())

# 假设CSV文件中的第一列是时间序列，第二列是要绘制的数据
# 如果您的数据结构不同，请相应地修改以下代码
time_column = data.columns[0]  # 时间列的名称
data_column = data.columns[1]  # 数据列的名称

# 将时间列转换为日期时间格式（如果它还不是）
# 假设时间列是字符串格式，例如 'YYYY-MM-DD HH:MM:SS'
# 如果时间格式不同，请修改format参数
data[time_column] = pd.to_datetime(data[time_column], format='%Y-%m-%d %H:%M:%S')
data=data[-16609:]
data.reset_index(drop=True, inplace=True)
# 设置时间列为索引
#data.set_index(time_column, inplace=True)

# 绘制折线图
plt.figure(figsize=(15, 7))
plt.plot( data[data_column], label=data_column)
plt.title(f'Time Series of {data_column}')
plt.xlabel('Time')
plt.ylabel(data_column)
plt.legend()
plt.grid(True)
plt.savefig('test_batch_y')