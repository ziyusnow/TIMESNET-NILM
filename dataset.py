import pandas as pd

# 读取 CSV 文件
data = pd.read_csv('./UKDALE_1_train.csv')

# 取出前 10000 个数据
subset_data = data.head(10000)

# 将结果保存到新的 CSV 文件中，可根据需求修改文件名
subset_data.to_csv('subset_UKDALE_10000.csv', index=False)