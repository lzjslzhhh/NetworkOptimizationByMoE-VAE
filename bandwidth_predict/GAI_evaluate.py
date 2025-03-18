import numpy as np
import pandas as pd

# 加载数据
df = pd.read_csv('../data/virtual_data.csv')
df2 = pd.read_csv('../data/expert_data/all_processed_data.csv')
# 5️⃣  查看数值型特征的统计信息

# 获取虚拟流量和真实流量的描述统计
virtual_stats = df.describe(include=[np.number])
real_stats = df2.describe(include=[np.number])

# 将描述统计信息保存到CSV
virtual_stats.to_csv('./data/virtual_flow_stats.csv')
real_stats.to_csv('./data/real_flow_stats.csv')
