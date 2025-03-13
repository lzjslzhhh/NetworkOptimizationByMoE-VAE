"""
data_preprocessing.py
功能：数据清洗、特征提取、压力系数计算 + 场景类型标注
"""

import pandas as pd
import numpy as np

def preprocess_data(input_path):
    # 加载数据
    df = pd.read_csv(input_path)

    # ========== 1. 列名规范化 ==========
    df.columns = df.columns.str.strip().str.replace(' ', '_')

    # ========== 2. 特征工程 ==========
    # 基础特征计算
    df = df.assign(
        bandwidth_util=df['Total_Length_of_Fwd_Packets'] / 1e6,  # 转换为MB
        packet_rate=df['Total_Fwd_Packets'] / (df['Flow_Duration'] + 1e-6)
    )

    # ========== 3. 压力系数计算 ==========
    attack_pressure = {
        'DoS slowloris': 0.95,
        'DoS Hulk': 0.7,
        'DoS Slowhttptest': 0.87,
        'DoS GoldenEye': 0.8,
        'BENIGN': 0.0
    }
    df['pressure'] = df['Label'].map(attack_pressure).fillna(0.5)

    # ========== 4. 场景类型标注 ==========
    # 规则1：标记攻击流量
    df['scenario_type'] = np.where(
        df['Label'].str.contains('BENIGN'),
        'non_attack',  # 待进一步处理
        'attack'
    )

    # 规则2：在正常流量中检测突发场景
    # 计算正常流量的带宽阈值（均值+2倍标准差）
    benign_data = df[df['Label'] == 'BENIGN']
    threshold = benign_data['bandwidth_util'].mean() + 2 * benign_data['bandwidth_util'].std()

    # 标记突发场景（使用滚动窗口检测持续高带宽）
    # 假设数据按时间排序，窗口大小=5（持续5个采样点）
    df['rolling_bandwidth'] = df['bandwidth_util'].rolling(5, min_periods=1).mean()
    burst_mask = (df['scenario_type'] == 'non_attack') & (df['rolling_bandwidth'] > threshold)
    df.loc[burst_mask, 'scenario_type'] = 'burst'

    # 剩余正常流量标记为normal
    df['scenario_type'] = np.where(
        df['scenario_type'] == 'non_attack',
        'normal',
        df['scenario_type']
    )

    # ========== 5. 保存结果 ==========
    output_cols = ['bandwidth_util', 'packet_rate', 'pressure', 'Label', 'scenario_type']
    df[output_cols].to_csv("../data/processed_data.csv", index=False)
    print("✅ 预处理完成，保存至 data/processed_data.csv")

    # 打印阈值信息
    print(f"📈 突发场景带宽阈值：{threshold:.2f} MB/s")
    return df[output_cols]

if __name__ == "__main__":
    preprocess_data("../data/Wednesday-workingHours.pcap_ISCX.csv")