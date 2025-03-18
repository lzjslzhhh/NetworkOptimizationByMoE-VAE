import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess_and_merge(input_path, output_path):
    """
    数据预处理：处理所有流量类型的数据，并将它们合并到一个 CSV 文件中
    :param input_path: 输入数据路径
    :param output_path: 输出数据路径
    """
    df = pd.read_csv(input_path)

    # 清理列名
    df.columns = df.columns.str.strip().str.replace(' ', '_')

    numerical_columns = df.select_dtypes(include=np.number).columns
    df[numerical_columns] = df[numerical_columns].applymap(lambda x: x if x >= 0 else np.nan)
    df = df.dropna(subset=numerical_columns)  # 去除包含NaN的行

    def determine_protocol(row):
        """根据 Destination Port 推断协议类型"""
        if row['Destination_Port'] in [80, 443]:  # 常见的 HTTP/HTTPS 端口
            return 'TCP'
        elif row['Destination_Port'] in [53, 161]:  # DNS/SNMP 端口
            return 'UDP'
        else:
            return 'UDP'

    df['Protocol'] = df.apply(determine_protocol, axis=1)

    attack_labels = ['DoS slowloris', 'DoS Hulk', 'DoS Slowhttptest', 'DoS GoldenEye']
    df['attack'] = df['Label'].apply(lambda x: 1 if x in attack_labels else 0)

    df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]

    features_to_normalize = ['Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets',
                             'Total_Length_of_Fwd_Packets', 'Flow_Bytes/s', 'Flow_Packets/s', 'Flow_IAT_Mean',
                             'Fwd_Packet_Length_Max', 'Fwd_IAT_Mean', 'Packet_Length_Mean']

    scaler = MinMaxScaler()
    df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

    # 保存归一化器
    joblib.dump(scaler, f"{output_path}/scaler_all.csv")

    # 根据 Label 进行筛选
    df_attack = df[df['attack'] == 1]  # 攻击流量
    df_tcp = df[(df['Protocol'] == 'TCP') & (df['attack'] == 0)]  # TCP 流量
    df_udp = df[(df['Protocol'] == 'UDP') & (df['attack'] == 0)]  # UDP 流量

    # 将所有非攻击和非TCP流量都归为UDP
    df_other = df[(df['Protocol'] != 'TCP') & (df['attack'] == 0)]
    df_udp = pd.concat([df_udp, df_other], axis=0)

    # 合并所有数据
    df_all = pd.concat([df_tcp, df_udp, df_attack], axis=0)

    # 保存合并后的数据
    output_file = f"{output_path}/all_processed_data.csv"
    df_all.to_csv(output_file, index=False)

    print(f"所有流量预处理完成，保存至 {output_file}")


if __name__ == "__main__":
    # 输入数据路径和输出路径
    input_data_path = "../data/Wednesday-workingHours.pcap_ISCX.csv"  # 修改为你的输入数据路径
    output_data_path = "../data/expert_data"  # 输出目录，保存处理结果

    # 处理所有流量数据并合并
    preprocess_and_merge(input_data_path, output_data_path)
