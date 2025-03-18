import pandas as pd
import torch
from torch.utils.data import Dataset


class MoEResourceDataset(Dataset):
    """MoE 训练数据集"""

    def __init__(self, real_data_path, virtual_data_path):
        # 加载数据
        real_df = pd.read_csv(real_data_path)
        virtual_df = pd.read_csv(virtual_data_path)

        # 选择特征
        self.features = {
            'Gating': ['Flow_Duration', 'Total_Length_of_Fwd_Packets', 'Flow_Bytes/s', 'Flow_IAT_Mean',
                       'Flow_Packets/s', 'Packet_Length_Mean', 'Flow_IAT_Mean', 'Fwd_IAT_Mean'],
            'TCP': ['Flow_Duration', 'Total_Length_of_Fwd_Packets', 'Flow_Bytes/s', 'Flow_IAT_Mean'],
            'UDP': ['Flow_Duration', 'Flow_Packets/s', 'Packet_Length_Mean', 'Fwd_IAT_Mean'],
            'Attack': ['Flow_Duration', 'Fwd_Packet_Length_Max', 'Flow_Bytes/s']
        }
        self.target_column = 'target'

        # 合并数据
        df = pd.concat([real_df, virtual_df], axis=0).reset_index(drop=True)
        df['allocated_bandwidth']=df['Flow_Bytes/s']
        df['packet_drop_rate'] = 1 - df['Flow_Packets/s']
        df['filtered_traffic_ratio'] = 1 - df['Flow_Bytes/s']

        df.reset_index(drop=True)
        df_tcp=df[df['Protocol']=='TCP'].reset_index(drop=True)
        df_udp=df[df['Protocol']=='UDP'].reset_index(drop=True)
        df_attack=df[df['attack']==1].reset_index(drop=True)

        # 计算最小数据集大小
        self.min_size = min(len(df), len(df_tcp), len(df_udp), len(df_attack))
        # 提取不同输入
        self.X_all = torch.tensor(df[self.features['Gating']].values, dtype=torch.float32)
        self.X_tcp = torch.tensor(df_tcp[self.features['TCP']].values, dtype=torch.float32)
        self.X_udp = torch.tensor(df_udp[self.features['UDP']].values, dtype=torch.float32)
        self.X_attack = torch.tensor(df_attack[self.features['Attack']].values, dtype=torch.float32)
        self.y_tcp = torch.tensor(df_tcp['Flow_Bytes/s'].values, dtype=torch.float32).unsqueeze(1)
        self.y_udp = torch.tensor(df_udp['packet_drop_rate'].values, dtype=torch.float32).unsqueeze(1)
        self.y_attack = torch.tensor(df_attack['filtered_traffic_ratio'].values, dtype=torch.float32).unsqueeze(1)

    def compute_target_value(self, df, expert_type):
        """
        计算不同专家类型的目标值
        """
        if expert_type == 'TCP':
            # TCP 目标值：归一化后的 `Flow_Bytes/s`
            return df['Flow_Bytes/s']

        elif expert_type == 'UDP':
            # UDP 目标值：丢包率（基于归一化 Flow_Packets/s）
            return 1 - df['Flow_Packets/s']

        elif expert_type == 'Attack':
            # 攻击流量目标值：流量过滤率（基于归一化 Flow_Bytes/s）
            return 1 - df['Flow_Bytes/s']

        else:
            raise ValueError(f"Unknown expert type: {expert_type}")

    def __len__(self):
        return self.min_size

    def __getitem__(self, idx):
        idx_tcp = min(idx, len(self.X_tcp) - 1)
        idx_udp = min(idx, len(self.X_udp) - 1)
        idx_attack = min(idx, len(self.X_attack) - 1)

        return (
            self.X_all[idx],
            self.X_tcp[idx_tcp],
            self.X_udp[idx_udp],
            self.X_attack[idx_attack],
            self.y_tcp[idx_tcp],
            self.y_udp[idx_udp],
            self.y_attack[idx_attack]
        )

if __name__ == "__main__":
    real_data_path = '../data/expert_data/all_processed_data.csv'
    virtual_data_path = '../data/virtual_data.csv'

    # TCP 专家数据集
    tcp_dataset = MoEResourceDataset(real_data_path, virtual_data_path)

    # 打印样本数据
    print("样本数据 (TCP):", tcp_dataset[2])

    # 数据集大小
    print(f"TCP 训练数据集大小: {len(tcp_dataset)}")
