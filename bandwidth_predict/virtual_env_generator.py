import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

class VirtualEnvGenerator:
    def __init__(self, real_data_path):
        """
        初始化生成器并进行数据预处理
        :param real_data_path: 真实数据路径
        """
        self.real_data = pd.read_csv(real_data_path)
        self.metadata = self._create_metadata()

    def _create_metadata(self):
        """ 创建 metadata，定义数据列的类型 """
        metadata = SingleTableMetadata()
        metadata.add_column('Flow_Duration', sdtype='numerical')
        metadata.add_column('Total_Length_of_Fwd_Packets', sdtype='numerical')
        metadata.add_column('Flow_Bytes/s', sdtype='numerical')
        metadata.add_column('Flow_IAT_Mean', sdtype='numerical')
        metadata.add_column('Flow_Packets/s', sdtype='numerical')
        metadata.add_column('Packet_Length_Mean', sdtype='numerical')
        metadata.add_column('Fwd_IAT_Mean', sdtype='numerical')
        metadata.add_column('Fwd_Packet_Length_Max', sdtype='numerical')
        metadata.add_column('Protocol', sdtype='categorical')
        metadata.add_column('attack', sdtype='categorical')
        return metadata

    def train_ctgan(self, model_save_path='../models/virtual_generator.pkl'):
        """ 训练 CTGAN 模型并保存 """
        # 选择训练特征
        features_to_train = ['Flow_Duration', 'Total_Length_of_Fwd_Packets', 'Flow_Bytes/s', 'Flow_IAT_Mean',
                             'Flow_Packets/s', 'Packet_Length_Mean', 'Fwd_IAT_Mean', 'Fwd_Packet_Length_Max',
                             'Protocol', 'attack']
        df_train = self.real_data[features_to_train]

        # 初始化并训练 CTGAN
        ctgan = CTGANSynthesizer(metadata=self.metadata,
                enforce_rounding=False,
                epochs=100,
                verbose=True
                )
        ctgan.fit(df_train)

        # 保存训练好的模型
        ctgan.save(model_save_path)
        print("🎉 CTGAN 模型训练完成并保存！")
        return ctgan

    def generate_virtual_data(self, ctgan, scenario_list, sample_size=10000):
        """
        使用训练好的 CTGAN 生成虚拟数据
        :param ctgan: 训练好的 CTGAN 模型
        :param scenario_list: 场景列表，如['tcp', 'udp', 'attack']
        :param sample_size: 每个场景生成的样本数量
        :return: 合并后的虚拟数据
        """
        full_virtual_data = pd.DataFrame()

        for scenario in scenario_list:
            print(f"⏳ 正在生成 {scenario} 场景数据...")

            # 生成虚拟数据
            synthetic_data = self.generate_scenario(ctgan, scenario, sample_size)

            # 将生成的虚拟数据添加到总数据中
            full_virtual_data = pd.concat([full_virtual_data, synthetic_data], ignore_index=True)

            # 保存每个场景的虚拟数据
            synthetic_data.to_csv(f'../data/virtual_{scenario}_data.csv', index=False)

        # 保存合并后的虚拟数据
        full_virtual_data.to_csv("../data/virtual_data.csv", index=False)
        print("🎉 所有虚拟数据生成完成！")
        return full_virtual_data

    def generate_scenario(self, ctgan, scenario, num_samples):
        """
        生成指定场景的虚拟数据
        :param ctgan: 训练好的CTGAN模型
        :param scenario: 场景类型
        :param num_samples: 样本数量
        :return: 生成的虚拟数据
        """
        # 创建 scenario_type 列，根据 scenario 值设置对应的场景类型
        if scenario == 'attack':
            condition = [1] * num_samples  # 如果 scenario 是 attack，所有样本的 condition 都是 '攻击'
            condition_df = pd.DataFrame({'attack': condition})
        else:
            # 如果 scenario 是 'tcp' 或 'udp'，根据该协议类型设置 condition
            condition = [scenario.upper()] * num_samples  # scenario 为 'tcp' 或 'udp'，则将其转换为大写 'TCP' 或 'UDP'
            condition_df = pd.DataFrame({'Protocol': condition})

        # 使用CTGAN模型生成剩余的列
        synthetic_data = ctgan.sample_remaining_columns(condition_df, batch_size=num_samples)

        # 根据场景类型调整生成的数据
        if scenario == 'attack':
            # 对攻击场景数据进行处理
            synthetic_data['Flow_Bytes/s'] = np.clip(synthetic_data['Flow_Bytes/s'] + 0.2, 0, 1)
            synthetic_data['Fwd_IAT_Mean'] = np.clip(synthetic_data['Fwd_IAT_Mean'] * 1.5, 0, np.inf)
            synthetic_data['Flow_IAT_Mean'] = np.clip(synthetic_data['Flow_IAT_Mean'] * 1.2, 0, np.inf)
            # 添加攻击流量的特征，如增加负载、增大流量等
            synthetic_data['attack'] = 1  # 标记为攻击流量
        elif scenario == 'tcp':
            # 对TCP场景数据进行处理
            synthetic_data['Flow_Bytes/s'] = np.clip(synthetic_data['Flow_Bytes/s'], 0, 0.9)
            synthetic_data['Fwd_IAT_Mean'] = np.clip(synthetic_data['Fwd_IAT_Mean'] * 0.9, 0, np.inf)
            # 设置一些TCP流量的特征，比如保持较低的负载和流量
            synthetic_data['Protocol']='TCP'
            synthetic_data['attack'] = 0  # 标记为非攻击流量
        elif scenario == 'udp':
            # 对UDP场景数据进行处理
            synthetic_data['Flow_Bytes/s'] = np.clip(synthetic_data['Flow_Bytes/s'], 0, 0.8)
            synthetic_data['Fwd_IAT_Mean'] = np.clip(synthetic_data['Fwd_IAT_Mean'] * 0.8, 0, np.inf)
            # 设置一些UDP流量的特征，比如较高的包速率和较小的负载
            synthetic_data['Protocol'] = 'UDP'
            synthetic_data['attack'] = 0  # 标记为非攻击流量
        else:
            raise ValueError(f"Unknown scenario type: {scenario}")


        return synthetic_data


if __name__ == '__main__':
    # 设置真实数据路径
    real_data_path = '../data/expert_data/all_processed_data.csv'

    # 初始化生成器
    generator = VirtualEnvGenerator(real_data_path)

    # 训练 CTGAN 模型并保存
    # ctgan = generator.train_ctgan()

    # 生成并保存多场景数据
    ctgan = CTGANSynthesizer.load('../models/virtual_generator.pkl')

    scenario_list = ['tcp', 'udp', 'attack']
    virtual_data = generator.generate_virtual_data(CTGANSynthesizer.load('../models/virtual_generator.pkl'), scenario_list, sample_size=50000)
