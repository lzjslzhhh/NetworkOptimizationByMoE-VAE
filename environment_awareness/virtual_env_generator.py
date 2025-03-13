"""
virtual_env_generator.py
功能：适配最新SDV API的虚拟环境生成器
依赖：sdv>=1.2.0, pandas
"""

import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

class VirtualEnvGenerator:
    def __init__(self, real_data_path):
        """初始化生成器"""
        # 加载预处理数据（需包含'scenario_type'列）
        self.real_data = pd.read_csv(real_data_path)
        self._prepare_metadata()

        # 初始化并训练CTGAN
        self.ctgan = CTGANSynthesizer(
            metadata=self.metadata,
            enforce_rounding=False,
            epochs=100,
            verbose=True
        )
        self.ctgan.fit(self.real_data)

    def _prepare_metadata(self):
        """配置元数据（关键步骤）"""
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(self.real_data)

        # 明确声明条件列
        self.metadata.update_column(
            column_name='scenario_type',
            sdtype='categorical'
        )

        # 确保压力列为数值型
        self.metadata.update_column(
            column_name='pressure',
            sdtype='numerical'
        )

    def generate_scenario(self, scenario_name, num_samples=1000):
        """生成指定场景数据

        参数：
            scenario_name: 'burst'/'normal'/'attack'
            num_samples: 生成样本数
        """
        # 创建条件数据框架
        condition_data = pd.DataFrame({
            'scenario_type': [scenario_name] * num_samples
        })

        # 生成数据
        synthetic_data = self.ctgan.sample_remaining_columns(
            known_columns=condition_data,
            batch_size=num_samples
        )

        # 后处理：确保压力系数范围
        synthetic_data['pressure'] = synthetic_data['pressure'].clip(0, 1)
        return synthetic_data

    def save_generator(self, path):
        """保存生成器模型"""
        self.ctgan.save(path)

# ================= 使用示例 =================
if __name__ == "__main__":
    # 初始化生成器（预处理数据需包含'scenario_type'列）
    generator = VirtualEnvGenerator("../data/processed_data.csv")

    # 生成突发攻击场景数据
    burst_data = generator.generate_scenario('burst', 5000)
    print("突发场景数据示例：")
    print(burst_data[['scenario_type', 'pressure', 'bandwidth_util']].head())

    # 保存生成器
    generator.save_generator("../models/virtual_generator.pkl")