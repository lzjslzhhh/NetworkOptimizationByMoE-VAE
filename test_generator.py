# test_generator.py
import pandas as pd
from sdv.single_table import CTGANSynthesizer


def test_generator(model_path, scenario_type='burst', num_samples=1000):
    """测试生成器的主函数"""
    # 1. 载入预训练模型
    ctgan = CTGANSynthesizer.load(model_path)

    # 2. 创建条件数据框架
    condition_data = pd.DataFrame({
        'scenario_type': [scenario_type] * num_samples
    })

    # 3. 生成数据（自动适配元数据）
    synthetic_data = ctgan.sample_remaining_columns(
        known_columns=condition_data,
        batch_size=num_samples
    )

    # 4. 后处理确保压力系数有效
    synthetic_data['pressure'] = synthetic_data['pressure'].clip(0, 1)

    return synthetic_data


if __name__ == "__main__":
    # 测试突发流量场景
    data = test_generator(
        model_path="models/virtual_generator.pkl",
        scenario_type='burst',
        num_samples=500
    )

    print("\n生成数据统计特征：")
    print(data[['pressure', 'bandwidth_util']].describe())

    print("\n场景类型分布：")
    print(data['scenario_type'].value_counts())