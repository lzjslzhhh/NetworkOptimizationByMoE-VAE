import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces, Env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from MoE_VAE import MoE_VAE
from config import Config

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False
from gymnasium.envs.registration import register

# 注册与训练代码相同的环境
register(
    id="NetworkSliceEnv-v0",
    entry_point="__main__:NetworkSliceEnv",  # 注意：在对比代码中需要重新定义环境类
    kwargs={
        "df": pd.DataFrame(),  # 使用空数据框架作为默认值
        "window_size": 5
    }
)
config = Config()


class NetworkSliceEnv(Env):
    """自定义强化学习环境"""

    def __init__(self, df, window_size=5):
        super().__init__()
        self.df = df.values
        self.window_size = window_size
        self.current_step = window_size  # 跳过初始窗口

        # 动作空间：3个切片的分配比例 [0,1]^3
        self.action_space = spaces.Box(
            low=0, high=1,
            shape=(3,),
            dtype=np.float32
        )

        # 状态空间：历史需求窗口 (window_size × 3个切片)
        self.observation_space = spaces.Box(
            low=0, high=100,
            shape=(window_size * 3,),
            dtype=np.float32
        )

        # 初始化状态
        self.state_buffer = []

    def reset(self, seed=None, options=None):
        # 处理新的reset参数
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.state_buffer = self.df[:self.window_size].flatten()
        return self.state_buffer.copy(), {}  # 添加空的info字典

    def step(self, action):
        # 归一化动作确保总和为1
        action = np.clip(action, 0, 1)
        action /= action.sum() + 1e-8

        # 获取当前实际需求
        true_demand = self.df[self.current_step]

        # 计算分配结果
        allocated = action * config.total_res
        shortage = np.sum(np.maximum(true_demand - allocated, 0))
        overprov = np.sum(np.maximum(allocated - true_demand, 0))

        # 计算奖励
        reward = -0.7 * shortage - 0.2 * overprov - 0.1 * np.linalg.norm(action - self._get_last_action())

        # 更新状态（滑动窗口）
        self.current_step += 1
        self.state_buffer = np.roll(self.state_buffer, -3)
        self.state_buffer[-3:] = true_demand

        # 定义终止条件
        terminated = self.current_step >= len(self.df) - 1
        truncated = False  # 无时间截断需求

        return (
            self.state_buffer.copy(),  # obs
            float(reward),  # reward
            terminated,  # terminated
            truncated,  # truncated
            {"demand": true_demand}  # info
        )

    def _get_last_action(self):
        """获取最近一次的有效动作"""
        if self.current_step == self.window_size:
            return np.array([0.33, 0.34, 0.33])
        allocated = self.df[self.current_step - 1] / config.total_res
        return allocated / (allocated.sum() + 1e-8)


class StrategyComparator:
    def __init__(self, test_data):
        self.config = Config()
        self.test_data = test_data  # 格式：[{'iot':, 'video':, 'web':}, ...]
        self.total_res = self.config.total_res
        self.window_size = self.config.window_size

    def _load_rl_model(self, model_path):
        """加载RL模型（关键修改）"""

        # 创建与训练时相同的虚拟环境
        def make_env():
            return gymnasium.make(
                "NetworkSliceEnv-v0",
                df=pd.DataFrame(self.test_data),
                window_size=self.window_size
            )

        return PPO.load(
            model_path,
            env=make_vec_env(make_env, n_envs=1),
            device=self.config.device
        )

    def _prepare_rl_input(self, data):
        """构建RL模型输入状态"""
        # 假设状态包含最近5个时间步的需求（3个切片 × 5步 = 15维）
        if not hasattr(self, 'state_history'):
            self.state_history = [np.zeros(3) for _ in range(self.window_size)]

        self.state_history.append([data['iot'], data['video'], data['web']])
        self.state_history.pop(0)

        # 展平并归一化（假设训练时最大需求为100）
        state = np.array(self.state_history).flatten() / 100.0
        return state.astype(np.float32)

    def rl_allocation(self, model_path):
        """RL策略分配（更新状态管理）"""
        model = self._load_rl_model(model_path)
        allocations = []
        state_history = []  # 维护状态历史

        for data in self.test_data:
            # 构建状态（与训练代码相同的逻辑）
            state_history.append([data['iot'], data['video'], data['web']])
            if len(state_history) > self.window_size:
                state_history.pop(0)

            # 填充不足的窗口
            while len(state_history) < self.window_size:
                state_history.insert(0, [0, 0, 0])

            state = np.array(state_history).flatten().astype(np.float32)

            # 模型预测
            action, _ = model.predict(state)
            alloc_ratio = np.exp(action) / np.sum(np.exp(action))  # softmax转换

            allocations.append({
                'iot': alloc_ratio[0] * self.total_res,
                'video': alloc_ratio[1] * self.total_res,
                'web': alloc_ratio[2] * self.total_res
            })

        return allocations

    def static_allocation(self, ratios=[0.3, 0.5, 0.2]):
        """静态比例分配"""
        allocations = []
        for demand in self.test_data:
            alloc = {
                'iot': ratios[0] * self.total_res,
                'video': ratios[1] * self.total_res,
                'web': ratios[2] * self.total_res
            }
            allocations.append(alloc)
        return allocations

    def moe_vae_allocation(self, model_path):
        """MOE-VAE策略（确保时间窗口长度正确）"""
        # 初始化模型结构
        model = MoE_VAE(
            moe_input_dim=self.config.window_size * 3,
            vae_input_dim=6  # 3预测需求 + 3历史分配
        )

        # 加载模型参数
        model.load_state_dict(
            torch.load(model_path,
                       map_location=self.config.device,
                       weights_only=True)
        )
        model.eval()

        allocations = []
        history = []  # 维护完整历史记录

        for i, data in enumerate(self.test_data):
            # ================== 时间窗口处理 ==================
            # 1. 更新历史记录
            history.append(list(data.values()))

            # 2. 构建时间窗口（始终保留最近的window_size个时间步）
            if len(history) > self.config.window_size:
                history.pop(0)

            # 3. 前向填充零（当历史不足window_size时）
            padded_window = []
            for t in range(-self.config.window_size + 1, 1):
                idx = i + t
                if 0 <= idx < len(history):
                    padded_window.append(history[idx])
                else:
                    padded_window.append([0.0, 0.0, 0.0])  # 填充零

            # 4. 转换为模型输入格式 [batch=1, window_size, 3]
            hist_tensor = torch.FloatTensor(padded_window).unsqueeze(0)

            # ================== 历史分配处理 ==================
            if i == 0:
                prev_alloc = torch.zeros(1, 3)
            else:
                prev_alloc = torch.FloatTensor([
                    [allocations[-1][k] / self.total_res
                     for k in ['iot', 'video', 'web']]
                ])

            # ================== 模型推理 ==================
            with torch.no_grad():
                hist_tensor = hist_tensor.to(self.config.device)
                prev_alloc = prev_alloc.to(self.config.device)

                # 执行前向传播
                alloc_ratio = model(hist_tensor, prev_alloc)[0].cpu().numpy()

            # 记录分配结果
            allocations.append({
                'iot': alloc_ratio[0, 0] * self.total_res,
                'video': alloc_ratio[0, 1] * self.total_res,
                'web': alloc_ratio[0, 2] * self.total_res
            })

        return allocations

    def evaluate(self, allocations):
        """计算关键指标"""
        metrics = {
            'MAE': [],  # 平均绝对误差
            'Utilization': [],  # 资源利用率
            'Shortage': [],  # 资源短缺量
            'Overprov': []  # 过度分配量
        }

        for alloc, true_demand in zip(allocations, self.test_data):
            # 计算各切片指标
            for slice_type in ['iot', 'video', 'web']:
                pred = alloc[slice_type]
                true = true_demand[slice_type]

                # 计算单指标
                metrics['MAE'].append(abs(pred - true))
                metrics['Shortage'].append(max(true - pred, 0))
                metrics['Overprov'].append(max(pred - true, 0))
                metrics['Utilization'].append(min(pred, true) / true if true > 0 else 0.0)

        return {
            'MAE': np.mean(metrics['MAE']),
            'Utilization': np.mean(metrics['Utilization']),
            'Shortage': np.mean(metrics['Shortage']),
            'Overprov': np.mean(metrics['Overprov']),
            'allocations': allocations
        }

    def generate_performance_report(self, results, report_dir="reports"):
        """生成性能对比可视化报告"""
        import os
        os.makedirs(report_dir, exist_ok=True)

        # 柱状图对比
        self._plot_metric_comparison(results, report_dir)
        # 时间序列对比
        self._plot_temporal_comparison(results, report_dir)

    def _plot_metric_comparison(self, results, save_dir):
        """绘制四维指标柱状图"""
        metrics = ['MAE', 'Utilization', 'Shortage', 'Overprov']
        titles = ['平均绝对误差', '资源利用率', '资源短缺量', '过度分配量']
        units = ['Units', 'Percentage', 'Units', 'Units']

        plt.figure(figsize=(16, 10))
        for i, (metric, title, unit) in enumerate(zip(metrics, titles, units)):
            plt.subplot(2, 2, i + 1)
            strategies = list(results.keys())
            values = [results[s][metric] for s in strategies]

            # 特殊处理百分比显示
            if metric == 'Utilization':
                values = [v * 100 for v in values]
                unit = 'Percentage (%)'

            bars = plt.bar(strategies, values, alpha=0.7)
            plt.title(title, fontsize=12)
            plt.ylabel(unit)

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.2f}' if metric != 'Utilization' else f'{height:.1f}%',
                         ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/metric_comparison.png")
        plt.close()

    def _plot_temporal_comparison(self, results, save_dir):
        """绘制时间序列对比图"""
        plt.figure(figsize=(18, 12))
        for idx, slice_type in enumerate(['iot', 'video', 'web'], 1):
            plt.subplot(3, 1, idx)

            # 真实需求
            true_values = [d[slice_type] for d in self.test_data]
            plt.plot(true_values, 'k--', label='真实需求', linewidth=2)

            # 各策略预测
            for strategy in results:
                pred_values = [a[slice_type] for a in results[strategy]['allocations']]
                plt.plot(pred_values, label=f'{strategy}分配')

            plt.title(f"{slice_type.upper()}切片资源分配对比", fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/temporal_comparison.png")
        plt.close()


# 生成动态需求示例
def generate_test_data(num_samples=100, total_res=100):
    data = []
    for _ in range(num_samples):
        # 生成随机需求（总和不超过总资源）
        demands = np.random.dirichlet(np.ones(3)) * total_res * 0.8  # 模拟资源紧张
        data.append({
            'iot': demands[0],
            'video': demands[1],
            'web': demands[2]
        })
    return data


if __name__ == "__main__":
    # 生成测试数据（示例）
    test_data = generate_test_data(24, 100)

    comparator = StrategyComparator(test_data)

    # 获取各策略结果（使用增强评估）
    results = {
        'Static': comparator.evaluate(comparator.static_allocation()),
        'PPO-RL': comparator.evaluate(comparator.rl_allocation(config.rl_model_path)),
        'MOE-VAE': comparator.evaluate(comparator.moe_vae_allocation(config.best_model_path))
    }

    # 生成可视化报告
    comparator.generate_performance_report(results)

    # 打印数值结果
    print("\n======== 预测性能对比结果 ========")
    for strategy in results:
        print(f"\n{strategy}策略:")
        print(f"  MAE: {results[strategy]['MAE']:.2f} units")
        print(f"  资源利用率: {results[strategy]['Utilization'] * 100:.1f}%")
        print(f"  平均短缺量: {results[strategy]['Shortage']:.2f} units")
        print(f"  过度分配量: {results[strategy]['Overprov']:.2f} units")
