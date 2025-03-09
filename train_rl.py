import gymnasium
from gymnasium import spaces, Env
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from config import Config
from gymnasium.envs.registration import register
config = Config()

register(
    id="NetworkSliceEnv-v0",  # 唯一环境ID，必须全局唯一
    entry_point="__main__:NetworkSliceEnv",  # 格式：模块名:类名
    kwargs={  # 默认参数（实际使用时会被覆盖）
        "df": None,
        "window_size": 5
    }
)

class NetworkSliceEnv(Env):
    """自定义强化学习环境"""

    def __init__(self, df, window_size=5):
        super().__init__()
        self.df = df.values
        self.window_size = window_size
        self.current_step = window_size  # 跳过初始窗口
        self._disable_env_checker = True  # 暂时禁用内置检查

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
        obs = np.clip(self.state_buffer.copy(), 0.0, 100.0).astype(np.float32)
        return obs, {}

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
        obs = self.state_buffer.copy().astype(np.float32)
        assert self.observation_space.contains(obs), f"非法观测值: {obs}"

        return (
            self.state_buffer.copy().astype(np.float32),  # obs
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


def train_rl_policy():
    # 使用gymnasium的make_vec_env
    from stable_baselines3.common.env_util import make_vec_env

    # 加载人工生成的数据
    df = pd.read_csv(config.data_path)

    # 创建环境时使用注册的ID
    env = make_vec_env(
        lambda: gymnasium.make(  # 通过注册ID创建环境
            "NetworkSliceEnv-v0",
            df=df,  # 覆盖默认参数
            window_size=config.window_size
        ),
        n_envs=4,
        vec_env_cls=DummyVecEnv
    )

    # 初始化PPO模型
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,  # 已设置为1，保留此设置
        learning_rate=1e-5,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01,
        clip_range=0.2,
        tensorboard_log=config.log_dir
    )

    # 训练模型
    model.learn(total_timesteps=200_000)
    model.save(config.rl_model_path)
    print(f"RL策略保存至 {config.rl_model_path}")


if __name__ == "__main__":
    # 测试环境注册
    test_df = pd.DataFrame({
        "iot": [10, 12, 15],
        "video": [20, 22, 25],
        "web": [5, 6, 7]
    })

    # 必须使用完整注册ID
    env = gymnasium.make(
        "NetworkSliceEnv-v0",  # 包含版本号
        df=test_df,
        window_size=3
    )

    train_rl_policy()