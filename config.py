import torch


class Config:
    def __init__(self):
        # 数据参数
        self.data_path = "data/network_slices.csv"
        self.window_size = 5
        self.num_samples = 20000
        # 训练参数
        self.batch_size = 128
        self.epochs = 50
        self.lr = 1e-3
        self.total_res = 100.0  # 总资源量
        self.save_interval = 10  # 保存检查点的间隔epoch数

        # 路径配置
        self.log_dir = "logs/train"
        self.best_model_path = "models/best_moe_vae.pth"
        self.ckpt_dir = "checkpoints"
        self.rl_model_path = "models/rl_policy.zip"
        self.rl_log_dir = "logs/rl_training"
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __str__(self):
        return f"""
        === 配置参数 ===
        数据路径: {self.data_path}
        窗口大小: {self.window_size}
        批大小: {self.batch_size}
        总epoch数: {self.epochs}
        学习率: {self.lr}
        总资源: {self.total_res}
        设备: {self.device}
        日志目录: {self.log_dir}
        最佳模型路径: {self.best_model_path}
        检查点目录: {self.ckpt_dir}
        """
