import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

# ========== 1. LSTM 预测模型 ==========
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# ========== 2. MoE（Mixture of Experts） ==========
class MoE(nn.Module):
    def __init__(self, input_dim, num_experts=3, output_dim=2):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([
            LSTMPredictor(input_dim, 64, output_dim) for _ in range(num_experts)
        ])
        self.gating_network = nn.Linear(input_dim, num_experts)  # Gating Network

    def forward(self, x):
        gate_outputs = torch.softmax(self.gating_network(x), dim=1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.sum(gate_outputs.unsqueeze(2) * expert_outputs, dim=1)
        return output

# ========== 3. 环境感知虚拟环境生成器 ==========
class VirtualEnvGenerator:
    def __init__(self, real_data_path, model_path, moe_model_path, epochs=300):
        """初始化生成器"""
        self.real_data = pd.read_csv(real_data_path)
        self._prepare_metadata()
        self.scaler = MinMaxScaler()

        # 载入预训练 CTGAN 模型
        self.ctgan = CTGANSynthesizer.load(model_path)

        # 载入或训练 MoE 预测器
        self.moe_model_path = moe_model_path
        if self._load_moe_model():
            print("✅ 成功载入已训练的 MoE 模型！")
        else:
            print("⚠ 未找到已训练的 MoE，开始训练...")
            self._train_moe_predictor()
            self._save_moe_model()

    def _prepare_metadata(self):
        """配置元数据"""
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(self.real_data)
        self.metadata.update_column('scenario_type', sdtype='categorical')
        self.metadata.update_column('pressure', sdtype='numerical')
        self.metadata.update_column('bandwidth_util', sdtype='numerical')
        self.metadata.update_column('packet_rate', sdtype='numerical')

    def _train_moe_predictor(self, seq_len=10, epochs=100, lr=0.001, patience=10, min_delta=1e-4):
        """训练 MoE 预测器，添加早停机制和学习率调整"""
        writer = SummaryWriter(log_dir="./tensorboard_logs")  # TensorBoard 记录

        # 1. 生成突发和攻击场景的虚拟数据
        burst_data = self.generate_scenario('burst', 50000)
        attack_data = self.generate_scenario('attack', 30000)

        # 2. 合并数据并归一化
        all_data = pd.concat([self.real_data, burst_data, attack_data], ignore_index=True)
        traffic_data = all_data[['bandwidth_util', 'packet_rate']]
        traffic_data = self.scaler.fit_transform(traffic_data)

        X, y = [], []
        for i in range(len(traffic_data) - seq_len):
            X.append(traffic_data[i:i+seq_len])
            y.append(traffic_data[i+seq_len])

        X, y = np.array(X), np.array(y)
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

        # 3. 初始化 MoE
        self.moe_model = MoE(input_dim=2, num_experts=3, output_dim=2)
        optimizer = optim.Adam(self.moe_model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        best_loss = float('inf')
        patience_counter = 0

        # 4. 训练模型
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.moe_model(X)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            # 记录到 TensorBoard
            writer.add_scalar("Loss/train", loss.item(), epoch)

            # 早停机制
            if loss.item() < best_loss - min_delta:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            scheduler.step(loss.item())

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

            if patience_counter >= patience:
                print(f"⏳ 早停触发：训练在 {epoch + 1} 轮后终止（最佳损失: {best_loss:.6f}）")
                break

        writer.close()
        print(f"✅ MoE 训练完成！")

    def _save_moe_model(self):
        """保存 MoE 预测器"""
        torch.save(self.moe_model.state_dict(), self.moe_model_path)
        print(f"📌 MoE 预测器已保存至 {self.moe_model_path}")

    def _load_moe_model(self):
        """尝试载入已训练的 MoE 预测器"""
        try:
            self.moe_model = MoE(input_dim=2, num_experts=3, output_dim=2)
            self.moe_model.load_state_dict(torch.load(self.moe_model_path))
            self.moe_model.eval()
            return True
        except FileNotFoundError:
            return False

    def generate_scenario(self, scenario_name, num_samples=1000):
        """生成指定场景数据"""
        condition_data = pd.DataFrame({'scenario_type': [scenario_name] * num_samples})
        synthetic_data = self.ctgan.sample_remaining_columns(known_columns=condition_data, batch_size=num_samples)
        synthetic_data['pressure'] = synthetic_data['pressure'].clip(0, 1)
        return synthetic_data

    def predict_future_traffic(self, recent_data):
        """使用 MoE 预测未来流量"""
        recent_data = self.scaler.transform(recent_data)
        recent_data = torch.tensor(recent_data, dtype=torch.float32).unsqueeze(0)
        predicted_traffic = self.moe_model(recent_data).detach().numpy()
        return self.scaler.inverse_transform(predicted_traffic)

# ========== 4. 运行测试 ==========
if __name__ == "__main__":
    generator = VirtualEnvGenerator(
        "../data/processed_data.csv",
        "../models/virtual_generator.pkl",
        "../models/moe_model_envawns.pth"
    )

    burst_data = generator.generate_scenario('burst', 5000)
    print("突发场景数据示例：")
    print(burst_data[['scenario_type', 'pressure', 'bandwidth_util']].head())

    recent_traffic = generator.real_data[['bandwidth_util', 'packet_rate']].tail(10).values
    future_traffic = generator.predict_future_traffic(recent_traffic)
    print("未来流量预测：", future_traffic)
