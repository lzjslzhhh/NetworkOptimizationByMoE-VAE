import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from bandwidth_predict.MoEdataset import MoEResourceDataset


class Expert(nn.Module):
    """单个专家网络"""

    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class GatingNetwork(nn.Module):
    """门控网络，决定每个样本分配到哪个专家"""

    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, num_experts)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


class MoEModel(nn.Module):
    """MoE 资源分配优化模型"""

    def __init__(self, input_dims, output_dim):
        super(MoEModel, self).__init__()

        # 3 个专家，输入维度不同
        self.tcp_expert = Expert(input_dims["TCP"], output_dim)
        self.udp_expert = Expert(input_dims["UDP"], output_dim)
        self.attack_expert = Expert(input_dims["Attack"], output_dim)

        # 门控网络使用全特征输入
        self.gating = GatingNetwork(input_dims["Gating"], num_experts=3)

    def forward(self, x_all, x_tcp, x_udp, x_attack):
        # 计算门控权重
        gating_weights = self.gating(x_all)

        # 获取每个专家的预测
        tcp_out = self.tcp_expert(x_tcp)
        udp_out = self.udp_expert(x_udp)
        attack_out = self.attack_expert(x_attack)

        # 计算最终输出
        final_output = (
                gating_weights[:, 0].unsqueeze(1) * tcp_out +
                gating_weights[:, 1].unsqueeze(1) * udp_out +
                gating_weights[:, 2].unsqueeze(1) * attack_out
        )

        return tcp_out, udp_out, attack_out, final_output, gating_weights


def train_moe(model, train_loader, val_loader, epochs=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, min_lr=1e-6,
                                                           verbose=True)
    writer = SummaryWriter('../logs')

    best_val_loss = float('inf')
    early_stop = 15  # 提高耐心
    patience = 0  # 早停计数
    scale_factor = 1e6  # 根据实际情况调整
    print('开始训练:')
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i, (x_all, x_tcp, x_udp, x_attack, y_tcp, y_udp, y_attack) in enumerate(train_loader):
            x_all, x_tcp, x_udp, x_attack, y_tcp, y_udp, y_attack = x_all.to(device), x_tcp.to(device), x_udp.to(
                device), x_attack.to(
                device), y_tcp.to(device), y_udp.to(device), y_attack.to(device)

            optimizer.zero_grad()
            tcp_out, udp_out, attack_out, outputs, gate_weights = model(x_all, x_tcp, x_udp, x_attack)
            loss_tcp = criterion(tcp_out, y_tcp)
            loss_udp = criterion(udp_out, y_udp)
            loss_attack = criterion(attack_out, y_attack)
            weighted_loss = (
                    gate_weights[:, 0].unsqueeze(1) * loss_tcp +
                    gate_weights[:, 1].unsqueeze(1) * loss_udp +
                    gate_weights[:, 2].unsqueeze(1) * loss_attack
            ).mean()
            weighted_loss.backward()

            # 每4个梯度步骤进行梯度裁剪
            if (i + 1) % 4 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += weighted_loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_loss_scaled = avg_train_loss * scale_factor
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # 验证步骤
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_all, x_tcp, x_udp, x_attack, y_tcp, y_udp, y_attack in val_loader:
                x_all, x_tcp, x_udp, x_attack, y_tcp, y_udp, y_attack = x_all.to(device), x_tcp.to(device), x_udp.to(
                    device), x_attack.to(device), y_tcp.to(device), y_udp.to(device), y_attack.to(device)

                tcp_out, udp_out, attack_out, outputs, gate_weights = model(x_all, x_tcp, x_udp, x_attack)
                # 计算每个专家的损失
                loss_tcp = criterion(tcp_out, y_tcp)
                loss_udp = criterion(udp_out, y_udp)
                loss_attack = criterion(attack_out, y_attack)

                # 计算加权损失
                weighted_loss = (
                        gate_weights[:, 0].unsqueeze(1) * loss_tcp +
                        gate_weights[:, 1].unsqueeze(1) * loss_udp +
                        gate_weights[:, 2].unsqueeze(1) * loss_attack
                ).mean()
                # 累加验证损失
                total_val_loss += weighted_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_loss_scaled = avg_val_loss * scale_factor
        writer.add_scalar('Loss/val', avg_val_loss_scaled, epoch)

        # 调整学习率
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss_scaled:.10f} - Val Loss: {avg_val_loss_scaled:.10f}")

        # 早停机制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), '../models/MoEModel_new.pth')
            print(f"Model saved with better validation loss: {avg_val_loss_scaled:.10f}")
            patience = 0
        else:
            patience += 1
            if patience >= early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    writer.close()


if __name__ == "__main__":
    real_data_path = '../data/expert_data/all_processed_data.csv'
    virtual_data_path = '../data/virtual_data.csv'

    dataset = MoEResourceDataset(real_data_path, virtual_data_path)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=256, shuffle=False)

    input_dims = {
        "Gating": len(dataset.features["Gating"]),
        "TCP": len(dataset.features["TCP"]),
        "UDP": len(dataset.features["UDP"]),
        "Attack": len(dataset.features["Attack"])
    }

    model = MoEModel(input_dims, output_dim=1)
    train_moe(model, train_loader, val_loader)

    # 测试门控权重
    test_input = dataset[0]  # 获取第一条样本
    x_all, x_tcp, x_udp, x_attack, _ = [t.unsqueeze(0) for t in test_input]
    output, gating_weights = model(x_all, x_tcp, x_udp, x_attack)

    print("MoE 输出:", output)
    print("门控权重:", gating_weights)
