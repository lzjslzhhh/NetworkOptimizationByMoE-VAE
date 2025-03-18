import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from bandwidth_predict.MoEdataset import MoEResourceDataset
from MoE import MoEModel


# 1. 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型
model = MoEModel(input_dims={"Gating": 8, "TCP": 4, "UDP": 4, "Attack": 3}, output_dim=1)
model.load_state_dict(torch.load('../models/MoEModel_new.pth'))
model.to(device)
model.eval()  # 设置为评估模式

# 2. 加载数据集
real_data_path = '../data/expert_data/all_processed_data.csv'
virtual_data_path = '../data/virtual_data.csv'

# 假设 MoEResourceDataset 是你的自定义数据集类
dataset = MoEResourceDataset(real_data_path, virtual_data_path)
val_loader = DataLoader(dataset, batch_size=128, shuffle=False)

# 3. 从验证集获取预测值与真实值
outputs_all = []
y_tcp_all = []
y_udp_all = []
y_attack_all = []

# 进行预测
with torch.no_grad():
    for x_all, x_tcp, x_udp, x_attack, y_tcp, y_udp, y_attack in val_loader:
        x_all, x_tcp, x_udp, x_attack, y_tcp, y_udp, y_attack = \
            x_all.to(device), x_tcp.to(device), x_udp.to(device), x_attack.to(device), \
            y_tcp.to(device), y_udp.to(device), y_attack.to(device)

        # 获取模型输出
        tcp_out, udp_out, attack_out, final_output, gate_weights = model(x_all, x_tcp, x_udp, x_attack)

        # 存储输出和真实值
        outputs_all.append(final_output.cpu().numpy())
        y_tcp_all.append(y_tcp.cpu().numpy())
        y_udp_all.append(y_udp.cpu().numpy())
        y_attack_all.append(y_attack.cpu().numpy())

# 将输出转换为 NumPy 数组
outputs_all = np.concatenate(outputs_all, axis=0)
y_tcp_all = np.concatenate(y_tcp_all, axis=0)
y_udp_all = np.concatenate(y_udp_all, axis=0)
y_attack_all = np.concatenate(y_attack_all, axis=0)

# 4. 绘制散点图

# TCP
plt.figure(figsize=(10, 6))
plt.scatter(y_tcp_all, outputs_all, alpha=0.5, label='TCP')
plt.plot([min(y_tcp_all), max(y_tcp_all)], [min(y_tcp_all), max(y_tcp_all)], color='red', linestyle='--', label='Ideal (y = x)')
plt.xlabel('True Values ')
plt.ylabel('Predicted Values ')
plt.title('True vs Predicted')
plt.legend()



plt.tight_layout()
plt.savefig('../reports/MoE_new.png', dpi=300, bbox_inches='tight')
plt.show()

