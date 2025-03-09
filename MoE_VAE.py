import torch
import torch.nn as nn

class MoE(nn.Module):
    def __init__(self, input_dim, num_experts=3):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 3)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)  # [batch*seq, input_dim]
        gates = self.gate(x)  # [batch, num_experts]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # 加权求和每个专家的输出
        outputs = torch.einsum('be,bej->bj', gates, expert_outputs)  # 正确维度操作
        return outputs

class VAEAllocator(nn.Module):
    def __init__(self, input_dim=6, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, latent_dim*2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, logvar = encoded.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        alloc = self.decoder(z)
        return alloc, mu, logvar

class MoE_VAE(nn.Module):
    def __init__(self,moe_input_dim, vae_input_dim=6):
        super().__init__()
        self.moe = MoE(moe_input_dim)
        self.vae = VAEAllocator(vae_input_dim)

    def forward(self, hist_data,prev_alloc):
        # hist_data形状: [batch, window_size, 3]
        batch_size = hist_data.size(0)

        # MoE预测
        flattened_hist = hist_data.reshape(batch_size, -1)  # [batch, window_size*3]
        pred_demand = self.moe(flattened_hist)  # [batch, 3]

        # 拼接预测需求和历史分配
        vae_input = torch.cat([pred_demand, prev_alloc], dim=1)  # [batch, 6]

        # VAE生成分配
        alloc, mu, logvar = self.vae(vae_input)
        return alloc, pred_demand, mu, logvar

# if __name__ == "__main__":
#     # 加载多切片数据集
#     dataset = MultiSliceDataset(config.data_path, config.window_size)
#     loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
#
#     # 初始化模型
#     # 修改输入维度：window_size * num_slices
#     model = MultiSliceMoE(
#         num_experts=3,
#         input_dim=config.window_size * len(dataset.slice_columns),  # 5时间步×3切片
#         output_dim=len(dataset.slice_columns)  # 预测3个切片的需求
#     ).to(config.device)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
#     writer = SummaryWriter(log_dir='logs/moe')
#
#     # 训练循环
#     for epoch in range(config.moe_epochs):
#         model.train()
#         epoch_loss = 0.0
#         for batch_x, batch_y in loader:
#             batch_x = batch_x.to(config.device)
#             batch_y = batch_y.to(config.device)
#
#             optimizer.zero_grad()
#             outputs = model(batch_x)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()
#
#             epoch_loss += loss.item()
#
#         avg_loss = epoch_loss / len(loader)
#         writer.add_scalar('Loss/Train', avg_loss, epoch)
#         print(f'Epoch {epoch + 1}/{config.moe_epochs} | Loss: {avg_loss:.4f}')
#
#     # 保存模型
#     torch.save(model.state_dict(), config.moe_model_path)
#     print(f"Model saved to {config.moe_model_path}")
#     writer.close()
