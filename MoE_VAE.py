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
            nn.Linear(64, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, logvar = encoded.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        alloc = self.decoder(z)
        return alloc, mu, logvar


class MoE_VAE(nn.Module):
    def __init__(self, moe_input_dim, vae_input_dim=6):
        super().__init__()
        self.moe = MoE(moe_input_dim)
        self.vae = VAEAllocator(vae_input_dim)

    def forward(self, hist_data, prev_alloc):
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
