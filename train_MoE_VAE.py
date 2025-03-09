import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from MoE_VAE import MoE_VAE
from config import Config
from utils.data_loader import SliceDataset


def calculate_loss(alloc, real_demand, pred, mu, logvar, total_res):
    """计算各损失项并确保非负"""
    # 1. 资源短缺损失
    allocated = alloc * total_res
    shortage = torch.clamp(real_demand - allocated, min=0).mean()

    # 2. 资源利用率项（限制0-1范围）
    alloc_sum = allocated.sum()
    demand_sum = real_demand.sum() + 1e-8
    utilization = torch.min(alloc_sum / demand_sum, torch.tensor(1.0).to(alloc.device))
    utilization_term = 1 - utilization

    # 3. 修正后的KL散度（确保正数）
    kl_div = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)

    # 4. 分配稳定性
    stability = F.mse_loss(alloc[:, 1:], alloc[:, :-1])

    # 组合损失项（权重经过验证）
    loss = (
            1.0 * shortage +
            0.5 * utilization_term +
            0.1 * kl_div +
            0.3 * stability
    )

    # 返回损失和指标
    metrics = {
        'shortage': shortage.item(),
        'utilization': utilization.item(),
        'kl_div': kl_div.item(),
        'stability': stability.item()
    }

    return loss, metrics

def main():
    config = Config()
    writer = SummaryWriter(config.log_dir)

    dataset = SliceDataset(config.data_path, window_size=config.window_size)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = MoE_VAE(
        moe_input_dim=config.window_size * 3,
        vae_input_dim=3 + 3
    ).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    best_loss = float('inf')
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        total_metrics = {
            'shortage': 0.0,
            'utilization': 0.0,
            'kl_div': 0.0,
            'stability': 0.0
        }

        for batch_idx, (hist_seq, real_demand) in enumerate(loader):

            hist_seq = hist_seq.to(config.device,non_blocking=True)
            real_demand = real_demand.to(config.device,non_blocking=True)

            prev_alloc = torch.zeros(hist_seq.size(0), 3).to(config.device)
            alloc, pred, mu, logvar = model(hist_seq, prev_alloc)

            # 计算损失项
            loss, batch_metrics = calculate_loss(
                alloc=alloc,
                real_demand=real_demand,
                pred=pred,
                mu=mu,
                logvar=logvar,
                total_res=config.total_res
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 累计指标
            total_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += batch_metrics[key]
                # 记录batch指标
            if batch_idx % 100 == 0:
                writer.add_scalar('Train/Batch_Loss', loss.item(),
                                      epoch * len(loader) + batch_idx)
                writer.add_scalars('Train/Batch_Metrics', batch_metrics,
                                       epoch * len(loader) + batch_idx)

        # 记录epoch指标
        avg_loss = total_loss / len(loader)
        avg_metrics = {k: v / len(loader) for k, v in total_metrics.items()}

        # 学习率调度
        scheduler.step(avg_loss)

        # 记录TensorBoard
        writer.add_scalar('Loss/Epoch', avg_loss, epoch)
        writer.add_scalars('Metrics', avg_metrics, epoch)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.best_model_path)
            print(f"Epoch {epoch + 1}: 发现新的最佳模型，损失值 {avg_loss:.4f}")

        # 定期保存检查点
        if (epoch + 1) % config.save_interval == 0:
            ckpt_path = f"{config.ckpt_dir}/epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss
            }, ckpt_path)

        print(f"Epoch {epoch + 1}/{config.epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"短缺: {avg_metrics['shortage']:.2f} | "
              f"利用率: {avg_metrics['utilization']:.1%} | "
              f"KL散度: {avg_metrics['kl_div']:.2f}")

    print(f"Model saved to {config.best_model_path}")
    writer.close()


if __name__ == '__main__':
    main()
