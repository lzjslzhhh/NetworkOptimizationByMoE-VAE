import matplotlib
import numpy as np
import torch
from sklearn.manifold import TSNE

from MoE_VAE import MoE_VAE
from config import Config
from utils.data_loader import SliceDataset

matplotlib.use('Agg')  # 非交互模式下使用Agg后端
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ModelEvaluator:
    def __init__(self, model_path):
        self.config = Config()
        self.model = self._load_model(model_path)
        self.dataset = SliceDataset(self.config.data_path, self.config.window_size)

    def _load_model(self, model_path):
        model = MoE_VAE(
            moe_input_dim=self.config.window_size * 3,
            vae_input_dim=6
        ).to(self.config.device)
        model.load_state_dict(torch.load(model_path, map_location=self.config.device, weights_only=True))
        model.eval()
        return model

    def full_evaluation(self):
        """执行完整评估流程"""
        results = {
            'predictions': [],
            'true_demand': [],
            'allocations': [],
            'latent_vectors': []
        }

        with torch.no_grad():
            for hist, real_demand in self.dataset:
                hist = hist.unsqueeze(0).to(self.config.device)
                prev_alloc = torch.zeros(1, 3).to(self.config.device)

                alloc, pred, mu, _ = self.model(hist, prev_alloc)

                results['predictions'].append(pred.squeeze().cpu().numpy())
                results['true_demand'].append(real_demand.numpy())
                results['allocations'].append(alloc.squeeze().cpu().numpy())
                results['latent_vectors'].append(mu.squeeze().cpu().numpy())

        # 转换为numpy数组
        for key in results:
            results[key] = np.array(results[key])

        return results

    def generate_report(self, results):
        """生成可视化评估报告"""
        # 1. 需求预测准确率
        mae = np.mean(np.abs(results['predictions'] - results['true_demand']))
        print(f"平均绝对误差(MAE): {mae:.2f}")

        # 2. 资源分配分析
        allocated = results['allocations'] * self.config.total_res
        shortage = np.mean(np.maximum(results['true_demand'] - allocated, 0))
        utilization = np.sum(np.minimum(allocated, results['true_demand'])) / np.sum(results['true_demand'])
        print(f"\n资源短缺率: {shortage:.2f} units/step")
        print(f"资源利用率: {utilization:.1%}")

        # 3. 潜在空间可视化
        self._plot_latent_space(results['latent_vectors'])

        # 4. 时间序列对比
        self._plot_temporal_comparison(results)
        # 添加合理性检查
        allocated = results['allocations'] * self.config.total_res
        perfect_match = np.isclose(allocated, results['true_demand'], atol=1e-3).mean()
        print(f"精确匹配比例: {perfect_match:.1%} (正常范围<10%)")

        # 绘制误差分布直方图
        plt.figure()
        errors = results['allocations'] * self.config.total_res - results['true_demand']
        plt.hist(errors.flatten(), bins=50)
        plt.title("分配误差分布")
        plt.savefig("reports/error_dist.png")
        plt.close()

    def _plot_latent_space(self, vectors):
        """潜在空间分布可视化"""
        tsne = TSNE(n_components=2)
        reduced = tsne.fit_transform(vectors)

        plt.figure(figsize=(10, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)
        plt.title("潜在空间t-SNE投影")
        plt.savefig("reports/latent_space.png")
        plt.close()

    def _plot_temporal_comparison(self, results):
        """需求与分配时间序列对比"""
        plt.figure(figsize=(15, 8))

        for i, name in enumerate(['IoT', 'Video', 'Web']):
            plt.subplot(3, 1, i + 1)
            plt.plot(results['true_demand'][:, i], label='真实需求')
            plt.plot(results['allocations'][:, i] * self.config.total_res, '--', label='分配资源')
            plt.title(f"{name}切片资源分析")
            plt.legend()

        plt.tight_layout()
        plt.savefig("reports/temporal_analysis.png")
        plt.close()


if __name__ == "__main__":
    evaluator = ModelEvaluator("models/best_moe_vae.pth")
    eval_results = evaluator.full_evaluation()
    evaluator.generate_report(eval_results)
