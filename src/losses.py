"""Reconstruction and line-structure losses with explicit configurable weights."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self, l1=1.0, mse=0.15, edge=0.35, ssim=0.0):
        super().__init__()
        self.weights = {"l1": float(l1), "mse": float(mse), "edge": float(edge), "ssim": float(ssim)}

    @staticmethod
    def _gradient(x):
        dx = x[..., :, 1:] - x[..., :, :-1]
        dy = x[..., 1:, :] - x[..., :-1, :]
        return dx, dy

    @staticmethod
    def _structural_loss(pred, target):
        # Differentiable local SSIM-style dissimilarity; no optional dependency.
        mu_p = F.avg_pool2d(pred, 7, stride=1, padding=3)
        mu_t = F.avg_pool2d(target, 7, stride=1, padding=3)
        var_p = F.avg_pool2d(pred * pred, 7, 1, 3) - mu_p.square()
        var_t = F.avg_pool2d(target * target, 7, 1, 3) - mu_t.square()
        cov = F.avg_pool2d(pred * target, 7, 1, 3) - mu_p * mu_t
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        score = ((2 * mu_p * mu_t + c1) * (2 * cov + c2)) / ((mu_p.square() + mu_t.square() + c1) * (var_p + var_t + c2) + 1e-8)
        return ((1 - score) / 2).clamp(0, 1).mean()

    def forward(self, pred, target):
        parts = {
            "l1": F.l1_loss(pred, target),
            "mse": F.mse_loss(pred, target),
            "edge": sum(F.l1_loss(a, b) for a, b in zip(self._gradient(pred), self._gradient(target))) * 0.5,
            "ssim": self._structural_loss(pred, target) if self.weights["ssim"] else pred.new_zeros(()),
        }
        total = sum(parts[k] * self.weights[k] for k in parts)
        return total, {k: float(v.detach()) for k, v in parts.items()}

