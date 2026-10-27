"""
Competition Metric (wMAE), Masked Multi-Task Loss, and Tanimoto Sphere-Exclusion Clustering Splits.
"""
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import KFold

TARGET_COLS = ["Tg", "FFV", "Tc", "Density", "Rg"]

TRAIN_STATS = {
    "Tg":      {"min": -148.029738, "max": 472.250000, "range": 620.279738, "count": 511},
    "FFV":     {"min":    0.226992, "max":   0.777097, "range":   0.550105, "count": 7030},
    "Tc":      {"min":    0.046500, "max":   0.524000, "range":   0.477500, "count": 737},
    "Density": {"min":    0.748691, "max":   1.840999, "range":   1.092308, "count": 613},
    "Rg":      {"min":    9.728355, "max":  34.672906, "range":  24.944551, "count": 614},
}


def compute_wmae_weights(counts=None, ranges=None):
    if ranges is None:
        ranges = np.array([TRAIN_STATS[c]["range"] for c in TARGET_COLS], dtype=np.float64)
    if counts is None:
        counts = np.array([TRAIN_STATS[c]["count"] for c in TARGET_COLS], dtype=np.float64)
    K = float(len(TARGET_COLS))
    inv_sqrt = np.sqrt(1.0 / np.maximum(counts, 1.0))
    reweight = (K * inv_sqrt) / np.sum(inv_sqrt)
    return ((1.0 / ranges) * reweight).astype(np.float32)


def evaluate_predictions(y_true, y_pred, mask=None):
    if mask is None:
        mask = ~np.isnan(y_true)
    else:
        mask = mask.astype(bool) & (~np.isnan(y_true))
    ranges = np.array([TRAIN_STATS[c]["range"] for c in TARGET_COLS], dtype=np.float64)
    train_counts = np.array([TRAIN_STATS[c]["count"] for c in TARGET_COLS], dtype=np.float64)
    w_train = compute_wmae_weights(counts=train_counts, ranges=ranges)

    abs_err = np.where(mask, np.abs(y_pred - y_true), 0.0)
    results = {}
    per_prop_mae = []
    for i, col in enumerate(TARGET_COLS):
        n_i = mask[:, i].sum()
        mae_i = float(abs_err[:, i].sum() / max(n_i, 1))
        results[f"MAE_{col}"] = mae_i
        per_prop_mae.append(mae_i)
    per_prop_mae = np.array(per_prop_mae, dtype=np.float64)

    inv_sqrt = np.sqrt(1.0 / train_counts)
    norm_w = inv_sqrt / np.sum(inv_sqrt)
    results["wMAE"] = float(np.sum((per_prop_mae / ranges) * norm_w))

    has_any = mask.any(axis=1)
    n_valid = max(int(has_any.sum()), 1)
    results["wMAE_sample_avg"] = float((abs_err * w_train.reshape(1, -1))[has_any].sum() / n_valid)
    return results


class MaskedMultiTaskWMAELoss(nn.Module):
    def __init__(self, y_mean, y_std, task_counts=None, beta=0.05):
        super().__init__()
        w = compute_wmae_weights(counts=task_counts)
        eff_w = w * y_std.astype(np.float32)
        eff_w = eff_w / eff_w.mean()
        self.register_buffer("eff_w", torch.tensor(eff_w, dtype=torch.float32))
        self.register_buffer("y_mean", torch.tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.tensor(y_std, dtype=torch.float32))
        self.beta = beta

    def forward(self, z_pred, y_true, mask):
        y_clean = torch.nan_to_num(y_true, nan=0.0)
        z_true = (y_clean - self.y_mean) / self.y_std
        diff = torch.abs(z_pred - z_true)
        loss_elem = torch.where(diff < self.beta, 0.5 * (diff ** 2) / self.beta, diff - 0.5 * self.beta)
        loss_elem = loss_elem * mask
        task_counts_batch = mask.sum(dim=0)
        active = task_counts_batch > 0
        if not active.any():
            return (z_pred * 0.0).sum()
        task_losses = loss_elem.sum(dim=0)[active] / task_counts_batch[active]
        return (task_losses * self.eff_w[active]).mean()
