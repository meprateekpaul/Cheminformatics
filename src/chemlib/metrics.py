"""chemlib.metrics - Performance metrics for virtual screening, classification, QSAR regression, and scaffold splitting."""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn import metrics as skmetrics


def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute comprehensive classification metrics matching course definitions:
    Sensitivity, Specificity, Accuracy, Precision, F1, MCC, Balanced Accuracy, ROC-AUC, PR-AUC.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tn, fp, fn, tp = skmetrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    balanced_acc = 0.5 * (sensitivity + specificity)

    f1 = (2.0 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0

    roc_auc = np.nan
    pr_auc = np.nan
    if y_prob is not None:
        try:
            roc_auc = float(skmetrics.roc_auc_score(y_true, y_prob))
            pr_auc = float(skmetrics.average_precision_score(y_true, y_prob))
        except Exception:
            pass

    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Sensitivity_Recall": float(sensitivity),
        "Specificity": float(specificity),
        "Precision": float(precision),
        "Accuracy": float(accuracy),
        "Balanced_Accuracy": float(balanced_acc),
        "F1_Score": float(f1),
        "MCC": float(mcc),
        "ROC_AUC": float(roc_auc),
        "PR_AUC": float(pr_auc),
    }


def regression_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, n_features: int = 1) -> Dict[str, float]:
    """Compute comprehensive regression metrics:
    R2, Adjusted R2, RMSE, MAE, Standard error s, Pearson r, Spearman rho.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    r2 = float(skmetrics.r2_score(y_true, y_pred))
    rmse = float(np.sqrt(skmetrics.mean_squared_error(y_true, y_pred)))
    mae = float(skmetrics.mean_absolute_error(y_true, y_pred))

    # Standard error of regression: s = sqrt(sum((y - y_hat)^2) / (n - p - 1))
    deg_freedom = n - n_features - 1
    if deg_freedom > 0:
        s = float(np.sqrt(np.sum((y_true - y_pred) ** 2) / deg_freedom))
        adj_r2 = float(1.0 - (1.0 - r2) * (n - 1) / deg_freedom)
    else:
        s = rmse
        adj_r2 = r2

    # Pearson and Spearman
    try:
        from scipy import stats
        pearson_r, _ = stats.pearsonr(y_true, y_pred)
        spearman_rho, _ = stats.spearmanr(y_true, y_pred)
    except Exception:
        pearson_r = np.nan
        spearman_rho = np.nan

    return {
        "R2": r2,
        "Adjusted_R2": adj_r2,
        "RMSE": rmse,
        "MAE": mae,
        "Standard_Error_s": s,
        "Pearson_r": float(pearson_r),
        "Spearman_rho": float(spearman_rho),
    }


def q2_loo(y_true: np.ndarray, y_pred_loo: np.ndarray) -> float:
    """Compute cross-validated LOO q^2:
    q^2 = 1 - sum((y_obs - y_pred_loo)^2) / sum((y_obs - y_mean)^2)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_loo = np.asarray(y_pred_loo, dtype=float)
    press = np.sum((y_true - y_pred_loo) ** 2)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - (press / tss)) if tss > 0 else 0.0


def r2_pred_external(y_train: np.ndarray, y_test: np.ndarray, y_pred_test: np.ndarray) -> float:
    """Compute Tropsha/Golbraikh external test set R2_pred:
    R2_pred = 1 - sum((y_test - y_pred_test)^2) / sum((y_test - y_train_mean)^2)
    """
    press = np.sum((y_test - y_pred_test) ** 2)
    tss = np.sum((y_test - np.mean(y_train)) ** 2)
    return float(1.0 - (press / tss)) if tss > 0 else 0.0


def enrichment_factor(y_true: np.ndarray, scores: np.ndarray, fraction: float = 0.01) -> float:
    """Compute Enrichment Factor at fraction x (e.g. EF1% for fraction=0.01).
    EF = (Actives in top fraction / Total molecules in top fraction) / (Total actives / Total molecules)
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    n_total = len(y_true)
    n_actives = np.sum(y_true == 1)
    if n_actives == 0 or n_total == 0:
        return 0.0

    k = max(1, int(np.ceil(n_total * fraction)))
    top_indices = np.argsort(scores)[::-1][:k]
    actives_in_top = np.sum(y_true[top_indices] == 1)

    ef = (actives_in_top / k) / (n_actives / n_total)
    return float(ef)


def bedroc_score(y_true: np.ndarray, scores: np.ndarray, alpha: float = 20.0) -> float:
    """Compute Boltzmann-Enhanced Discrimination of ROC (BEDROC) metric.
    alpha=20.0 weights the top 8% of the ranked list.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    n = len(y_true)
    n_actives = np.sum(y_true == 1)
    if n_actives == 0 or n_actives == n:
        return 0.0

    order = np.argsort(scores)[::-1]
    sorted_labels = y_true[order]

    active_ranks = np.where(sorted_labels == 1)[0] + 1  # 1-based ranks
    r_a = n_actives / n

    s = np.sum(np.exp(-alpha * active_ranks / n))
    r_exp = (1.0 - np.exp(-alpha)) / (np.exp(alpha / n) - 1.0)
    ra_exp = (1.0 - np.exp(-alpha * r_a)) / (np.exp(alpha / n) - 1.0)

    bedroc = (s * (1.0 - np.exp(-alpha / n)) / (r_a * (1.0 - np.exp(-alpha)))) - (
        np.exp(-alpha / (n * r_a)) / (np.exp(alpha / n) - 1.0)
    )
    # Normalized version:
    c = (1.0 - np.exp(-alpha * r_a)) / (r_a * (1.0 - np.exp(-alpha)))
    val = (s / (r_a * (1.0 - np.exp(-alpha)) / (np.exp(alpha / n) - 1.0)))
    val_norm = (val - (1.0 / (np.exp(alpha / (n * r_a)) - 1.0))) / (1.0 - (1.0 / (np.exp(alpha / (n * r_a)) - 1.0)))
    return float(np.clip(val_norm, 0.0, 1.0))


def scaffold_split(
    smiles_list: List[str],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Split molecules based on Bemis-Murcko scaffolds to prevent data leakage."""
    rng = np.random.RandomState(seed)
    scaffold_dict: Dict[str, List[int]] = {}

    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            except Exception:
                scaffold = ""
        else:
            scaffold = ""
        scaffold_dict.setdefault(scaffold, []).append(idx)

    scaffold_groups = list(scaffold_dict.values())
    rng.shuffle(scaffold_groups)

    train_indices = []
    test_indices = []
    target_train_count = int(len(smiles_list) * train_ratio)

    for group in scaffold_groups:
        if len(train_indices) + len(group) <= target_train_count:
            train_indices.extend(group)
        else:
            test_indices.extend(group)

    return sorted(train_indices), sorted(test_indices)
