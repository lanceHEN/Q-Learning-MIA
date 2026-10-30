import glob
import json
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import auc, roc_curve

LOW_FPRS = (0.001, 0.005, 0.01, 0.05, 0.1)


def tpr_at_fprs(labels: np.ndarray, scores: np.ndarray, fprs: Sequence[float] = LOW_FPRS) -> np.ndarray:
    """
    TPR at each target FPR, interpolating linearly between ROC points. Tied
    scores form a single segment of the ROC curve, so this is the expected TPR
    when ties are broken at random, which an attacker can always achieve.
    Scores are negated since lower scores mean "more likely a member".
    """
    fpr, tpr, _ = roc_curve(labels, -np.asarray(scores))
    return np.interp(np.asarray(fprs), fpr, tpr)


def load_runs(pattern: str) -> List[Dict]:
    """
    Loads raw run files (.npz) matching a glob pattern, e.g.
    "data/raw_results/cluster-sepsis-tabular-ensemble_(10,1000,*".
    Each run is returned as a dict of arrays plus its parsed "meta" dict.
    """
    runs = []
    for path in sorted(glob.glob(pattern)):
        with np.load(path, allow_pickle=True) as f:
            run = {key: f[key] for key in f.files}
        run["meta"] = json.loads(str(run["meta"]))
        run["path"] = path
        runs.append(run)
    return runs


def _scores(run: Dict) -> np.ndarray:
    """
    Per-trajectory scores for one saved run. Files saved before output perturbation
    was removed hold one row per perturbation std; the noise-free row (std 0) is used.
    """
    scores = run["scores"]
    if scores.ndim == 1:
        return scores
    matches = np.flatnonzero(np.isclose(run["stds"], 0.0))
    if len(matches) == 0:
        raise ValueError(f"No std 0 row in run {run['path']} (has {run['stds']})")
    return scores[int(matches[0])]


def run_tprs(run: Dict, fprs: Sequence[float] = LOW_FPRS) -> np.ndarray:
    """TPR at the given FPRs for one saved run."""
    return tpr_at_fprs(run["labels"], _scores(run), fprs)


def average_roc(
    runs: List[Dict],
    fpr_grid: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregated ROC curve by vertical averaging: each run's TPR is evaluated on a
    shared FPR grid (as in tpr_at_fprs), then averaged across runs. Returns (fpr_grid, mean_tpr, std_tpr). The default grid is log-spaced
    down to the smallest FPR resolvable in every run (1 / n_nonmembers).
    """
    if fpr_grid is None:
        min_fpr = max(1.0 / np.sum(run["labels"] == 0) for run in runs)
        fpr_grid = np.logspace(np.log10(min_fpr), 0, 200)
    tprs = np.array([run_tprs(run, fpr_grid) for run in runs])
    return fpr_grid, tprs.mean(axis=0), tprs.std(axis=0)


Curves = Union[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray, np.ndarray]]


def plot_roc(curves: Curves, ax=None, title: str = None, log: bool = False):
    """
    Plots aggregated ROC curves with a +/- 1 std band, on linear axes or (log=True)
    log-log axes, and prints each curve's AUC (also shown in the legend). `curves`
    maps a legend label to the output of average_roc, or is a single such output.
    """
    import matplotlib.pyplot as plt

    if isinstance(curves, tuple):
        curves = {"attack": curves}
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    lo = min(fpr[0] for fpr, _, _ in curves.values()) if log else 0.0
    floor = 1e-4 if log else 0.0
    for label, (fpr, mean_tpr, std_tpr) in curves.items():
        # AUC of the mean curve, i.e. the mean of the runs' AUCs on the same FPR grid;
        # the curve is extended to the origin since the default grid stops at 1 / n_nonmembers
        roc_auc = auc(np.r_[0.0, fpr], np.r_[0.0, mean_tpr])
        print(f"{label}: AUC = {roc_auc:.3f}")
        if not log:
            fpr, mean_tpr, std_tpr = np.r_[0.0, fpr], np.r_[0.0, mean_tpr], np.r_[0.0, std_tpr]
        ax.plot(fpr, mean_tpr, label=f"{label} (AUC = {roc_auc:.3f})")
        ax.fill_between(fpr, np.clip(mean_tpr - std_tpr, floor, 1), np.clip(mean_tpr + std_tpr, floor, 1), alpha=0.2)
    ax.plot([lo, 1], [lo, 1], "k--", linewidth=1, label="random")
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlim(lo, 1)
    ax.set_ylim(lo, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    if title:
        ax.set_title(title)
    ax.legend()
    return ax


def plot_log_roc(curves: Curves, ax=None, title: str = None):
    """plot_roc on log-log axes, which shows the low-FPR region."""
    return plot_roc(curves, ax=ax, title=title, log=True)
