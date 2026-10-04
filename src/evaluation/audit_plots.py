"""Figure functions for the CatBoost audit (300 dpi PNG, consistent style).

Every function takes already-computed arrays/frames, draws one figure, saves it
to ``path`` and returns the path. No metric is typed in: titles and lines are
formatted from the values passed in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

DPI = 300
COLORS = {"CatBoost (final)": "#1f77b4", "CatBoost (raw)": "#1f77b4", "CatBoost (calibrated)": "#17becf",
          "CatBoost (baseline raw)": "#9467bd", "Dummy prior": "#7f7f7f", "Logistic Regression": "#d62728"}
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 100, "savefig.bbox": "tight"})


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _c(name, i=0):
    return COLORS.get(name, f"C{i}")


def fig_target_distribution(path, data: pd.DataFrame, target="Resigned", fold_col="cv_fold"):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    counts = data[target].value_counts().sort_index()
    ax[0].bar(["Stayed (0)", "Resigned (1)"], counts.values, color=["#7f7f7f", "#1f77b4"])
    for i, v in enumerate(counts.values):
        ax[0].text(i, v, f"{v:,}\n({v / counts.sum():.2%})", ha="center", va="bottom")
    ax[0].set(title="Development target distribution", ylabel="Employees (n)")
    ax[0].set_ylim(0, counts.max() * 1.18)
    rate = data.groupby(fold_col)[target].mean()
    ax[1].bar(rate.index.astype(str), rate.values * 100, color="#1f77b4")
    ax[1].axhline(data[target].mean() * 100, color="k", ls="--", label=f"overall {data[target].mean():.2%}")
    for i, (f, v) in enumerate(rate.items()):
        ax[1].text(i, v * 100, f"{v:.3%}", ha="center", va="bottom", fontsize=8)
    ax[1].set(title="Positive rate per outer fold (stratification)", xlabel="Outer fold", ylabel="Resigned rate (%)",
              ylim=(0, rate.max() * 100 * 1.3))
    ax[1].legend()
    return _save(fig, path)


def fig_nested_cv_diagram(path, n_dev: int, n_holdout: int, fold_size: int, inner_val_fraction: float, n_folds: int = 5):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    outer_train = n_dev - fold_size
    inner_val = int(round(outer_train * inner_val_fraction))
    inner_tr = outer_train - inner_val
    ax.set_xlim(0, 100); ax.set_ylim(0, n_folds + 1.6); ax.axis("off"); ax.grid(False)
    for i in range(n_folds):
        y = n_folds - i
        ax.barh(y, 100 * inner_tr / n_dev, left=0, color="#aec7e8", edgecolor="k")
        ax.barh(y, 100 * inner_val / n_dev, left=100 * inner_tr / n_dev, color="#ffbb78", edgecolor="k")
        ax.barh(y, 100 * fold_size / n_dev, left=100 * outer_train / n_dev, color="#98df8a", edgecolor="k")
        ax.text(-1, y, f"Outer {i + 1}", ha="right", va="center")
    ax.text(100 * inner_tr / n_dev / 2, n_folds + 0.9, f"inner train\n{inner_tr:,}", ha="center", va="center")
    ax.text(100 * (inner_tr + inner_val / 2) / n_dev, n_folds + 0.9, f"inner val\n{inner_val:,}", ha="center", va="center")
    ax.text(100 * (outer_train + fold_size / 2) / n_dev, n_folds + 0.9, f"outer val\n{fold_size:,}", ha="center", va="center")
    ax.set_title(f"Nested CV on {n_dev:,} development rows (illustrative layout; folds are interleaved by registry). "
                 f"Formal holdout: {n_holdout:,} rows, never loaded.", fontsize=9)
    return _save(fig, path)


def roc_band(y, p, n_boot=200, seed=42, grid=np.linspace(0, 1, 101)):
    rng = np.random.default_rng(seed)
    tprs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        f, t, _ = roc_curve(y[idx], p[idx])
        tprs.append(np.interp(grid, f, t))
    return grid, np.percentile(tprs, [2.5, 97.5], axis=0)


def fig_roc(path, y, preds: Mapping[str, np.ndarray], aucs: Mapping[str, tuple[float, float, float]], n_boot=200):
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    for i, (name, p) in enumerate(preds.items()):
        f, t, _ = roc_curve(y, p)
        a, lo, hi = aucs[name]
        ax.plot(f, t, color=_c(name, i), label=f"{name} AUC={a:.4f} [{lo:.4f}, {hi:.4f}]")
        if "Dummy" not in name:
            g, band = roc_band(np.asarray(y), np.asarray(p), n_boot=n_boot)
            ax.fill_between(g, band[0], band[1], color=_c(name, i), alpha=0.15)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set(title="ROC (pooled OOF, bootstrap 95% bands)", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(fontsize=7, loc="lower right")
    return _save(fig, path)


def fig_pr(path, y, preds: Mapping[str, np.ndarray], aps: Mapping[str, tuple[float, float, float]]):
    y = np.asarray(y)
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    for i, (name, p) in enumerate(preds.items()):
        pr, rc, _ = precision_recall_curve(y, p)
        a, lo, hi = aps[name]
        ax.plot(rc, pr, color=_c(name, i), label=f"{name} AP={a:.4f} [{lo:.4f}, {hi:.4f}]")
    ax.axhline(y.mean(), color="k", ls="--", lw=1, label=f"no-skill = prevalence {y.mean():.4f}")
    ax.set(title="Precision-Recall (pooled OOF)", xlabel="Recall", ylabel="Precision", ylim=(0, max(0.35, y.mean() * 2.5)))
    ax.legend(fontsize=7)
    return _save(fig, path)


def fig_fold_strip(path, fold_tables: Mapping[str, pd.DataFrame], prevalence: float):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    names = list(fold_tables)
    for k, (metric, ref, lab) in enumerate((("pr_auc", prevalence, "prevalence"), ("roc_auc", 0.5, "chance"))):
        data = [fold_tables[n][metric].to_numpy() for n in names]
        ax[k].boxplot(data, tick_labels=names, showfliers=False)
        for i, d in enumerate(data):
            ax[k].scatter(np.full(len(d), i + 1) + np.linspace(-0.12, 0.12, len(d)), d, color=_c(names[i], i), zorder=3)
        ax[k].axhline(ref, color="k", ls="--", lw=1, label=lab)
        ax[k].set(title=f"Per-fold {metric.replace('_', '-').upper()} (5 outer folds)", ylabel=metric)
        ax[k].tick_params(axis="x", rotation=20)
        ax[k].legend()
    return _save(fig, path)


def fig_bootstrap_delta(path, deltas: Mapping[str, np.ndarray], title: str):
    fig, ax = plt.subplots(1, len(deltas), figsize=(5 * len(deltas), 3.8))
    ax = np.atleast_1d(ax)
    for a, (name, d) in zip(ax, deltas.items()):
        a.hist(d, bins=40, color="#1f77b4", alpha=0.8)
        lo, hi = np.percentile(d, [2.5, 97.5])
        a.axvline(0, color="k", lw=1.5, label="zero")
        a.axvline(lo, color="r", ls="--"); a.axvline(hi, color="r", ls="--", label=f"95% CI [{lo:.4f}, {hi:.4f}]")
        a.set(title=name, xlabel="paired bootstrap difference", ylabel="resamples")
        a.legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    return _save(fig, path)


def fig_learning_curves(path, curves: Mapping[str, Mapping], selected: Mapping[str, int], fold_label: str):
    """curves[candidate] = {metric_name: [value per iteration]} for one outer fold's inner validation.

    One panel per metric present in the data (typically the early-stopping metric).
    """
    metrics = sorted({m for c in curves.values() for m in c})
    fig, ax = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 4.2), squeeze=False)
    ax = ax[0]
    for i, (cid, c) in enumerate(curves.items()):
        for k, metric in enumerate(metrics):
            if metric in c:
                ax[k].plot(np.arange(1, len(c[metric]) + 1), c[metric], color=f"C{i % 10}", label=cid, lw=1)
                if cid in selected and selected[cid] and selected[cid] <= len(c[metric]):
                    ax[k].scatter([selected[cid]], [c[metric][selected[cid] - 1]], color=f"C{i % 10}", marker="*", s=90, zorder=4)
    for k, metric in enumerate(metrics):
        ax[k].set(title=f"Inner-validation {metric} ({fold_label}; star = early-stop pick)", xlabel="iteration", ylabel=metric, xscale="log")
    ax[-1].legend(fontsize=6, ncol=2)
    return _save(fig, path)


def fig_candidate_comparison(path, cand: pd.DataFrame, value_col: str, ylabel: str, ref: float | None = None, ref_label=""):
    g = cand.groupby("candidate_id")[value_col].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(g.index, g["mean"], yerr=g["std"], capsize=3, color="#1f77b4")
    if ref is not None:
        ax.axhline(ref, color="k", ls="--", label=ref_label); ax.legend()
    lo = (g["mean"] - g["std"]).min()
    ax.set_ylim(max(0, lo - 0.01), (g["mean"] + g["std"]).max() + 0.01)
    ax.set(title=f"Candidate comparison on inner validation (mean ± sd over outer folds)", ylabel=ylabel)
    return _save(fig, path)


def fig_tree_ablation(path, abl: pd.DataFrame, prevalence: float, base_logloss: float):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(abl.trees, abl.pr_auc, "o-"); ax[0].axhline(prevalence, color="k", ls="--", label="prevalence")
    ax[0].set(xscale="log", title="OOF PR-AUC vs number of trees", xlabel="trees", ylabel="PR-AUC"); ax[0].legend()
    ax[1].plot(abl.trees, abl.log_loss, "o-"); ax[1].axhline(base_logloss, color="k", ls="--", label="prior-only log loss")
    ax[1].set(xscale="log", title="OOF Log Loss vs number of trees", xlabel="trees", ylabel="log loss"); ax[1].legend()
    return _save(fig, path)


def fig_reliability(path, y, preds: Mapping[str, np.ndarray], bins=10):
    y = np.asarray(y)
    fig, ax = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1.2]}, sharex=True)
    hi = 0
    for i, (name, p) in enumerate(preds.items()):
        q = pd.qcut(pd.Series(p).rank(method="first"), q=bins, labels=False)
        d = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q").agg(p=("p", "mean"), y=("y", "mean"))
        ax[0].plot(d.p, d.y, "o-", color=_c(name, i), label=name)
        ax[1].hist(p, bins=40, alpha=0.5, color=_c(name, i), label=name)
        hi = max(hi, d.p.max(), d.y.max())
    lim = max(hi * 1.1, 0.2)
    ax[0].plot([0, lim], [0, lim], "k--", label="perfect"); ax[0].set(ylabel="observed resignation rate (decile)", title="Reliability (decile bins, pooled OOF)")
    ax[0].legend(fontsize=8)
    ax[1].set(xlabel="mean predicted probability", ylabel="count"); ax[1].set_yscale("log"); ax[1].legend(fontsize=7)
    return _save(fig, path)


def fig_prob_by_class(path, y, p):
    y = np.asarray(y)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bins = np.linspace(p.min(), p.max() if p.max() > p.min() else p.min() + 1e-6, 50)
    ax.hist(p[y == 0], bins=bins, alpha=0.6, density=True, label="Stayed (0)", color="#7f7f7f")
    ax.hist(p[y == 1], bins=bins, alpha=0.6, density=True, label="Resigned (1)", color="#1f77b4")
    ax.set(title="Predicted-probability distribution by true class (final model, pooled OOF)", xlabel="predicted probability", ylabel="density")
    ax.legend()
    return _save(fig, path)


def fig_gain_lift(path, y, preds: Mapping[str, np.ndarray], tb: Mapping[str, pd.DataFrame]):
    y = np.asarray(y)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for i, (name, t) in enumerate(tb.items()):
        ax[0].plot(t.top_k * 100, t.recall * 100, "o-", color=_c(name, i), label=name)
        ax[1].plot(t.top_k * 100, t.lift, "o-", color=_c(name, i), label=name)
    ks = next(iter(tb.values())).top_k.to_numpy()
    ax[0].plot(ks * 100, ks * 100, "k--", label="random")
    ax[1].axhline(1, color="k", ls="--", label="random")
    ax[0].set(title="Cumulative gain: recall@top-k", xlabel="% employees flagged (exact capacity)", ylabel="% of resignations captured"); ax[0].legend(fontsize=7)
    ax[1].set(title="Lift@top-k", xlabel="% employees flagged", ylabel="lift vs base rate"); ax[1].legend(fontsize=7)
    return _save(fig, path)


def fig_importance(path, pvc: pd.DataFrame, perm: pd.DataFrame):
    """pvc/perm: rows=feature, columns=fold -> value."""
    order = pvc.mean(axis=1).sort_values().index
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
    ax[0].barh(order, pvc.loc[order].mean(axis=1), xerr=pvc.loc[order].std(axis=1), color="#1f77b4")
    ax[0].set(title="PredictionValuesChange (mean ± sd over 5 fold models)", xlabel="importance (%)")
    o2 = perm.mean(axis=1).sort_values().index
    ax[1].barh(o2, perm.loc[o2].mean(axis=1), xerr=perm.loc[o2].std(axis=1), color="#d62728")
    ax[1].axvline(0, color="k", lw=1)
    ax[1].set(title="Permutation importance on outer-val (ΔPR-AUC; mean ± sd over folds)", xlabel="drop in PR-AUC when shuffled")
    return _save(fig, path)


def fig_shap(path, shap_vals: np.ndarray, X_display: pd.DataFrame, numeric_cols: Sequence[str]):
    mean_abs = pd.Series(np.abs(shap_vals).mean(axis=0), index=X_display.columns).sort_values()
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
    ax[0].barh(mean_abs.index, mean_abs.values, color="#1f77b4"); ax[0].set(title="Mean |SHAP| (log-odds)", xlabel="mean |SHAP|")
    rng = np.random.default_rng(42)
    for yi, col in enumerate(mean_abs.index):
        j = list(X_display.columns).index(col)
        v = X_display[col]
        if col in numeric_cols:
            c = (v - v.min()) / (v.max() - v.min() + 1e-12)
            sc = ax[1].scatter(shap_vals[:, j], yi + rng.uniform(-0.3, 0.3, len(v)), c=c, cmap="coolwarm", s=3)
        else:
            ax[1].scatter(shap_vals[:, j], yi + rng.uniform(-0.3, 0.3, len(v)), c="#999999", s=3)
    ax[1].set_yticks(range(len(mean_abs))); ax[1].set_yticklabels(mean_abs.index)
    ax[1].set(title="SHAP values per row (numeric: colour = feature value)", xlabel="SHAP value")
    return _save(fig, path)


def fig_single_feature_signal(path, scan: pd.DataFrame):
    s = scan.sort_values("auc_distance_from_chance")
    fig, ax = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True)
    ax[0].barh(s.feature, s.single_feature_auc - 0.5, left=0.5, color="#1f77b4")
    ax[0].axvspan(0.4, 0.6, color="green", alpha=0.08, label="no-signal band (0.4–0.6)")
    ax[0].set(xlim=(0.38, 0.62), title="Single-feature ROC-AUC", xlabel="AUC"); ax[0].legend(fontsize=7)
    ax[1].barh(s.feature, s.pearson_r.abs().fillna(0), color="#ff7f0e"); ax[1].set(title="|Pearson r| (numeric only)", xlabel="|r|")
    ax[2].barh(s.feature, s.cramers_v, color="#2ca02c"); ax[2].set(title="Cramér's V (numeric: 10 quantile bins)", xlabel="V")
    return _save(fig, path)


def fig_permutation_test(path, null: Sequence[float], real: float, metric: str, p_value: float):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(null, bins=min(15, max(5, len(null) // 2)), color="#7f7f7f", alpha=0.8, label=f"shuffled labels (n={len(null)})")
    ax.axvline(real, color="r", lw=2, label=f"real labels = {real:.5f}")
    ax.set(title=f"Label-shuffle test, full nested protocol ({metric}); permutation p = {p_value:.3f}", xlabel=f"pooled OOF {metric}", ylabel="permutations")
    ax.legend()
    return _save(fig, path)


def fig_subgroups(path, tables: Mapping[str, pd.DataFrame]):
    fig, ax = plt.subplots(1, len(tables), figsize=(5 * len(tables), 4), sharey=True)
    ax = np.atleast_1d(ax)
    for a, (col, t) in zip(ax, tables.items()):
        x = np.arange(len(t))
        a.bar(x - 0.2, t.average_precision, 0.4, label="AP", color="#1f77b4")
        a.bar(x + 0.2, t.roc_auc, 0.4, label="ROC-AUC", color="#ff7f0e")
        a.scatter(x - 0.2, t.prevalence, color="k", marker="_", s=300, label="group prevalence", zorder=4)
        a.axhline(0.5, color="gray", ls=":")
        a.set_xticks(x); a.set_xticklabels(t[col], rotation=35, ha="right"); a.set_title(f"By {col} (descriptive)")
    ax[0].legend(fontsize=7)
    return _save(fig, path)
