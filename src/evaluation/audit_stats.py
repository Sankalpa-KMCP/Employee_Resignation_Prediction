"""Reusable statistics for the CatBoost audit: metrics, bootstrap, scans, calibration.

Every function is pure (no file writes) and receives the arrays it needs, so the
audit scripts, notebook and tests can share one implementation. Nothing here
reads the formal holdout: inputs come from ``load_audit_inputs`` which delegates
to the hash-verified development loader that skips holdout rows before parsing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    mutual_info_score,
    roc_auc_score,
)

DEFAULT_KS = (0.01, 0.05, 0.10, 0.20, 0.30, 0.50)
BOOT_METRICS = ("pr_auc", "roc_auc", "brier", "log_loss")


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
def load_audit_inputs(root: Path, all_columns: bool = False):
    """Return (dev_frame_with_cv_fold, split_registry, predictors, categoricals).

    Uses ``load_development_data`` so formal-holdout predictors and labels are
    never parsed. ``all_columns`` also keeps excluded columns for leakage scans.
    """
    from src.evaluation.inputs import load_development_data

    root = Path(root)
    contract = yaml.safe_load((root / "configs/feature_contract.yaml").read_text(encoding="utf-8"))
    feats = list(contract["primary_model_features"])
    cats = [c["name"] for c in contract["categorical_features"]]
    usecols = None if all_columns else feats
    data, split = load_development_data(root, usecols=usecols)
    data = data.merge(split[["Employee_ID", "cv_fold"]], on="Employee_ID", validate="one_to_one")
    data["cv_fold"] = data["cv_fold"].astype(int)
    return data.sort_values("Employee_ID").reset_index(drop=True), split, feats, cats


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def metric_row(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Project-standard metrics for one set of probabilities."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "balanced_accuracy": float(balanced_accuracy_score(y, (p >= threshold).astype(int))),
    }


def topk_table(y, p, ks: Sequence[float] = DEFAULT_KS, tiebreak_seed: int = 42) -> pd.DataFrame:
    """Recall@k, precision@k and lift@k with exact capacity and seeded random tie-breaks.

    Exact capacity means repeated probabilities cannot inflate the selected count.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(tiebreak_seed)
    order = np.lexsort((rng.random(len(p)), -p))
    base_rate = y.mean()
    rows = []
    for k in ks:
        n_sel = max(1, int(round(k * len(y))))
        hit = int(y[order[:n_sel]].sum())
        precision = hit / n_sel
        rows.append(
            {"top_k": k, "n_selected": n_sel, "recall": hit / max(1, int(y.sum())),
             "precision": precision, "lift": precision / base_rate}
        )
    return pd.DataFrame(rows)


def per_fold_metrics(oof: pd.DataFrame, prob_col: str = "y_probability") -> pd.DataFrame:
    """Metric rows per outer fold, computed from saved OOF rows."""
    rows = []
    for fold, part in oof.groupby("cv_fold"):
        rows.append({"fold": int(fold), **metric_row(part["y_true"], part[prob_col])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Bootstrap and paired comparison
# ----------------------------------------------------------------------------
def bootstrap_pooled(
    y: np.ndarray,
    preds: Mapping[str, np.ndarray],
    n_boot: int = 1000,
    seed: int = 42,
    metrics: Iterable[str] = BOOT_METRICS,
) -> dict[str, dict[str, np.ndarray]]:
    """Row-resample the pooled OOF set; all models share the same resample indices.

    Sharing indices makes the difference between two models a paired statistic.
    """
    y = np.asarray(y, dtype=int)
    n = len(y)
    rng = np.random.default_rng(seed)
    metrics = tuple(metrics)
    out = {name: {m: np.empty(n_boot) for m in metrics} for name in preds}
    arrays = {name: np.asarray(p, dtype=float) for name, p in preds.items()}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():  # vanishingly unlikely at 10% prevalence
            idx = rng.integers(0, n, n)
            yb = y[idx]
        for name, p in arrays.items():
            pb = p[idx]
            row = metric_row(yb, pb)
            for m in metrics:
                out[name][m][b] = row[m]
    return out


def ci95(samples: np.ndarray) -> tuple[float, float]:
    """Percentile 95% interval."""
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def paired_fold_comparison(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """Paired fold-wise difference a - b with a two-sided t-test and t-interval.

    With only five folds the test has little power; it is reported alongside, not
    instead of, the pooled paired bootstrap.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    n = len(diff)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else float("nan")
    if n > 1 and sd > 0:
        se = sd / np.sqrt(n)
        t_stat = mean / se
        p_val = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
        half = float(stats.t.ppf(0.975, df=n - 1) * se)
    else:
        t_stat, p_val, half = float("nan"), float("nan"), float("nan")
    return {"mean_diff": mean, "sd_diff": sd, "t_stat": float(t_stat), "p_value": p_val,
            "ci_lower": mean - half, "ci_upper": mean + half, "n_folds": n}


def better_than_baseline(
    delta_pr_ci: tuple[float, float], delta_roc_ci: tuple[float, float],
    delta_pr: float, delta_roc: float, min_pr: float = 0.01, min_roc: float = 0.01,
) -> bool:
    """Pre-declared verdict rule: both CI lower bounds > 0 AND both effects practically meaningful."""
    return bool(
        delta_pr_ci[0] > 0 and delta_roc_ci[0] > 0 and delta_pr >= min_pr and delta_roc >= min_roc
    )


def permutation_pvalue(real: float, null: Sequence[float], greater_is_better: bool = True) -> float:
    """Add-one permutation p-value: P(null >= real) (or <= when lower is better)."""
    null = np.asarray(null, dtype=float)
    hits = (null >= real).sum() if greater_is_better else (null <= real).sum()
    return float((1 + hits) / (1 + len(null)))


# ----------------------------------------------------------------------------
# Calibration (cross-fitted over saved OOF predictions)
# ----------------------------------------------------------------------------
def crossfit_calibration(oof: pd.DataFrame, method: str, prob_col: str = "y_probability") -> np.ndarray:
    """Calibrate each outer fold with a map fitted only on the other folds' OOF rows.

    Supported: ``none``, ``platt`` (logistic on logit), ``isotonic``. The fold
    being scored is never in its own calibrator's fit set. A second-order caveat
    remains: other folds' models were trained on rows that include the scored
    fold, so this is standard OOF cross-fitting, not nested calibration.
    """
    p = oof[prob_col].to_numpy(dtype=float)
    y = oof["y_true"].to_numpy(dtype=int)
    folds = oof["cv_fold"].to_numpy(dtype=int)
    if method == "none":
        return p.copy()
    out = np.empty(len(p))
    for f in np.unique(folds):
        tr, va = folds != f, folds == f
        if method == "platt":
            clip = lambda v: np.clip(v, 1e-6, 1 - 1e-6)
            logit = lambda v: np.log(clip(v) / (1 - clip(v))).reshape(-1, 1)
            cal = LogisticRegression(C=1e5, solver="lbfgs").fit(logit(p[tr]), y[tr])
            out[va] = cal.predict_proba(logit(p[va]))[:, 1]
        elif method == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p[tr], y[tr])
            out[va] = cal.predict(p[va])
        else:
            raise ValueError(f"Unknown calibration method {method!r}")
    return out


# ----------------------------------------------------------------------------
# Leakage and signal scans
# ----------------------------------------------------------------------------
def cramers_v(a: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramer's V between two discrete variables."""
    table = pd.crosstab(a, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def univariate_scan(
    df: pd.DataFrame, columns: Sequence[str], categorical: Sequence[str], target: str = "Resigned",
    fold_col: str = "cv_fold",
) -> pd.DataFrame:
    """One row per column: single-feature AUC, correlation, mutual information, Cramer's V.

    Categorical AUC uses fold-local target encoding (level means from the other
    folds only), so the scan itself cannot leak labels. Numeric Cramer's V and MI
    use 10 quantile bins. ``suspicious`` flags single-feature AUC > 0.6 or < 0.4.
    """
    y = df[target].to_numpy(dtype=int)
    folds = df[fold_col].to_numpy()
    rows = []
    for col in columns:
        s = df[col]
        row = {"feature": col, "type": "categorical" if col in categorical else "numeric"}
        if col in categorical:
            enc = np.empty(len(df))
            for f in np.unique(folds):
                tr, va = folds != f, folds == f
                means = pd.Series(y[tr]).groupby(s[tr].to_numpy()).mean()
                enc[va] = s[va].map(means).fillna(y[tr].mean()).to_numpy()
            auc = roc_auc_score(y, enc)
            binned = s.astype(str)
            row.update(pearson_r=np.nan, pearson_p=np.nan, spearman_rho=np.nan, spearman_p=np.nan)
        else:
            x = pd.to_numeric(s, errors="coerce") if s.dtype.kind not in "biuf" else s
            if x.dtype.kind not in "biuf" or x.isna().all():
                x = pd.to_datetime(s, errors="coerce").astype("int64")  # e.g. Hire_Date
            x = x.to_numpy(dtype=float)
            auc = roc_auc_score(y, x)
            pr, pp = stats.pearsonr(x, y)
            sr, sp = stats.spearmanr(x, y)
            row.update(pearson_r=float(pr), pearson_p=float(pp), spearman_rho=float(sr), spearman_p=float(sp))
            binned = pd.qcut(pd.Series(x), q=10, duplicates="drop").astype(str)
        row["single_feature_auc"] = float(auc)
        row["auc_distance_from_chance"] = float(abs(auc - 0.5))
        row["mutual_info_nats"] = float(mutual_info_score(binned.to_numpy(), y))
        row["cramers_v"] = cramers_v(pd.Series(binned.to_numpy()), pd.Series(y))
        row["suspicious"] = bool(auc > 0.6 or auc < 0.4)
        rows.append(row)
    return pd.DataFrame(rows)


def duplicate_audit(df: pd.DataFrame, features: Sequence[str], fold_col: str = "cv_fold") -> dict[str, float]:
    """Exact duplicate feature vectors overall and across each fold's train/validation boundary."""
    hashes = pd.util.hash_pandas_object(df[list(features)].astype(str), index=False).to_numpy()
    folds = df[fold_col].to_numpy()
    cross = {}
    for f in np.unique(folds):
        cross[int(f)] = int(np.isin(hashes[folds == f], hashes[folds != f]).sum())
    return {
        "duplicate_feature_rows": int(pd.Series(hashes).duplicated().sum()),
        "cross_boundary_duplicates_by_fold": cross,
        "cross_boundary_duplicates_total": int(sum(cross.values())),
    }


def near_duplicate_audit(
    df: pd.DataFrame, numeric: Sequence[str], fold: int = 1, sample: int = 2000, seed: int = 42,
) -> dict[str, float]:
    """Nearest train-neighbour distance (standardised numeric space) for a validation sample.

    Reports the minimum and 1st percentile; values near zero would indicate
    near-copies of training rows. Fitted statistics come from the training rows only.
    """
    from sklearn.neighbors import NearestNeighbors

    tr = df[df["cv_fold"] != fold]
    va = df[df["cv_fold"] == fold].sample(n=min(sample, int((df["cv_fold"] == fold).sum())), random_state=seed)
    mu, sd = tr[list(numeric)].mean(), tr[list(numeric)].std().replace(0, 1)
    nn = NearestNeighbors(n_neighbors=1).fit(((tr[list(numeric)] - mu) / sd).to_numpy())
    dist = nn.kneighbors(((va[list(numeric)] - mu) / sd).to_numpy())[0].ravel()
    return {"nn_dist_min": float(dist.min()), "nn_dist_p01": float(np.percentile(dist, 1)),
            "nn_dist_median": float(np.median(dist)), "fraction_below_1e-6": float((dist < 1e-6).mean())}


# ----------------------------------------------------------------------------
# One-call model comparison (shared by the Phase 1 audit and the final audit)
# ----------------------------------------------------------------------------
def compare_models(
    oofs: Mapping[str, pd.DataFrame], reference: str, n_boot: int = 1000, seed: int = 42,
    min_effect: float = 0.01, ks: Sequence[float] = DEFAULT_KS,
):
    """Metrics, bootstrap CIs, top-k and paired differences vs ``reference`` for aligned OOF frames.

    Every frame needs Employee_ID, cv_fold, y_true, y_probability in identical row order
    (asserted). Returns (summary, paired, topk, fold_table, boot) where ``boot`` keeps the
    raw resample arrays so callers can plot paired difference distributions.
    """
    names = list(oofs)
    base = oofs[names[0]]
    for name, f in oofs.items():
        for col in ("Employee_ID", "y_true", "cv_fold"):
            if not (f[col].to_numpy() == base[col].to_numpy()).all():
                raise ValueError(f"{name}: {col} is not row-aligned with {names[0]}")
    y = base["y_true"].to_numpy()
    preds = {n: f["y_probability"].to_numpy() for n, f in oofs.items()}
    fold_tab = {n: per_fold_metrics(f) for n, f in oofs.items()}
    boot = bootstrap_pooled(y, preds, n_boot, seed)
    pooled = {n: metric_row(y, preds[n]) for n in names}
    rows, tk = [], []
    for n in names:
        row = {"model": n, **{f"pooled_{k}": v for k, v in pooled[n].items()}}
        for m in BOOT_METRICS:
            lo, hi = ci95(boot[n][m])
            row.update({f"{m}_ci_lower": lo, f"{m}_ci_upper": hi,
                        f"fold_mean_{m}": fold_tab[n][m].mean(), f"fold_std_{m}": fold_tab[n][m].std(ddof=1)})
        row["fold_mean_balanced_accuracy"] = fold_tab[n]["balanced_accuracy"].mean()
        rows.append(row)
        t = topk_table(y, preds[n], ks, seed)
        t.insert(0, "model", n)
        tk.append(t)
    paired = []
    for n in names:
        if n == reference:
            continue
        row = {"model": n, "reference": reference}
        for m in BOOT_METRICS:
            lo, hi = ci95(boot[n][m] - boot[reference][m])
            fw = paired_fold_comparison(fold_tab[n][m], fold_tab[reference][m])
            row.update({f"delta_{m}": float(pooled[n][m] - pooled[reference][m]),
                        f"delta_{m}_ci_lower": lo, f"delta_{m}_ci_upper": hi,
                        f"fold_delta_{m}_mean": fw["mean_diff"], f"fold_delta_{m}_p": fw["p_value"],
                        f"fold_delta_{m}_t_ci_lower": fw["ci_lower"], f"fold_delta_{m}_t_ci_upper": fw["ci_upper"]})
        row["better_than_baseline"] = better_than_baseline(
            (row["delta_pr_auc_ci_lower"], row["delta_pr_auc_ci_upper"]),
            (row["delta_roc_auc_ci_lower"], row["delta_roc_auc_ci_upper"]),
            row["delta_pr_auc"], row["delta_roc_auc"], min_effect, min_effect)
        paired.append(row)
    return pd.DataFrame(rows), pd.DataFrame(paired), pd.concat(tk, ignore_index=True), fold_tab, boot
