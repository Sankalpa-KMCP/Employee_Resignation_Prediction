"""Supplementary nested, development-only regularized Logistic Regression."""
from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from src.evaluation.inputs import load_development_data
from src.evaluation.metrics import calculate_binary_metrics
from src.evaluation.oof_validation import validate_oof_predictions
from src.prioritization.ranking import evaluate_ranking

MODEL_NAME = "Regularized Logistic Regression"
DIRECTORY = Path("results/models/logistic_regression/individual_IT25100429")
PROXY_FEATURES = ["Performance_Score", "Monthly_Salary", "Years_At_Company"]
CANDIDATES = [
    {"candidate_id": f"L{i + 1:02d}", "C": c, "class_weight": weight}
    for i, (c, weight) in enumerate(
        (c, weight) for c in (0.001, 0.01, 0.1, 1.0, 10.0) for weight in (None, "balanced")
    )
]


def build_pipeline(features, categorical, c, weight):
    """Fit scaling and reference-category encoding only within each training set."""
    numeric = [name for name in features if name not in categorical]
    preprocess = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(drop="first", handle_unknown="ignore",
                                      sparse_output=False), categorical),
    ], verbose_feature_names_out=False)
    # lbfgs uses L2 regularization by default; smaller C means stronger shrinkage.
    return Pipeline([("preprocess", preprocess), ("logistic", LogisticRegression(
        C=c, class_weight=weight, solver="lbfgs", max_iter=2000, random_state=42))])


def fit_checked(model, x, y):
    """Stop rather than silently accepting an unconverged demonstration model."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y)
    return model


def fold_value_proxy(train, valid):
    """Equal-weight proxy with min/max fitted on outer training predictors only."""
    scaler = MinMaxScaler(clip=True).fit(train[PROXY_FEATURES])
    return scaler.transform(valid[PROXY_FEATURES]).mean(axis=1)


def run(root):
    root = Path(root)
    pd.set_option("display.max_columns", None)
    contract = yaml.safe_load((root / "configs/feature_contract.yaml").read_text(encoding="utf-8"))
    features = contract["primary_model_features"]
    categorical = [item["name"] for item in contract["categorical_features"]]
    data, registry = load_development_data(root, usecols=features)
    before = data.shape
    data = data.merge(registry.loc[registry.partition == "development", ["Employee_ID", "cv_fold"]],
                      on="Employee_ID", validate="one_to_one")
    print(f"Shape before/after fold join: {before} -> {data.shape}; no rows dropped.", flush=True)
    print(f"Target distribution: {data.Resigned.value_counts().sort_index().to_dict()}; "
          "stratified inner split; none/balanced weights tuned; no resampling.", flush=True)
    out = root / DIRECTORY
    out.mkdir(parents=True, exist_ok=True)
    frames, fold_rows, evidence, candidate_rows, coefficient_rows, ranking_rows = [], [], [], [], [], []
    for fold in range(1, 6):
        train = data.loc[data.cv_fold != fold]
        valid = data.loc[data.cv_fold == fold]
        inner_train, inner_valid = train_test_split(
            train, test_size=0.15, stratify=train.Resigned, random_state=42)
        scores = []
        for candidate in CANDIDATES:
            model = build_pipeline(features, categorical, candidate["C"], candidate["class_weight"])
            fit_checked(model, inner_train[features], inner_train.Resigned)
            p = model.predict_proba(inner_valid[features])[:, 1]
            scores.append({**candidate, "inner_log_loss": float(log_loss(inner_valid.Resigned, p)),
                           **{f"inner_{k}": v for k, v in calculate_binary_metrics(inner_valid.Resigned, p).items()}})
        winner = min(scores, key=lambda item: item["inner_log_loss"])
        candidate_rows.extend({"outer_fold": fold, **score} for score in scores)
        evidence.append({"outer_fold": fold, "inner_train_rows": len(inner_train),
                         "inner_validation_rows": len(inner_valid), "outer_train_rows": len(train),
                         "outer_validation_rows": len(valid), "winner": winner["candidate_id"],
                         "selection_metric": "minimum inner log loss", "candidates": scores})
        model = build_pipeline(features, categorical, winner["C"], winner["class_weight"])
        fit_checked(model, train[features], train.Resigned)
        p = model.predict_proba(valid[features])[:, 1]
        metrics = calculate_binary_metrics(valid.Resigned, p)
        baseline = np.full(len(valid), train.Resigned.mean())
        fold_rows.append({"fold": fold, "candidate_id": winner["candidate_id"], "C": winner["C"],
                          "class_weight": winner["class_weight"] or "none", **metrics,
                          "log_loss": float(log_loss(valid.Resigned, p)),
                          "baseline_log_loss": float(log_loss(valid.Resigned, baseline)),
                          "baseline_brier": float(np.mean((valid.Resigned.to_numpy() - baseline) ** 2))})
        frame = pd.DataFrame({"Employee_ID": valid.Employee_ID.to_numpy(), "cv_fold": fold,
                              "y_true": valid.Resigned.to_numpy(), "y_probability": p,
                              "y_pred_0_5": (p >= 0.5).astype(int), "model_name": MODEL_NAME,
                              "retention_value_proxy": fold_value_proxy(train, valid)})
        frame["combined_score"] = frame.y_probability * frame.retention_value_proxy
        frames.append(frame)
        for capacity in (0.1, 0.2, 0.3):
            for score in ("y_probability", "retention_value_proxy", "combined_score"):
                ranking_rows.append({"fold": fold, "strategy": score, **evaluate_ranking(
                    frame, score, capacity, value_col="retention_value_proxy")})
        names = model.named_steps["preprocess"].get_feature_names_out()
        coefficient_rows.extend({"fold": fold, "feature": name, "coefficient": float(coef)}
                                for name, coef in zip(names, model.named_steps["logistic"].coef_[0]))
        print(f"Completed fold {fold}: {winner['candidate_id']}; log loss={fold_rows[-1]['log_loss']:.6f}", flush=True)
    oof = pd.concat(frames, ignore_index=True).sort_values("Employee_ID")
    validate_oof_predictions(oof, registry, source_data=data, expected_model_name=MODEL_NAME)
    folds = pd.DataFrame(fold_rows)
    summary = calculate_binary_metrics(oof.y_true, oof.y_probability)
    summary.update(log_loss=float(log_loss(oof.y_true, oof.y_probability)),
                   mean_fold_ap=float(folds.average_precision.mean()), sd_fold_ap=float(folds.average_precision.std()),
                   mean_baseline_log_loss=float(folds.baseline_log_loss.mean()),
                   mean_baseline_brier=float(folds.baseline_brier.mean()),
                   prevalence=float(data.Resigned.mean()), development_rows=len(data),
                   formal_holdout_accessed=False, final_model_frozen=False, inner_fits=50, outer_refits=5,
                   selection_metric="minimum inner log loss", sklearn_version=sklearn.__version__,
                   pandas_version=pd.__version__, numpy_version=np.__version__,
                   matplotlib_version=matplotlib.__version__,
                   calibration="raw selected-model probabilities; reliability assessed on outer OOF; no recalibration")
    # Record-level artifacts stay local and are explicitly ignored by version control.
    oof.to_csv(out / "oof_predictions.csv", index=False)
    folds.to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(out / "inner_candidate_metrics.csv", index=False)
    coefficients = pd.DataFrame(coefficient_rows)
    coefficients.to_csv(out / "coefficients.csv", index=False)
    rankings = pd.DataFrame(ranking_rows)
    rankings.to_csv(out / "ranking_fold_metrics.csv", index=False)
    (out / "selection_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    (out / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    observed, predicted = calibration_curve(oof.y_true, oof.y_probability, n_bins=10, strategy="quantile")
    pd.DataFrame({"mean_predicted_probability": predicted, "observed_resignation_rate": observed}).to_csv(
        out / "calibration_bins.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axes[0].plot(predicted, observed, "o-", label="Nested OOF (quantile bins)")
    axes[0].set(xlabel="Mean predicted probability", ylabel="Observed resignation rate", title="Calibration (full probability scale)")
    axes[0].legend()
    axes[1].hist(oof.y_probability, bins=30)
    axes[1].axvline(summary["prevalence"], color="gray", linestyle="--")
    axes[1].set(xlabel="Predicted probability", ylabel="Count", title="OOF risk distribution")
    fig.tight_layout(); fig.savefig(out / "calibration.png", dpi=150); plt.close(fig)
    precision, recall, _ = precision_recall_curve(oof.y_true, oof.y_probability)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(recall, precision, label=f"OOF AP={summary['average_precision']:.4f}")
    ax.axhline(summary["prevalence"], color="gray", linestyle="--", label="Prevalence baseline")
    ax.set(xlabel="Recall", ylabel="Precision", title="Regularized Logistic Regression", ylim=(0, 1))
    ax.legend(); fig.tight_layout(); fig.savefig(out / "precision_recall_curve.png", dpi=150); plt.close(fig)
    illustration = coefficients.loc[coefficients.fold == 1].set_index("feature").coefficient
    illustration = illustration.loc[illustration.abs().nlargest(12).index].sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    illustration.plot.barh(ax=ax)
    ax.set(xlabel="Log-odds coefficient", title="Fold 1 selected model: illustrative associations")
    fig.tight_layout(); fig.savefig(out / "coefficients.png", dpi=150); plt.close(fig)
    ranking_summary = rankings.groupby(["capacity", "strategy"]).mean(numeric_only=True).drop(columns="fold")
    ranking_summary.to_csv(out / "ranking_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    for strategy, group in rankings.groupby("strategy"):
        grouped = group.groupby("capacity").lift_at_capacity.agg(["mean", "std"])
        ax.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"], marker="o", capsize=4, label=strategy)
    ax.axhline(1, color="gray", linestyle="--", label="Random expected lift")
    ax.set(xlabel="Review capacity", ylabel="Mean fold lift ± SD", title="Academic ranking demonstration")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "ranking.png", dpi=150); plt.close(fig)
    report = f"""# IT25100429 — supplementary Regularized Logistic Regression

Standalone L2 Logistic Regression without PCA; EBM remains the assigned model.
Uses the approved predictors and frozen five-fold development registry. All
scaling and encoding are training-local. Ten C/weight candidates per outer fold
are selected by minimum log loss on a stratified 15% inner validation split,
then refitted on the entire outer training partition (50 inner fits + 5 refits).
No rows are dropped or resampled; class imbalance is addressed through stratified
splitting, none/balanced weighting candidates and prevalence-based comparisons.

## Observed results

Development rows: {len(data):,}; prevalence: {summary['prevalence']:.6f}.
OOF AP: {summary['average_precision']:.6f}; mean fold AP:
{summary['mean_fold_ap']:.6f} ± {summary['sd_fold_ap']:.6f} (descriptive SD).
ROC-AUC: {summary['roc_auc']:.6f} (chance 0.5).
Log loss: {summary['log_loss']:.6f}; fold-training prevalence baseline:
{summary['mean_baseline_log_loss']:.6f}.
Brier: {summary['brier_score']:.6f}; prevalence baseline:
{summary['mean_baseline_brier']:.6f}.

```text
{folds[['fold', 'candidate_id', 'C', 'class_weight', 'average_precision', 'log_loss']].to_string(index=False)}
```

Calibration is evaluated with OOF reliability bins; probabilities are not
post-hoc recalibrated. Low probability error can reflect learning prevalence,
without useful ranking. Small score differences do not establish operational
usefulness. Coefficients are regularized conditional associations, not causes.

The current result {'does not improve on' if summary['log_loss'] >= summary['mean_baseline_log_loss'] else 'numerically improves on'}
the prevalence baseline for log loss. Check AP, ROC-AUC and capacity lift before
interpreting the probabilities as a useful ranking. At threshold 0.50, recall is
{summary['recall']:.6f}; the threshold is fixed and not tuned on outer labels.

## Academic prioritization

Combined score = OOF probability × equal-weight normalized performance/salary/
tenure proxy, with fold-training min/max bounds and clipping. Selection is within
each outer validation fold. See ranking_summary.csv for 10%, 20%, 30% capacity.
Random expected lift is 1; the line is not an empirical allocation experiment.
The score is neither employee worth nor intervention benefit. No model is frozen
for deployment, and the formal holdout is excluded. Local record-level OOF is
ignored by Git. The shared PCA benchmark and group registry are unchanged.

## Reproduction

Run `python scripts/run_logistic_regression_experiment.py`, then Restart and Run All
in `11_IT25100429_logistic_regression_extra.ipynb` (saved review by default).
Runtime versions: scikit-learn {sklearn.__version__}, pandas {pd.__version__},
NumPy {np.__version__}, Matplotlib {matplotlib.__version__}.
"""
    (root / "reports/IT25100429_Logistic_Regression_Experiment.md").write_text(report, encoding="utf-8")
    return summary
