"""IT25100429: bounded EBM experiments with nested selection and scaling sensitivity."""
from pathlib import Path
import json, time, hashlib, platform
import numpy as np
import pandas as pd
import sklearn, interpret
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from interpret.glassbox import ExplainableBoostingClassifier
from src.evaluation.metrics import calculate_binary_metrics
from src.evaluation.oof_validation import validate_oof_predictions

SETTINGS = [
    dict(setting="E01", learning_rate=0.015, max_leaves=2, min_samples_leaf=10, interactions=0),
    dict(setting="E02", learning_rate=0.03, max_leaves=2, min_samples_leaf=4, interactions=5),
    dict(setting="E03", learning_rate=0.005, max_leaves=3, min_samples_leaf=2, interactions=10),
]
CANDIDATES = [dict(s, preprocessing=mode, candidate_id=s["setting"]+"_"+mode)
              for s in SETTINGS for mode in ("raw", "standardized")]

def fit_candidate(X, y, candidate, categorical):
    X = X.copy()
    numeric = [c for c in X if c not in categorical]
    scaler = None
    if candidate["preprocessing"] == "standardized":
        scaler = StandardScaler().fit(X[numeric])
        X[numeric] = scaler.transform(X[numeric])
    params = {k: candidate[k] for k in ("learning_rate", "max_leaves", "min_samples_leaf", "interactions")}
    model = ExplainableBoostingClassifier(
        feature_names=list(X), feature_types=["nominal" if c in categorical else "continuous" for c in X],
        objective="log_loss", random_state=42, n_jobs=2, max_bins=256, max_interaction_bins=32,
        max_rounds=5000, early_stopping_rounds=100, validation_size=0.15, outer_bags=8, **params)
    model.fit(X, y)
    return model, scaler, numeric

def predict(fitted, X):
    model, scaler, numeric = fitted
    X = X.copy()
    if scaler is not None:
        X[numeric] = scaler.transform(X[numeric])
    return model.predict_proba(X)[:, list(model.classes_).index(1)]

def choose(records):
    # Predeclared practical equivalence margin: 0.001 absolute AP.
    best = max(r["inner_ap"] for r in records)
    eligible = [r for r in records if r["inner_ap"] >= best - 0.001]
    return min(eligible, key=lambda r: (r["interactions"], r["max_leaves"], r["candidate_id"]))

def run_experiments(root):
    root = Path(root)
    out = root / "results/models/ebm/individual_IT25100429"
    out.mkdir(parents=True, exist_ok=True)
    import yaml
    contract = yaml.safe_load((root / "configs/feature_contract.yaml").read_text())
    features = contract["primary_model_features"]
    categorical = [f["name"] for f in contract["categorical_features"]]
    registry = pd.read_csv(root / "data/processed/split_registry.csv")
    path = root / "data/raw/Extended_Employee_Performance_and_Productivity_Data.csv"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == contract["dataset"]["sha256"]
    assert hashlib.sha256((root / "data/processed/split_registry.csv").read_bytes()).hexdigest() == contract["split_registry"]["sha256"]
    dev_registry = registry.loc[registry.partition == "development"].copy()
    holdout_ids = set(registry.loc[registry.partition == "holdout", "Employee_ID"])
    # Inspect identifiers first; skip holdout rows when parsing predictors and target.
    identifiers = pd.read_csv(path, usecols=["Employee_ID"])
    dev_ids = set(dev_registry.Employee_ID)
    skip = [i+1 for i, value in enumerate(identifiers.Employee_ID) if value not in dev_ids]
    data = pd.read_csv(path, skiprows=skip).merge(dev_registry, on="Employee_ID", validate="one_to_one")
    data.Resigned = data.Resigned.astype(int)
    assert len(data) == 80000 and set(data.Employee_ID).isdisjoint(holdout_ids)
    print(f"Rows before/after development isolation: {len(identifiers)} -> {len(data)}; prevalence={data.Resigned.mean():.5f}", flush=True)
    inner_records, fold_records, nested_records, predictions = [], [], [], []
    for fold in range(1, 6):
        train = data.loc[data.cv_fold != fold]
        valid = data.loc[data.cv_fold == fold]
        inner_train, inner_valid = train_test_split(train, test_size=0.15, random_state=42, stratify=train.Resigned)
        decisions = []
        for candidate in CANDIDATES:
            started = time.perf_counter()
            fitted = fit_candidate(inner_train[features], inner_train.Resigned, candidate, categorical)
            ap = average_precision_score(inner_valid.Resigned, predict(fitted, inner_valid[features]))
            rec = dict(candidate, outer_fold=fold, inner_ap=float(ap), inner_train_rows=len(inner_train), inner_validation_rows=len(inner_valid))
            decisions.append(rec); inner_records.append(rec)
            # Fixed-variety outer evaluation is separate from nested selection.
            fitted = fit_candidate(train[features], train.Resigned, candidate, categorical)
            prob = predict(fitted, valid[features])
            fold_records.append(dict(candidate, fold=fold, fit_seconds=time.perf_counter()-started,
                                     **calculate_binary_metrics(valid.Resigned, prob)))
            predictions.append(pd.DataFrame(dict(Employee_ID=valid.Employee_ID.to_numpy(), cv_fold=fold,
                y_true=valid.Resigned.to_numpy(), y_probability=prob, model_name=candidate["candidate_id"])))
            print(f"Fold {fold} {candidate['candidate_id']}: inner AP={ap:.6f}", flush=True)
        # Primary contract uses raw numeric values; scaling is sensitivity only.
        winner = choose([r for r in decisions if r["preprocessing"] == "raw"])
        selected = next(c for c in CANDIDATES if c["candidate_id"] == winner["candidate_id"])
        # Prediction from that fixed candidate's refit: no outer metrics used in selection.
        chosen = next(q for q in predictions if q.cv_fold.iloc[0] == fold and q.model_name.iloc[0] == winner["candidate_id"]).copy()
        chosen["model_name"] = "EBM"
        chosen["selected_candidate"] = winner["candidate_id"]
        nested_records.append(chosen)
    all_predictions = pd.concat(predictions, ignore_index=True)
    nested = pd.concat(nested_records, ignore_index=True)
    validate_oof_predictions(nested, registry, source_data=data, expected_model_name="EBM")
    summaries = []
    for candidate in CANDIDATES:
        q = all_predictions.loc[all_predictions.model_name == candidate["candidate_id"]]
        validate_oof_predictions(q, registry, source_data=data, expected_model_name=candidate["candidate_id"])
        fm = pd.DataFrame(fold_records).loc[lambda d: d.candidate_id == candidate["candidate_id"]]
        summaries.append(dict(candidate, **calculate_binary_metrics(q.y_true, q.y_probability),
                             cv_mean_ap=float(fm.average_precision.mean()), cv_std_ap=float(fm.average_precision.std())))
    comparison = pd.DataFrame(summaries)
    nested_metrics = calculate_binary_metrics(nested.y_true, nested.y_probability)
    nested_fold = pd.DataFrame([dict(fold=f, **calculate_binary_metrics(q.y_true, q.y_probability)) for f,q in nested.groupby("cv_fold")])
    nested_metrics.update(cv_mean_ap=float(nested_fold.average_precision.mean()), cv_std_ap=float(nested_fold.average_precision.std()))
    for name, frame in [("candidate_comparison",comparison),("inner_selection",pd.DataFrame(inner_records)),
                        ("candidate_fold_metrics",pd.DataFrame(fold_records)),("candidate_oof_predictions",all_predictions),
                        ("oof_predictions",nested),("fold_metrics",nested_fold)]:
        frame.to_csv(out / (name+".csv"), index=False)
    metadata = dict(student_id="IT25100429", varieties=6, inner_candidate_fits=30, outer_candidate_refits=30,
                    formal_holdout_used=False, holdout_overlap=0, seed=42, practical_ap_margin=0.001,
                    python=platform.python_version(), sklearn=sklearn.__version__, interpret=interpret.__version__,
                    candidate_settings=CANDIDATES, primary_preprocessing="raw", final_model_frozen=False)
    (out / "summary_metrics.json").write_text(json.dumps(nested_metrics, indent=2))
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    return data, comparison, nested, nested_metrics, out

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    _, comparison, _, metrics, output_dir = run_experiments(root)
    print(comparison.to_string(index=False))
    print(json.dumps(metrics, indent=2))
    print(f"Results: {output_dir}")
