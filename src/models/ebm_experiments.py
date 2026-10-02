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

from src.models.config import load_model_config

CONFIG = load_model_config(Path(__file__).resolve().parents[2] / "configs/models/ebm.yaml", strict_schema=True)
CANDIDATES = [dict(c["params"], setting=c["setting"], preprocessing=c["preprocessing"], candidate_id=c["id"])
              for c in CONFIG["candidates"]]

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
        **CONFIG["fixed_params"], **params)
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
    eligible = [r for r in records if r["inner_ap"] >= best - CONFIG["inner_validation"]["practical_ap_margin"]]
    return min(eligible, key=lambda r: (r["interactions"], r["max_leaves"], r["candidate_id"]))

def run_experiments(root):
    root = Path(root)
    out = root / CONFIG["canonical_results"]["directory"]
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
        inner_train, inner_valid = train_test_split(train, test_size=CONFIG["inner_validation"]["fraction"], random_state=CONFIG["reproducibility"]["random_seed"], stratify=train.Resigned)
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
                    formal_holdout_used=False, holdout_overlap=0, seed=CONFIG["reproducibility"]["random_seed"], practical_ap_margin=CONFIG["inner_validation"]["practical_ap_margin"],
                    python=platform.python_version(), sklearn=sklearn.__version__, interpret=interpret.__version__,
                    candidate_settings=CANDIDATES, fixed_parameters=CONFIG["fixed_params"], primary_preprocessing="raw", final_model_frozen=False)
    (out / "summary_metrics.json").write_text(json.dumps(nested_metrics, indent=2))
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    write_formal_manifest(root)
    return data, comparison, nested, nested_metrics, out

def text_sha256(path):
    """Hash text bytes after CRLF-to-LF normalization, preserving all other bytes.

    Git may check text out with CRLF on Windows. Only that representation change
    is ignored: spaces, numbers, encoding and lone CR bytes remain significant.
    """
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_formal_manifest(root):
    root = Path(root)
    config = load_model_config(root / "configs/models/ebm.yaml", strict_schema=True)
    files = {name: {"path": config["canonical_results"][name],
                   "sha256": text_sha256(root / config["canonical_results"][name])}
             for name in ("oof_predictions", "fold_metrics", "summary_metrics")}
    manifest = dict(model_name="EBM", protocol="five_fold_nested_development_raw_candidates",
                    hash_normalization="crlf_to_lf", config_sha256=text_sha256(root / "configs/models/ebm.yaml"),
                    rows=80000, formal_holdout_used=False, files=files)
    (root / config["canonical_results"]["manifest"]).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_formal_ebm_input(root):
    """Load only manifest-bound formal EBM results; never fall back to historical files."""
    root = Path(root)
    config = load_model_config(root / "configs/models/ebm.yaml", strict_schema=True)
    manifest = json.loads((root / config["canonical_results"]["manifest"]).read_text())
    # Legacy manifests stored hashes of LF files; their values already match
    # this canonical representation. Reject any unsupported hash convention.
    if manifest.get("hash_normalization", "crlf_to_lf") != "crlf_to_lf":
        raise ValueError("Unsupported EBM manifest hash normalization.")
    if manifest["config_sha256"] != text_sha256(root / "configs/models/ebm.yaml"):
        raise ValueError("EBM config changed since formal artifact reconciliation; verify or rerun experiments.")
    for name, entry in manifest["files"].items():
        if entry["path"] != config["canonical_results"][name]:
            raise ValueError("EBM manifest does not point to the configured canonical input.")
        if text_sha256(root / entry["path"]) != entry["sha256"]:
            raise ValueError("EBM formal artifact checksum mismatch.")
    predictions = pd.read_csv(root / config["canonical_results"]["oof_predictions"])
    registry = pd.read_csv(root / "data/processed/split_registry.csv")
    validate_oof_predictions(predictions, registry, expected_model_name="EBM")
    return predictions, pd.read_csv(root / config["canonical_results"]["fold_metrics"]), json.loads(
        (root / config["canonical_results"]["summary_metrics"]).read_text())

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    _, comparison, _, metrics, output_dir = run_experiments(root)
    print(comparison.to_string(index=False))
    print(json.dumps(metrics, indent=2))
    print(f"Results: {output_dir}")
