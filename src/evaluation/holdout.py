"""One-time assessment of a trusted, development-fitted frozen pipeline.

No training or selection takes place here. The freeze manifest must be created
by the group before enabling evaluation; the default readiness path reads no data.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import yaml

from src.evaluation.calibration import reliability_table, expected_calibration_error
from src.evaluation.metrics import calculate_binary_metrics


FREEZE_PATH = "models/frozen/freeze.json"
RESULT_PATH = "results/final_holdout"


def _assessment_exists(root):
    """An empty reserved output directory or .gitkeep does not mean evaluation."""
    out = Path(root) / RESULT_PATH
    return out.exists() and (not out.is_dir() or any(path.name != ".gitkeep" for path in out.iterdir()))


def _digest(path, normalized=False):
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if normalized else data).hexdigest()


def _local_path(root, relative):
    """Keep manifest references inside the checkout; reject absolute paths."""
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Artifact paths must be repository-relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact paths must stay inside the project")
    return path


def holdout_readiness(root):
    """Check freeze metadata and artifact hashes without reading employee data.

    Ready means recorded prerequisites are satisfied, not permission to evaluate.
    Full-development training provenance is a group attestation, not reconstructed
    from serialized weights. No fitted artifact is deserialized in this function.
    """
    root = Path(root).resolve()
    checks = []

    def check(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    documents = {}
    for name, relative in [
        ("comparison", "results/comparison/comparison_summary.json"),
        ("calibration", "results/calibration/calibration_decision.json"),
        ("fairness", "results/fairness/fairness_summary.json"),
        ("prioritization", "results/prioritization/prioritization_decision.json"),
    ]:
        path = root / relative
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError("Expected a JSON object")
            documents[name] = document
            check(f"{name} decision", True, relative)
        except (OSError, ValueError):
            check(f"{name} decision", False, f"Missing or invalid: {relative}")
    check("final model frozen", documents.get("comparison", {}).get("final_model_frozen") is True,
          "A numerical OOF leader alone is not a final freeze.")
    check("no previous assessment", not _assessment_exists(root),
          "Any receipt or result blocks reruns; an empty reserved directory or .gitkeep is allowed.")

    manifest = None
    try:
        manifest = json.loads((root / FREEZE_PATH).read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Expected a JSON object")
        contract = yaml.safe_load((root / "configs/feature_contract.yaml").read_text(encoding="utf-8"))
        comparison = documents.get("comparison", {})
        calibration = documents.get("calibration", {})
        check("freeze declaration", manifest.get("pipeline_frozen") is True and bool(manifest.get("freeze_id")),
              "Freeze identifier and affirmative declaration are required.")
        check("training partition", manifest.get("training_partition") == "development"
              and manifest.get("training_rows") == contract["split_registry"]["development_rows"],
              "The entire development partition must be the final fit input.")
        check("feature contract", manifest.get("feature_columns") == contract["primary_model_features"],
              "Exact approved predictor order; target and ID excluded.")
        check("frozen model matches decision", isinstance(manifest.get("model"), str)
              and bool(manifest["model"].strip()) and manifest.get("model") == comparison.get("final_model"),
              "Comparison must name final_model explicitly.")
        threshold = manifest.get("threshold")
        check("threshold fixed", isinstance(threshold, (float, int)) and not isinstance(threshold, bool)
              and np.isfinite(threshold) and 0 <= threshold <= 1 and bool(manifest.get("threshold_source")),
              "Threshold and its development-only rationale must be recorded.")
        check("probability source frozen", calibration.get("probability_source_frozen") is True
              and manifest.get("probability_source") == calibration.get("probability_source")
              and manifest.get("probability_source") in {"raw", "sigmoid", "isotonic"},
              "Explicit raw or development-fitted sigmoid/isotonic source.")
        check("pipeline includes calibration", manifest.get("includes_frozen_probability_source") is True,
              "pipeline.joblib must include the selected preprocessing and probability transformation.")
        check("downstream policy frozen", manifest.get("downstream_policy_frozen") is True
              and documents.get("prioritization", {}).get("policy_frozen") is True,
              "Proxy, capacity and strategy must be decided before access, including an explicit omission.")
        check("prediction-only assessment scope", manifest.get("holdout_scope") == "prediction_only"
              and documents.get("prioritization", {}).get("holdout_scope") == "prediction_only",
              "This evaluator requires an explicit prediction-only scope; downstream corroboration needs a separately frozen evaluator.")
        check("historical exposure disclosed", manifest.get("historical_audit_exposure_disclosed") is True,
              "The earlier preliminary audit prevents a guaranteed-pristine-holdout claim.")
        expected_paths = {
            "configs/feature_contract.yaml", "configs/analysis_registry.yaml",
            "results/comparison/comparison_summary.json", "results/calibration/calibration_decision.json",
            "results/fairness/fairness_summary.json", "results/prioritization/prioritization_decision.json",
        }
        artifacts = manifest.get("artifacts", {})
        pipeline_path = manifest.get("pipeline_path")
        if not isinstance(pipeline_path, str):
            raise ValueError("pipeline_path is required")
        resolved_pipeline = _local_path(root, pipeline_path)
        if (resolved_pipeline.suffix != ".joblib"
                or not resolved_pipeline.is_relative_to((root / "models/frozen").resolve())):
            raise ValueError("Pipeline must be a joblib artifact under models/frozen")
        expected_paths.add(pipeline_path)
        exact_artifacts = isinstance(artifacts, dict) and set(artifacts) == expected_paths
        check("all freeze inputs bound", exact_artifacts,
              "Manifest must bind the pipeline, contracts and four decisions by SHA-256.")
        if not exact_artifacts:
            # Do not let a malformed manifest make safe readiness hash raw data.
            raise ValueError("Artifact inventory must contain exactly the declared freeze inputs")
        for relative, digest in artifacts.items():
            path = _local_path(root, relative)
            check(f"artifact: {relative}", path.is_file() and _digest(path) == digest,
                  "Exact-byte SHA-256; copied or changed artifacts require a new development freeze.")
        check("no prior holdout use declared", manifest.get("formal_holdout_used_for_selection") is False,
              "Recorded group attestation; not inferred from model scores.")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        check("valid freeze manifest", False, f"Missing or invalid: {FREEZE_PATH}")
    return pd.DataFrame(checks), manifest


def positive_probability(pipeline, X):
    """Support project adapters (1D) and classifiers (2D with classes_)."""
    scores = np.asarray(pipeline.predict_proba(X), dtype=float)
    if scores.ndim == 2:
        classes = np.asarray(getattr(pipeline, "classes_", []))
        if scores.shape[1] != 2 or set(classes.tolist()) != {0, 1}:
            raise ValueError("Two-column output requires explicit binary classes_")
        scores = scores[:, int(np.flatnonzero(classes == 1)[0])]
    if scores.ndim != 1 or len(scores) != len(X):
        raise ValueError("Pipeline must return one positive-class probability per row")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    return scores


def run_final_holdout(root, *, authorized=False):
    """Evaluate once, only after explicit activation and a verified freeze.

    The joblib file must be a trusted local complete pipeline or project adapter.
    It exposes predict_proba(DataFrame), never fit. Exports contain aggregates only.
    A durable started marker remains after errors to prevent silent repeat access.
    """
    if authorized is not True:
        raise RuntimeError("Holdout evaluation is disabled; explicit activation is required")
    root = Path(root).resolve()
    checks, manifest = holdout_readiness(root)
    if not checks.passed.all():
        raise RuntimeError("Holdout blocked: " + "; ".join(checks.loc[~checks.passed, "check"]))
    contract = yaml.safe_load((root / "configs/feature_contract.yaml").read_text(encoding="utf-8"))
    # Verify frozen data without parsing labels, before recording assessment start.
    for key in ("dataset", "split_registry"):
        if _digest(root / contract[key]["path"], normalized=True) != contract[key]["sha256"]:
            raise ValueError(f"Frozen {key} checksum mismatch")
    split = pd.read_csv(root / contract["split_registry"]["path"])
    if split.Employee_ID.isna().any() or not split.Employee_ID.is_unique:
        raise ValueError("Split IDs must be present and unique")
    holdout_ids = set(split.loc[split.partition == "holdout", "Employee_ID"])
    if len(holdout_ids) != contract["split_registry"]["formal_holdout_rows"]:
        raise ValueError("Frozen holdout row count mismatch")
    import joblib
    pipeline = joblib.load(_local_path(root, manifest["pipeline_path"]))
    if not callable(getattr(pipeline, "predict_proba", None)):
        raise ValueError("Frozen artifact lacks predict_proba")
    out = root / RESULT_PATH
    out.mkdir(parents=True, exist_ok=True)
    receipt = {"status": "started", "freeze_id": manifest["freeze_id"],
               "freeze_sha256": _digest(root / FREEZE_PATH)}
    # Exclusive creation is the atomic claim: a second process cannot evaluate again.
    with (out / "assessment_receipt.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2))
    source_path = root / contract["dataset"]["path"]
    identifiers = pd.read_csv(source_path, usecols=["Employee_ID"])
    skip = [i + 1 for i, value in enumerate(identifiers.Employee_ID) if value not in holdout_ids]
    data = pd.read_csv(source_path, skiprows=skip,
                       usecols=["Employee_ID", "Resigned", *manifest["feature_columns"]])
    if not data.Employee_ID.is_unique or set(data.Employee_ID) != holdout_ids:
        raise ValueError("Source holdout IDs do not match the frozen registry")
    if not data.Resigned.isin([0, 1]).all():
        raise ValueError("Holdout labels must be binary")
    print(f"Rows before/after holdout isolation: {len(identifiers):,} -> {len(data):,}; no assigned rows dropped.")
    # Never call fit, choose thresholds, select calibration or inspect alternative models.
    y = data.Resigned.to_numpy(dtype=int)
    p = positive_probability(pipeline, data[manifest["feature_columns"]])
    threshold = float(manifest["threshold"])
    metrics = calculate_binary_metrics(y, p, threshold=threshold)
    metrics.update({"ece_10_bins": expected_calibration_error(y, p),
                    "prevalence": float(y.mean()), "threshold": threshold, "rows": len(y)})
    table = reliability_table(y, p)
    # Conditional on this fixed prediction set; resamples do not represent retraining uncertainty.
    rng = np.random.default_rng(42)
    classes = [np.flatnonzero(y == value) for value in (0, 1)]
    bootstrap = []
    for _ in range(2000):
        index = np.concatenate([rng.choice(group, len(group), replace=True) for group in classes])
        row = calculate_binary_metrics(y[index], p[index], threshold=threshold)
        bootstrap.append({key: row[key] for key in ["average_precision", "roc_auc", "brier_score", "precision", "recall", "f1", "balanced_accuracy"]})
    bootstrap = pd.DataFrame(bootstrap)
    intervals = pd.DataFrame({"metric": bootstrap.columns,
                              "lower_95": bootstrap.quantile(.025).values,
                              "upper_95": bootstrap.quantile(.975).values})
    metrics.update({"model": manifest["model"], "probability_source": manifest["probability_source"],
                    "formal_holdout_used_for_selection": False,
                    "historical_audit_exposure_disclosed": True,
                    "downstream_scope": "Prediction corroboration; downstream policies require their separately frozen evaluators."})
    (out / "holdout_summary.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    table.to_csv(out / "holdout_reliability.csv", index=False)
    intervals.to_csv(out / "holdout_bootstrap_intervals.csv", index=False)
    from sklearn.metrics import precision_recall_curve, roc_curve
    precision, recall, _ = precision_recall_curve(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    pd.DataFrame({"precision": precision, "recall": recall}).to_csv(out / "holdout_pr_curve.csv", index=False)
    pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr}).to_csv(out / "holdout_roc_curve.csv", index=False)
    receipt["status"] = "completed"
    (out / "assessment_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return metrics, table, intervals
