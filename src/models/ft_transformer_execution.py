"""Formal FT-Transformer nested evaluation; no formal-holdout fitting or scoring."""
from pathlib import Path
import hashlib, json, time, platform
import numpy as np
import pandas as pd
import yaml
from src.models.config import load_model_config
from src.evaluation.metrics import calculate_binary_metrics
from src.evaluation.oof_validation import validate_oof_predictions


def digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _resolve_output_directory(root, output_dir=None):
    """Keep explicit experiments inside the FT rerun boundary."""
    root = Path(root).resolve()
    if output_dir is None:
        return root / "results/models/ft_transformer/strict_nested"
    output = Path(output_dir)
    output = (output if output.is_absolute() else root / output).resolve()
    boundary = (root / "results/models/ft_transformer/reruns").resolve()
    if output == boundary or not output.is_relative_to(boundary):
        raise ValueError("Explicit FT output must be a directory beneath results/models/ft_transformer/reruns.")
    return output


def run(root, output_dir=None):
    """Execute/resume nested CV; explicit outputs must be isolated reruns.

    Existing callers retain the canonical destination. Notebook retraining passes
    a fresh directory from create_run_directory, preserving canonical evidence.
    """
    root = Path(root).resolve()
    out = _resolve_output_directory(root, output_dir)
    import torch
    import src.models.ft_transformer as ft
    out.mkdir(parents=True, exist_ok=True)
    config = load_model_config(root / "configs/models/ft_transformer.yaml", strict_schema=True)
    contract = yaml.safe_load((root / "configs/feature_contract.yaml").read_text())
    features = contract["primary_model_features"]
    categorical = [c["name"] for c in contract["categorical_features"]]
    registry = pd.read_csv(root / "data/processed/split_registry.csv")
    raw = root / contract["dataset"]["path"]
    assert digest(raw) == contract["dataset"]["sha256"]
    assert digest(root / "data/processed/split_registry.csv") == contract["split_registry"]["sha256"]
    devreg = registry.loc[registry.partition == "development"]
    dev_ids = set(devreg.Employee_ID)
    ids = pd.read_csv(raw, usecols=["Employee_ID"])
    skip = [i+1 for i,value in enumerate(ids.Employee_ID) if value not in dev_ids]
    data = pd.read_csv(raw, skiprows=skip).merge(devreg, on="Employee_ID", validate="one_to_one")
    assert len(data) == 80000 and set(data.Employee_ID) == dev_ids
    print(f"Rows before/after development isolation: {len(ids)} -> {len(data)}; prevalence={data.Resigned.mean():.5f}", flush=True)
    signature = dict(config=digest(root / "configs/models/ft_transformer.yaml"),
                     adapter=digest(root / "src/models/ft_transformer.py"),
                     registry=digest(root / "data/processed/split_registry.csv"), dataset=digest(raw),
                     torch=torch.__version__, python=platform.python_version(),
                     device=str(ft._select_device(config["reproducibility"].get("device", "auto"))))
    signature_path = out / "execution_signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError("Existing FT checkpoints belong to different inputs or runtime; use a separate run directory.")
    signature_path.write_text(json.dumps(signature, indent=2))
    torch.set_num_threads(2)
    predictions, fold_metrics, evidence = [], [], []
    original = ft._evaluate_candidate
    for fold in range(1,6):
        checkpoint = out / f"fold_{fold:02}_predictions.csv"
        evpath = out / f"fold_{fold:02}_evidence.json"
        metpath = out / f"fold_{fold:02}_metrics.json"
        if checkpoint.exists() and evpath.exists() and metpath.exists():
            predictions.append(pd.read_csv(checkpoint)); evidence.append(json.loads(evpath.read_text()));fold_metrics.append(json.loads(metpath.read_text()))
            print(f"Resumed completed fold {fold}", flush=True);continue
        train = data.loc[data.cv_fold != fold]
        valid = data.loc[data.cv_fold == fold]
        def logged_evaluation(**kwargs):
            candidate_id = kwargs["candidate_dict"]["id"]
            print(f"Fold {fold}: starting {candidate_id}", flush=True)
            result = original(**kwargs)
            print(f"Fold {fold}: {candidate_id} inner AP={result['inner_ap']:.6f}; selected epochs={result['selected_epoch_count']}; seconds={result['fit_time_seconds']:.1f}", flush=True)
            (out / "progress.json").write_text(json.dumps(dict(fold=fold, candidate=candidate_id, result=result), indent=2))
            return result
        ft._evaluate_candidate = logged_evaluation
        try:
            adapter = ft.FTTransformerAdapter(config=config, candidate_id="auto", categorical_features=categorical)
            started = time.perf_counter(); adapter.fit(train[features], train.Resigned.astype(int)); fit_seconds = time.perf_counter()-started
        finally:
            ft._evaluate_candidate = original
        ev = adapter.get_selection_evidence()
        assert ev["fitted_on_all_outer_rows"] and len(ev["all_candidates"]) == 8
        started = time.perf_counter(); prob = adapter.predict_proba(valid[features]); predict_seconds = time.perf_counter()-started
        frame = pd.DataFrame(dict(Employee_ID=valid.Employee_ID.to_numpy(), cv_fold=fold,
                                  y_true=valid.Resigned.astype(int).to_numpy(), model_name="FT-Transformer", y_probability=prob,
                                  y_pred_0_5=(prob>=.5).astype(int)))
        metrics = dict(fold=fold, fit_time_seconds=fit_seconds, predict_time_seconds=predict_seconds,
                       selected_candidate=ev["winning_candidate_id"], selected_epoch_count=ev["selected_epoch_count"],
                       **calculate_binary_metrics(frame.y_true, frame.y_probability))
        ev["outer_fold"] = fold
        frame.to_csv(checkpoint,index=False);evpath.write_text(json.dumps(ev,indent=2));metpath.write_text(json.dumps(metrics,indent=2))
        predictions.append(frame);fold_metrics.append(metrics);evidence.append(ev)
        print(f"Completed outer fold {fold}: AP={metrics['average_precision']:.6f}",flush=True)
        del adapter
    oof = pd.concat(predictions,ignore_index=True).sort_values("Employee_ID").reset_index(drop=True)
    validate_oof_predictions(oof,registry,source_data=data,expected_model_name="FT-Transformer")
    folds = pd.DataFrame(fold_metrics)
    summary = calculate_binary_metrics(oof.y_true,oof.y_probability)
    summary.update(model_name="FT-Transformer", cv_mean_ap=float(folds.average_precision.mean()),cv_std_ap=float(folds.average_precision.std()),
                   cv_mean_roc_auc=float(folds.roc_auc.mean()), total_fit_time_seconds=float(folds.fit_time_seconds.sum()),
                   total_predict_time_seconds=float(folds.predict_time_seconds.sum()), development_rows=80000,
                   outer_folds=5, candidate_evaluations=40, formal_holdout_used=False, final_model_frozen=False,
                   protocol="strict_nested_eight_candidate_selection")
    oof.to_csv(out / "oof_predictions.csv",index=False);folds.to_csv(out / "fold_metrics.csv",index=False)
    (out / "summary_metrics.json").write_text(json.dumps(summary,indent=2))
    (out / "selection_evidence.json").write_text(json.dumps(evidence,indent=2))
    pd.DataFrame([dict(outer_fold=ev["outer_fold"],candidate_id=cid,inner_ap=r["inner_ap"],selected_epoch_count=r["selected_epoch_count"],weighting=r["weighting"],fit_time_seconds=r["fit_time_seconds"]) for ev in evidence for cid,r in ev["all_candidates"].items()]).to_csv(out / "candidate_selection.csv",index=False)
    manifest = dict(signature=signature, files={name:digest(out/name) for name in ["oof_predictions.csv","fold_metrics.csv","summary_metrics.json","selection_evidence.json","candidate_selection.csv"]})
    (out / "formal_input_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return oof, folds, summary


def load_formal_ft_input(root):
    root=Path(root);out=root / "results/models/ft_transformer/strict_nested"
    manifest=json.loads((out / "formal_input_manifest.json").read_text())
    for key,relative in [("config","configs/models/ft_transformer.yaml"),("adapter",manifest.get("adapter_source", "src/models/ft_transformer.py")),("registry","data/processed/split_registry.csv")]:
        if digest(root/relative) != manifest["signature"][key]:raise ValueError("FT formal input provenance mismatch.")
    for name,expected in manifest["files"].items():
        if digest(out/name)!=expected:raise ValueError("FT formal input checksum mismatch.")
    q=pd.read_csv(out / "oof_predictions.csv")
    validate_oof_predictions(q,pd.read_csv(root / "data/processed/split_registry.csv"),expected_model_name="FT-Transformer")
    return q,pd.read_csv(out / "fold_metrics.csv"),json.loads((out / "summary_metrics.json").read_text())
