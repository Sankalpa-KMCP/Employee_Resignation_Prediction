"""Supplementary MLP for IT25100142: frozen development folds, inner AP selection."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import inspect
import json
import time
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from threadpoolctl import threadpool_limits
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.evaluation.inputs import load_development_data
from src.evaluation.metrics import calculate_binary_metrics
from src.evaluation.oof_validation import validate_oof_predictions

DIRECTORY = Path("results/models/mlp/individual_IT25100142")
CONFIG = Path("configs/models/mlp_IT25100142.yaml")


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def make_preprocessor(features, categorical):
    """Learn scaling and category columns only from the applicable training rows."""
    numeric = [name for name in features if name not in categorical]
    return ColumnTransformer([
        ("numerical", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ], verbose_feature_names_out=False)


def make_model(candidate, config):
    # A stateful RNG keeps shuffling across partial_fit calls reproducible.
    return MLPClassifier(
        hidden_layer_sizes=tuple(candidate["hidden_layers"]),
        activation=config["fixed"]["activation"], solver=config["fixed"]["solver"],
        learning_rate_init=candidate["learning_rate"], alpha=candidate["alpha"],
        batch_size=config["fixed"]["batch_size"], shuffle=True,
        random_state=np.random.RandomState(config["seed"]), early_stopping=False,
        tol=0.0, n_iter_no_change=config["fixed"]["max_epochs"] + 1,
    )


def training_weights(y, weighting):
    """Balanced weights average one and give positives roughly nine times the weight."""
    if weighting == "none":
        return None
    if weighting != "balanced":
        raise ValueError("Unknown weighting policy")
    return compute_sample_weight("balanced", y)


def train_candidate(x_train, y_train, x_valid, y_valid, candidate, config):
    """One partial_fit call is one epoch; only inner AP controls epoch selection."""
    model = make_model(candidate, config)
    weights = training_weights(y_train, candidate["weighting"])
    history = []
    best_ap, best_epoch, stale = -np.inf, 0, 0
    for epoch in range(1, config["fixed"]["max_epochs"] + 1):
        model.partial_fit(x_train, y_train, classes=np.array([0, 1]), sample_weight=weights)
        ap = float(average_precision_score(y_valid, model.predict_proba(x_valid)[:, 1]))
        history.append({"epoch": epoch, "training_loss": float(model.loss_), "inner_ap": ap})
        if ap > best_ap + config["fixed"]["minimum_ap_improvement"]:
            best_ap, best_epoch, stale = ap, epoch, 0
        else:
            stale += 1
        if stale >= config["fixed"]["patience"]:
            break
    return best_ap, best_epoch, history


def refit(x_train, y_train, candidate, config, epochs):
    """Start new weights and train for the selected count without validation feedback."""
    model = make_model(candidate, config)
    weights = training_weights(y_train, candidate["weighting"])
    losses = []
    for _ in range(epochs):
        model.partial_fit(x_train, y_train, classes=np.array([0, 1]), sample_weight=weights)
        losses.append(float(model.loss_))
    return model, losses


def save_figure(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_figures(out, oof, folds, histories, selections):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    prevalence = float(oof.y_true.mean())
    precision, recall, _ = precision_recall_curve(oof.y_true, oof.y_probability)
    fpr, tpr, _ = roc_curve(oof.y_true, oof.y_probability)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(recall, precision, label="MLP OOF")
    axes[0].axhline(prevalence, ls="--", color="gray", label="Prevalence")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision-recall", ylim=(0, 1))
    axes[1].plot(fpr, tpr, label="MLP OOF")
    axes[1].plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    for ax in axes: ax.legend()
    save_figure(fig, out, "ranking_curves.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(oof.y_true, oof.y_probability >= .5,
        display_labels=["Stayed", "Resigned"], ax=ax, colorbar=False)
    ax.set_title("MLP OOF: fixed threshold 0.50")
    save_figure(fig, out, "confusion_matrix.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(folds.fold, folds.average_precision, "o-", label="Outer AP")
    axes[0].axhline(prevalence, ls="--", color="gray", label="Prevalence")
    axes[0].set(xlabel="Outer fold", ylabel="AP", xticks=range(1, 6), title="Fold stability")
    axes[0].legend()
    axes[1].bar(folds.fold, folds.selected_epoch_count, color="#7651a8")
    axes[1].set(xlabel="Outer fold", ylabel="Epochs", xticks=range(1, 6), title="Selected training length")
    save_figure(fig, out, "fold_results.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for winner in folds.itertuples():
        h = pd.DataFrame(histories[f"{winner.fold}/{winner.selected_candidate}"])
        label = f"Fold {winner.fold}: {winner.selected_candidate}"
        axes[0].plot(h.epoch, h.training_loss, label=label)
        axes[1].plot(h.epoch, h.inner_ap, label=label)
        axes[1].scatter([winner.selected_epoch_count], [winner.inner_ap], s=30)
    axes[0].set(xlabel="Epoch", ylabel="Training loss", title="Selected candidates: inner training")
    axes[1].set(xlabel="Epoch", ylabel="Inner validation AP", title="Dots mark selected epochs")
    for ax in axes: ax.legend(fontsize=8)
    save_figure(fig, out, "learning_curves.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, g in oof.groupby("y_true"):
        axes[0].hist(g.y_probability, bins=np.linspace(0, 1, 31), density=True, alpha=.5,
                     label="Resigned" if label else "Stayed")
    axes[0].set(xlabel="Model score", ylabel="Density", title="OOF score distributions")
    axes[0].legend()
    observed, predicted = calibration_curve(oof.y_true, oof.y_probability, n_bins=10, strategy="quantile")
    axes[1].plot(predicted, observed, "o-", label="Quantile bins")
    axes[1].plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    axes[1].set(xlabel="Mean score", ylabel="Observed resignation rate", title="Reliability diagnostic", xlim=(0, 1), ylim=(0, 1))
    axes[1].legend()
    save_figure(fig, out, "probability_diagnostics.png")


def compare_models(root, out, oof, registry, data):
    """Recompute current registered-model metrics; never replace the group registry."""
    analysis = yaml.safe_load((root / "configs/analysis_registry.yaml").read_text(encoding="utf-8"))
    rows = []
    for name, entry in analysis["models"].items():
        frame = pd.read_csv(root / entry["oof_path"])
        validate_oof_predictions(frame, registry, source_data=data,
                                 expected_model_name=entry.get("model_name", name))
        scores = calculate_binary_metrics(frame.y_true, frame.y_probability)
        rows.append({"Model": name, **{k: scores[k] for k in ["average_precision", "roc_auc", "brier_score"]}})
    scores = calculate_binary_metrics(oof.y_true, oof.y_probability)
    rows.append({"Model": "MLP (supplementary)", **{k: scores[k] for k in ["average_precision", "roc_auc", "brier_score"]}})
    p = float(oof.y_true.mean())
    rows.append({"Model": "Constant prevalence reference", "average_precision": p, "roc_auc": .5, "brier_score": p*(1-p)})
    comparison = pd.DataFrame(rows).sort_values("average_precision", ascending=False)
    comparison.to_csv(out / "model_comparison.csv", index=False)
    return comparison


def run(root):
    """Run 40 inner candidates and 5 fresh outer refits, saving each completed stage."""
    root = Path(root).resolve()
    pd.set_option("display.max_columns", None)
    if "sample_weight" not in inspect.signature(MLPClassifier.partial_fit).parameters:
        raise RuntimeError("This experiment needs an MLPClassifier version with partial_fit sample_weight support (scikit-learn >=1.7).")
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    contract = yaml.safe_load((root / "configs/feature_contract.yaml").read_text(encoding="utf-8"))
    features = contract["primary_model_features"]
    categorical = [c["name"] for c in contract["categorical_features"]]
    assert len(features) == 17 and len(categorical) == 4
    assert len(config["candidates"]) == 8 and len({c["candidate_id"] for c in config["candidates"]}) == 8
    data, registry = load_development_data(root)
    before = data.shape
    data = data.merge(registry.loc[registry.partition == "development", ["Employee_ID", "cv_fold"]],
                      on="Employee_ID", validate="one_to_one")
    if data[features + ["Resigned"]].isna().any().any():
        raise ValueError("Missing values differ from the audited dataset; review before training.")
    assert len(data) == 80000 and set(data.cv_fold) == set(range(1, 6))
    print(f"Development shape {before}; after fold alignment {data.shape}; predictors {data[features].shape}. No rows removed.", flush=True)
    out = root / DIRECTORY / (datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "model": "MLP", "student_id": "IT25100142",
                "sklearn_version": sklearn.__version__, "config": config,
                "source_hashes": {str(p): hashlib.sha256((root / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                  for p in [CONFIG, Path("src/models/mlp_experiment.py"), Path("configs/feature_contract.yaml")]},
                "raw_shape": [100000, 20], "development_shape": list(before),
                "predictor_shape": [80000, 17], "prevalence": float(data.Resigned.mean()),
                "formal_holdout_evaluated": False, "thread_limit": 1}
    save_json(out / "manifest.json", manifest)
    selection, folds, predictions, histories, refit_losses = [], [], [], {}, {}
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        for fold in range(1, 6):
            train = data.loc[data.cv_fold != fold]
            valid = data.loc[data.cv_fold == fold]
            inner_train, inner_valid = train_test_split(train, test_size=config["inner_validation_fraction"],
                stratify=train.Resigned, random_state=config["seed"])
            assert (len(train), len(valid), len(inner_train), len(inner_valid)) == (64000, 16000, 54400, 9600)
            assert not set(inner_train.Employee_ID) & set(inner_valid.Employee_ID)
            assert not set(train.Employee_ID) & set(valid.Employee_ID)
            scores = []
            for candidate in config["candidates"]:
                t0 = time.perf_counter()
                prep = make_preprocessor(features, categorical)
                x_train = prep.fit_transform(inner_train[features])
                x_valid = prep.transform(inner_valid[features])
                ap, epochs, history = train_candidate(x_train, inner_train.Resigned.to_numpy(),
                    x_valid, inner_valid.Resigned.to_numpy(), candidate, config)
                row = {"outer_fold": fold, "candidate_id": candidate["candidate_id"], "inner_ap": ap,
                       "selected_epoch_count": epochs, "epochs_run": len(history), "weighting": candidate["weighting"],
                       "fit_time_seconds": time.perf_counter() - t0, "encoded_features": x_train.shape[1],
                       "inner_train_rows": len(inner_train), "inner_validation_rows": len(inner_valid)}
                selection.append(row); scores.append(row)
                histories[f"{fold}/{candidate['candidate_id']}"] = history
                pd.DataFrame(selection).to_csv(out / "candidate_selection.csv", index=False)
                save_json(out / "training_history.json", histories)
                print(f"Fold {fold}, {candidate['candidate_id']}: AP={ap:.6f}, selected epoch={epochs}, ran={len(history)}", flush=True)
            # Stable tie-breaking uses the earlier declared candidate.
            winner = max(scores, key=lambda row: row["inner_ap"])
            candidate = next(c for c in config["candidates"] if c["candidate_id"] == winner["candidate_id"])
            prep = make_preprocessor(features, categorical)
            x_train = prep.fit_transform(train[features]); x_valid = prep.transform(valid[features])
            model, losses = refit(x_train, train.Resigned.to_numpy(), candidate, config, winner["selected_epoch_count"])
            prob = model.predict_proba(x_valid)[:, 1]
            refit_losses[str(fold)] = losses
            metrics = calculate_binary_metrics(valid.Resigned, prob)
            folds.append({"fold": fold, "selected_candidate": candidate["candidate_id"],
                "selected_epoch_count": winner["selected_epoch_count"], "inner_ap": winner["inner_ap"],
                "training_ap": float(average_precision_score(train.Resigned, model.predict_proba(x_train)[:, 1])),
                "outer_train_rows": len(train), "outer_validation_rows": len(valid), **metrics})
            predictions.append(pd.DataFrame({"Employee_ID": valid.Employee_ID.to_numpy(), "cv_fold": fold,
                "y_true": valid.Resigned.to_numpy(), "y_probability": prob, "model_name": "MLP"}))
            pd.DataFrame(folds).to_csv(out / "fold_metrics.csv", index=False)
            pd.concat(predictions).to_csv(out / "partial_oof_predictions.csv", index=False)
            print(f"Fold {fold} winner {candidate['candidate_id']}; OUTER AP={metrics['average_precision']:.6f}", flush=True)
    oof = pd.concat(predictions, ignore_index=True).sort_values("Employee_ID").reset_index(drop=True)
    validate_oof_predictions(oof, registry, source_data=data, expected_model_name="MLP")
    assert len(selection) == 40 and len(folds) == 5
    oof.to_csv(out / "oof_predictions.csv", index=False)
    (out / "partial_oof_predictions.csv").unlink()
    fold_frame = pd.DataFrame(folds)
    summary = {**calculate_binary_metrics(oof.y_true, oof.y_probability),
        "mean_fold_ap": float(fold_frame.average_precision.mean()), "sd_fold_ap": float(fold_frame.average_precision.std()),
        "prevalence": float(oof.y_true.mean()), "development_rows": len(oof), "inner_fits": 40,
        "outer_refits": 5, "total_fits": 45, "training_seconds": time.perf_counter()-started,
        "formal_holdout_evaluated": False, "final_model_frozen": False}
    save_json(out / "summary_metrics.json", summary)
    save_json(out / "refit_training_loss.json", refit_losses)
    generate_figures(out, oof, fold_frame, histories, selection)
    compare_models(root, out, oof, registry, data)
    manifest.update(status="complete", candidate_records=40, winner_records=5,
                    oof_sha256=hashlib.sha256((out / "oof_predictions.csv").read_bytes()).hexdigest())
    save_json(out / "manifest.json", manifest)
    save_json(root / DIRECTORY / "latest_run.json", {"directory": out.relative_to(root).as_posix()})
    print(json.dumps(summary, indent=2), flush=True)
    return out


if __name__ == "__main__":
    run(Path.cwd())
