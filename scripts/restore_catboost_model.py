"""Rebuild the git-ignored CatBoost development model (`models/catboost/model.cbm`).

    python scripts/restore_catboost_model.py            # only writes missing files
    python scripts/restore_catboost_model.py --force    # overwrite existing files
    python scripts/restore_catboost_model.py --out DIR  # write elsewhere (e.g. for verification)

`models/*/*` is git-ignored, so a fresh clone has no serialized model even though the tracked
evidence (`results/models/catboost/improvement/final/`) references it. This refits the promoted
v2.1.0 configuration (candidate R09, unweighted, 28 trees, seed 42) on the 80,000 development
rows only; the formal holdout is never loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from catboost import CatBoostClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.audit_stats import load_audit_inputs  # noqa: E402

CANDIDATE_ID = "R09"
TREE_COUNT = 28  # selected_tree_count in the audited final run (final/fold_selection_summary.csv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "models" / "catboost")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args()

    model_path, meta_path = args.out / "model.cbm", args.out / "metadata.json"
    if not args.force and model_path.exists() and meta_path.exists():
        print(f"Nothing to do: {model_path} and {meta_path.name} already exist (use --force to rebuild).")
        return

    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "models" / "catboost.yaml").read_text(encoding="utf-8"))
    candidate = next(c for c in cfg["candidates"] if c["id"] == CANDIDATE_ID)
    params = {**cfg["fixed_params"], **candidate["params"], "iterations": TREE_COUNT}
    params.pop("early_stopping_rounds", None)

    data, _, features, categorical = load_audit_inputs(PROJECT_ROOT)
    X = data[features].copy()
    for col in categorical:
        X[col] = X[col].astype(str)
    print(f"Refitting {CANDIDATE_ID} with {TREE_COUNT} trees on development rows: {X.shape}")

    model = CatBoostClassifier(**params, cat_features=categorical)
    model.fit(X, data["Resigned"])
    assert model.tree_count_ == TREE_COUNT, model.tree_count_

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    meta = {
        "name": cfg["model"]["name"], "model_id": cfg["model"]["id"], "candidate_id": CANDIDATE_ID,
        "fitted": True, "selected_tree_count": TREE_COUNT, "fitted_on_all_outer_rows": True,
        "configured_max_iterations": candidate["params"]["iterations"], "params": params,
        "categorical_features": categorical, "feature_names": features, "model_file": "model.cbm",
        "config_version": cfg["model"]["config_version"], "restored_by": "scripts/restore_catboost_model.py",
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {model_path} and {meta_path}")


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    main()
