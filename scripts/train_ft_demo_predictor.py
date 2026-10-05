"""Train a single FT-Transformer (FT01) on development rows for the notebook's interactive demo.

This is NOT the deployment model. It uses only the 80,000 development rows (the formal
holdout is never read), one fixed candidate instead of the nested search, and exists only
so the notebook can score a hand-entered employee. Run from anywhere:

    python scripts/train_ft_demo_predictor.py
"""
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.inputs import load_development_data
from src.models.config import load_model_config
from src.models.ft_transformer import FTTransformerAdapter

CANDIDATE_ID = "FT01"
MAX_EPOCHS = 40  # bounded for a CPU demo run; early stopping (patience 15) usually ends sooner
OUTPUT_DIR = ROOT / "results/models/ft_transformer/demo_predictor"


def main():
    config = load_model_config(ROOT / "configs/models/ft_transformer.yaml", strict_schema=True)
    contract = yaml.safe_load((ROOT / "configs/feature_contract.yaml").read_text(encoding="utf-8"))
    features = contract["primary_model_features"]
    categorical = [item["name"] for item in contract["categorical_features"]]

    development, _ = load_development_data(ROOT)
    print(f"Development rows used: {len(development):,} (holdout not read); "
          f"Resigned prevalence={development.Resigned.mean():.5f}", flush=True)

    adapter = FTTransformerAdapter(config=config, candidate_id=CANDIDATE_ID,
                                   categorical_features=categorical)
    started = time.perf_counter()
    adapter.fit(development[features], development.Resigned.astype(int), epochs=MAX_EPOCHS)
    seconds = time.perf_counter() - started

    adapter.save(OUTPUT_DIR)
    (OUTPUT_DIR / "demo_metadata.json").write_text(json.dumps({
        "purpose": "interactive notebook demo only",
        "candidate_id": CANDIDATE_ID,
        "rows_used": len(development),
        "selected_epoch_count": adapter.selected_epoch_count_,
        "max_epochs_cap": MAX_EPOCHS,
        "fit_seconds": round(seconds, 1),
        "holdout_used": False,
        "deployment_model": False,
    }, indent=2), encoding="utf-8")
    print(f"Saved demo predictor to {OUTPUT_DIR.relative_to(ROOT)} in {seconds:.0f}s "
          f"(epochs={adapter.selected_epoch_count_})")


if __name__ == "__main__":
    main()
