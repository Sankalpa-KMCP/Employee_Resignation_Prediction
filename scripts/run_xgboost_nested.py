"""Review Member 3's saved XGBoost notebook, or deliberately train a new run.

The notebook and ``configs/models/xgboost.yaml`` own the candidate registry,
preprocessing variants, training protocol, and result export. This launcher
keeps the command-line path aligned with that single implementation.
"""

from pathlib import Path
import argparse

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Review saved XGBoost evidence, or deliberately train a new isolated run.")
    parser.add_argument("--train", action="store_true", help="Repeat the complete experiment in a new reruns directory.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    notebook_path = (
        root / "notebooks" / "team_contributions" / "models"
        / "member_3_xgboost" / "07_m3_xgboost.ipynb"
    )
    notebook = nbformat.read(notebook_path, as_version=4)
    if args.train:
        for cell in notebook.cells:
            if cell.cell_type == "code" and "RUN_XGBOOST_TRAINING = False" in cell.source:
                cell.source = cell.source.replace("RUN_XGBOOST_TRAINING = False", "RUN_XGBOOST_TRAINING = True", 1)
    NotebookClient(
        notebook,
        timeout=None,
        resources={"metadata": {"path": str(root)}},
    ).execute()
    print("New isolated XGBoost experiment completed." if args.train else "Saved XGBoost evidence review completed.")
    print("Canonical group input remains:", root / "results" / "models" / "xgboost" / "oof_predictions.csv")


if __name__ == "__main__":
    main()
