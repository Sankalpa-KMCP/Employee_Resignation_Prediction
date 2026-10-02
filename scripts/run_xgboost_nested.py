"""Execute Member 3's canonical XGBoost notebook from the repository root.

The notebook and ``configs/models/xgboost.yaml`` own the candidate registry,
preprocessing variants, training protocol, and result export. This launcher
keeps the command-line path aligned with that single implementation.
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook_path = (
        root / "notebooks" / "team_contributions" / "models"
        / "member_3_xgboost" / "07_m3_xgboost.ipynb"
    )
    notebook = nbformat.read(notebook_path, as_version=4)
    NotebookClient(
        notebook,
        timeout=None,
        resources={"metadata": {"path": str(root)}},
    ).execute()
    print("Canonical Member 3 XGBoost notebook completed.")
    print("Primary OOF:", root / "results" / "models" / "xgboost" / "oof_predictions.csv")


if __name__ == "__main__":
    main()
