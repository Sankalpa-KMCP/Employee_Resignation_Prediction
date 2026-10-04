"""Execute the supplementary Logistic Regression study (no formal holdout)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.logistic_regression_experiment import run

if __name__ == "__main__":
    run(ROOT)
