"""Run supplementary development-only Decision Tree experiment."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.models.decision_tree_experiment import run
if __name__ == "__main__":
    run(ROOT)
