"""Run the canonical six-variety EBM experiment from the authoritative YAML.

Primary nested selection uses three raw candidates. Their standardized pairs are
sensitivity experiments. This supersedes the earlier unexecuted eight-candidate
script; its source remains available in Git history.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    from src.models.ebm_experiments import run_experiments
    _, comparison, _, metrics, output_dir = run_experiments(ROOT)
    print(comparison.to_string(index=False))
    print(metrics)
    print(f"Canonical EBM results: {output_dir}")

if __name__ == "__main__":
    main()
