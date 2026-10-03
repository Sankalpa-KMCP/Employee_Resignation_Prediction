"""Review saved EBM evidence or deliberately repeat its configured six-variety experiment.

Primary nested selection uses three raw candidates. Their standardized pairs are
sensitivity experiments. This supersedes the earlier unexecuted eight-candidate
script; its source remains available in Git history.
"""
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description="Review saved EBM evidence, or train a new isolated six-variety run.")
    parser.add_argument("--train", action="store_true", help="Perform 30 inner fits and 30 outer refits in a new reruns directory.")
    args = parser.parse_args()
    from src.models.ebm_experiments import CONFIG, run_experiments
    if args.train:
        _, comparison, _, metrics, output_dir = run_experiments(ROOT)
        print(comparison.to_string(index=False))
    else:
        from src.evaluation.inputs import load_saved_model_evidence
        output_dir = ROOT / CONFIG["canonical_results"]["directory"]
        _, _, metrics = load_saved_model_evidence(ROOT, "EBM", output_dir)
    print(metrics)
    print(f"{'New isolated EBM run' if args.train else 'Verified saved EBM evidence'}: {output_dir}")

if __name__ == "__main__":
    main()
