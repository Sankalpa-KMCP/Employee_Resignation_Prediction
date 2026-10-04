"""
Generate CatBoost operational confusion matrices and diagnostic visual comparison.

Demonstrates:
1. Why the naive 0.50 decision cutoff collapses to 0 predicted positives on a calibrated
   imbalanced dataset (prevalence = 10.01%, max predicted probability = 0.1166).
2. The operational confusion matrix at the empirical base-rate threshold (tau = 0.10010).
3. The capacity-constrained operational confusion matrix at Top 10% budget (tau = 0.10226).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "models" / "catboost"

def main():
    oof_path = RESULTS_DIR / "catboost_oof_predictions.csv"
    assert oof_path.exists(), f"Missing OOF predictions at {oof_path}"

    df = pd.read_csv(oof_path)
    y_true = df["y_true"].to_numpy(dtype=int)
    probs = df["y_probability"].to_numpy(dtype=float)
    prevalence = float(np.mean(y_true))

    # Threshold specifications
    thresholds = [
        {
            "name": "Default 0.50 Cutoff",
            "threshold": 0.50,
            "description": "Standard naive classification cutoff (unadjusted for class imbalance)",
            "y_pred": (probs >= 0.50).astype(int),
        },
        {
            "name": "Base-Rate Prevalence",
            "threshold": prevalence,
            "description": f"Flags employees with predicted risk above company prevalence ({prevalence:.4f})",
            "y_pred": (probs >= prevalence).astype(int),
        },
        {
            "name": "Top 10% Budget Capacity",
            "threshold": float(np.quantile(probs, 0.90)),
            "description": "Flags highest-risk 10% of employees (8,000 capacity-constrained cohort)",
            "y_pred": (probs >= np.quantile(probs, 0.90)).astype(int),
        },
        {
            "name": "Top 20% Budget Capacity",
            "threshold": float(np.quantile(probs, 0.80)),
            "description": "Flags highest-risk 20% of employees (16,000 capacity-constrained cohort)",
            "y_pred": (probs >= np.quantile(probs, 0.80)).astype(int),
        },
    ]

    records = []
    cms = []

    for t in thresholds:
        yp = t["y_pred"]
        tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0, 1]).ravel()
        prec = float(precision_score(y_true, yp, zero_division=0))
        rec = float(recall_score(y_true, yp, zero_division=0))
        f1 = float(f1_score(y_true, yp, zero_division=0))
        bal_acc = float(balanced_accuracy_score(y_true, yp))
        flagged = int(np.sum(yp))
        flagged_rate = float(flagged / len(y_true))

        records.append({
            "policy": t["name"],
            "decision_threshold": round(t["threshold"], 5),
            "flagged_count": flagged,
            "flagged_rate": round(flagged_rate, 4),
            "true_negatives_tn": tn,
            "false_positives_fp": fp,
            "false_negatives_fn": fn,
            "true_positives_tp": tp,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "operational_description": t["description"],
        })
        cms.append((t["name"], t["threshold"], np.array([[tn, fp], [fn, tp]]), prec, rec, f1))

    # Save summary table
    summary_df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "catboost_operational_confusion_matrix.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved operational confusion matrix table to: {csv_path}")

    # Generate 3-panel visual plot comparing Default 0.50, Base-Rate, and Top 10%
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=160)
    plt.subplots_adjust(wspace=0.35)

    plot_configs = [
        (cms[0], axes[0], "Reds", "Naive Cutoff (Theoretical Failure)"),
        (cms[1], axes[1], "Blues", "Prevalence Cutoff (Sensitivity Focus)"),
        (cms[2], axes[2], "Greens", "Top 10% Budget (Operational Capacity)"),
    ]

    for (name, thresh, cm, prec, rec, f1), ax, cmap, subtitle in plot_configs:
        cax = ax.imshow(cm, cmap=cmap, aspect="auto")
        ax.set_title(
            f"{name} (Cutoff: {thresh:.4f})\n{subtitle}",
            fontsize=11,
            fontweight="bold",
            pad=12,
        )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Stayed (0)", "Pred Resigned (1)"], fontsize=10)
        ax.set_yticklabels(["Act Stayed (0)", "Act Resigned (1)"], fontsize=10)
        ax.set_xlabel("Predicted Class", fontsize=10, fontweight="semibold")
        ax.set_ylabel("Actual Class", fontsize=10, fontweight="semibold")

        # Overlay text
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = val / np.sum(cm) * 100
                color = "white" if val > cm.max() * 0.55 else "black"
                label = f"{val:,}\n({pct:.1f}%)"
                ax.text(j, i, label, ha="center", va="center", color=color, fontsize=11, fontweight="bold")

        # Metric subtitle box
        metrics_box = (
            f"Recall: {rec:.1%}  |  Precision: {prec:.1%}  |  F1: {f1:.3f}"
        )
        ax.text(
            0.5,
            -0.22,
            metrics_box,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#ccc"),
        )
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "CatBoost Decision Threshold Analysis: Naive 0.50 Cutoff vs. Calibrated Operational Thresholds\n"
        "(Evaluated on 80,000 Out-of-Fold Development Rows, Base Rate = 10.01%)",
        fontsize=13,
        fontweight="bold",
        y=1.05,
    )

    plot_path = RESULTS_DIR / "figure_operational_confusion_matrix.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostic visual comparison to: {plot_path}")

if __name__ == "__main__":
    main()
