"""Supplementary Decision Tree experiment for IT25100429; development only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve, ConfusionMatrixDisplay
from src.evaluation.inputs import load_development_data
from src.evaluation.metrics import calculate_binary_metrics
from src.evaluation.oof_validation import validate_oof_predictions

CANDIDATES = [("D01",3,None),("D02",3,"balanced"),("D03",6,None),
              ("D04",6,"balanced"),("D05",12,"balanced"),("D06",None,"balanced")]

def build_pipeline(features, categorical, depth, weight, scaled=False):
    numeric = [c for c in features if c not in categorical]
    prep = ColumnTransformer([
        ("numeric", StandardScaler() if scaled else "passthrough", numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ], verbose_feature_names_out=False)
    return Pipeline([("preprocess",prep),("tree",DecisionTreeClassifier(
        max_depth=depth, min_samples_leaf=50, class_weight=weight, random_state=42))])

def render_simple_tree(model, names, path):
    """Show every split and leaf; hide statistical labels for a readable demo."""
    fig,ax=plt.subplots(figsize=(18,7))
    annotations=plot_tree(model.named_steps["tree"],
        feature_names=[str(n).replace("_"," ") for n in names],
        class_names=["Stays","Resigned"],filled=True,rounded=True,
        impurity=False,precision=2,fontsize=12,ax=ax)
    for annotation in annotations:
        lines=annotation.get_text().splitlines()
        lines=[line.replace("class = ","Predict: ") for line in lines
               if not line.startswith(("samples =", "value ="))]
        annotation.set_text("\n".join(lines))
    ax.set_title("Decision Tree: follow True left or False right to a prediction",fontsize=16,pad=12)
    fig.text(.5,.02,"Depth-3 D02 illustration · fold 1 training only · colours show weighted class tendency",ha="center",fontsize=11)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(path,dpi=160,bbox_inches="tight");plt.close(fig)


def run(root):
    root=Path(root)
    pd.set_option("display.max_columns",None)
    data,registry=load_development_data(root)
    contract=yaml.safe_load((root/"configs/feature_contract.yaml").read_text())
    features=contract["primary_model_features"]
    categorical=[c["name"] for c in contract["categorical_features"]]
    data=data.merge(registry.loc[registry.partition=="development",["Employee_ID","cv_fold"]],on="Employee_ID",validate="one_to_one")
    out=root/"results/models/decision_tree/individual_IT25100429"
    out.mkdir(parents=True,exist_ok=True)
    selected,rows,evidence,sensitivity=[],[],[],[]
    for fold in range(1,6):
        train=data.loc[data.cv_fold!=fold];valid=data.loc[data.cv_fold==fold]
        inner_train,inner_valid=train_test_split(train,test_size=.15,stratify=train.Resigned,random_state=42)
        candidates=[]
        for cid,depth,weight in CANDIDATES:
            m=build_pipeline(features,categorical,depth,weight)
            m.fit(inner_train[features],inner_train.Resigned)
            candidates.append(dict(candidate_id=cid,inner_ap=float(average_precision_score(inner_valid.Resigned,m.predict_proba(inner_valid[features])[:,1]))))
        winner=max(candidates,key=lambda c:c["inner_ap"])["candidate_id"]
        evidence.append(dict(outer_fold=fold,inner_train_rows=len(inner_train),inner_validation_rows=len(inner_valid),outer_train_rows=len(train),outer_validation_rows=len(valid),winner=winner,candidates=candidates))
        for cid,depth,weight in CANDIDATES:
            m=build_pipeline(features,categorical,depth,weight)
            m.fit(train[features],train.Resigned)
            prob=m.predict_proba(valid[features])[:,1]
            metrics=calculate_binary_metrics(valid.Resigned,prob)
            rows.append(dict(candidate_id=cid,fold=fold,depth="unlimited" if depth is None else depth,class_weight=weight or "none",train_ap=average_precision_score(train.Resigned,m.predict_proba(train[features])[:,1]),**metrics))
            frame=pd.DataFrame(dict(Employee_ID=valid.Employee_ID.to_numpy(),cv_fold=fold,y_true=valid.Resigned.to_numpy(),y_probability=prob,model_name="Decision Tree",y_pred_0_5=(prob>=.5).astype(int)))
            if cid==winner:selected.append(frame)
            if cid=="D02":
                scaled=build_pipeline(features,categorical,depth,weight,scaled=True)
                scaled.fit(train[features],train.Resigned);sp=scaled.predict_proba(valid[features])[:,1]
                sensitivity.append(dict(fold=fold,raw_ap=metrics["average_precision"],standardized_ap=average_precision_score(valid.Resigned,sp),max_absolute_probability_difference=float(np.max(np.abs(prob-sp))),identical_predictions=bool(np.array_equal(prob,sp))))
            if fold==1 and cid=="D02":demo=m
        print(f"Completed fold {fold}; inner-selected winner {winner}",flush=True)
    oof=pd.concat(selected,ignore_index=True).sort_values("Employee_ID")
    validate_oof_predictions(oof,registry,source_data=data,expected_model_name="Decision Tree")
    folds=pd.DataFrame(rows)
    comparison=folds.groupby("candidate_id",sort=False).agg(mean_fold_ap=("average_precision","mean"),sd_fold_ap=("average_precision","std"),mean_train_ap=("train_ap","mean"),mean_roc_auc=("roc_auc","mean"),mean_brier=("brier_score","mean"),mean_recall=("recall","mean"),mean_f1=("f1","mean")).reset_index()
    summary=calculate_binary_metrics(oof.y_true,oof.y_probability)
    aps=[calculate_binary_metrics(g.y_true,g.y_probability)["average_precision"] for _,g in oof.groupby("cv_fold")]
    summary.update(mean_fold_ap=float(np.mean(aps)),sd_fold_ap=float(np.std(aps,ddof=1)),development_rows=len(oof),formal_holdout_accessed=False,primary_protocol="six-candidate inner AP selection; full outer-training refit",inner_fits=30,outer_variety_refits=30,standardization_sensitivity_fits=5,final_model_frozen=False)
    folds.to_csv(out/"variety_fold_metrics.csv",index=False);comparison.to_csv(out/"variety_comparison.csv",index=False)
    oof.to_csv(out/"oof_predictions.csv",index=False)
    pd.DataFrame(sensitivity).to_csv(out/"standardization_sensitivity.csv",index=False)
    (out/"selection_evidence.json").write_text(json.dumps(evidence,indent=2))
    (out/"summary_metrics.json").write_text(json.dumps(summary,indent=2))
    names=demo.named_steps["preprocess"].get_feature_names_out()
    render_simple_tree(demo, names, out/"tree_diagram.png")
    importance=pd.Series(demo.named_steps["tree"].feature_importances_,index=names).sort_values(ascending=False)
    importance.rename("importance").to_csv(out/"feature_importance.csv")
    fig,ax=plt.subplots(figsize=(9,5));importance.head(12).sort_values().plot.barh(ax=ax);ax.set_title("Illustrative D02 impurity importance (model behaviour, not causality)");fig.tight_layout();fig.savefig(out/"feature_importance.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));ax.plot(comparison.candidate_id,comparison.mean_train_ap,"o-",label="Training AP");ax.errorbar(comparison.candidate_id,comparison.mean_fold_ap,yerr=comparison.sd_fold_ap,fmt="o-",capsize=4,label="Outer validation AP ± fold SD");ax.axhline(data.Resigned.mean(),ls="--",color="gray",label="Prevalence baseline");ax.set_ylabel("Average Precision");ax.set_title("Tree complexity and class weighting: training vs validation");ax.legend();fig.tight_layout();fig.savefig(out/"overfitting_comparison.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5));ConfusionMatrixDisplay.from_predictions(oof.y_true,oof.y_pred_0_5,display_labels=["Stays","Resigned"],ax=ax,colorbar=False);ax.set_title("Nested OOF predictions at threshold 0.50");fig.tight_layout();fig.savefig(out/"confusion_matrix.png",dpi=150);plt.close(fig)
    precision,recall,_=precision_recall_curve(oof.y_true,oof.y_probability)
    fig,ax=plt.subplots(figsize=(7,5));ax.plot(recall,precision,label=f"Nested OOF AP={summary['average_precision']:.4f}");ax.axhline(data.Resigned.mean(),ls="--",color="gray",label="Prevalence");ax.set(xlabel="Recall",ylabel="Precision",title="Decision Tree precision–recall curve",ylim=(0,1));ax.legend();fig.tight_layout();fig.savefig(out/"precision_recall_curve.png",dpi=150);plt.close(fig)
    table=comparison.to_string(index=False,float_format=lambda v:f"{v:.6f}")
    report=f"""# IT25100429 — supplementary Decision Tree experiment

EBM remains the assigned individual model; this Decision Tree is supplementary and does not replace one of the group's six registered inputs.

## Implementation and validation
Scikit-learn DecisionTreeClassifier, ColumnTransformer, OneHotEncoder and Pipeline. Categorical encoding is fitted inside training. Six candidates pair depth 3/6 with none/balanced weighting, plus depth 12/unlimited balanced trees. Every tree uses min_samples_leaf=50 and random_state=42. A bounded manual grid evaluates all six on a stratified 15% inner validation split inside each frozen outer training fold. Inner AP selects the winner, which is freshly fitted on the full 64,000 outer-training rows. Fixed-candidate outer scores are descriptive variety comparisons, not the selection rule. All 80,000 development employees receive exactly one selected-model OOF prediction. The 20,000-row holdout is excluded before reading its labels or predictors.

30 inner fits + 30 fixed-candidate outer refits (including selected winners) + 5 standardized D02 sensitivity fits = 65 total fits. The demonstration tree reuses fold 1 D02, without an additional all-data fit.

## Results
Nested mean fold AP: {summary['mean_fold_ap']:.6f} ± {summary['sd_fold_ap']:.6f}; pooled AP: {summary['average_precision']:.6f}; pooled ROC-AUC: {summary['roc_auc']:.6f}; Brier: {summary['brier_score']:.6f}; recall at .50: {summary['recall']:.6f}; F1 at .50: {summary['f1']:.6f}. Prevalence/no-skill AP is {data.Resigned.mean():.5f}. Fold SD is descriptive, not a confidence interval.

```text
{table}
```

## Interpretation and limitations
AP is appropriate for the 10.01% minority class; ROC-AUC describes ranking, Brier probability error, and recall/F1 describe detection at the fixed .50 threshold. Accuracy alone is misleading. Interpret scores against chance AP .1001 and ROC .5; small numerical differences do not establish practical usefulness. Greater training performance without validation improvement illustrates overfitting. Balanced class weights can shift probability scales and calibration. No threshold was tuned on outer folds.

Tree splits are interpretable but can be unstable. Impurity importance is model-specific, can favour high-cardinality features, and is not causal. The visualization is one shallow fold-local illustration, not a frozen final selected model. Repeated seeds/inner validation, pruning, better predictors and probability calibration are possible future improvements. Raw vs standardized D02 scores are in standardization_sensitivity.csv; scaling generally preserves tree split ordering, with possible floating-point differences.

## Demonstration
Open the notebook, Run All in saved-review mode, display the shallow tree, follow a root-to-leaf path, compare train/validation AP, then explain the confusion matrix and PR curve. Saved files reside in results/models/decision_tree/individual_IT25100429/. No final holdout evaluation or deployment model is claimed.
"""
    (root/"reports/IT25100429_Decision_Tree_Experiment.md").write_text(report)
    return summary
