from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score
from src.models.decision_tree_experiment import build_pipeline
from src.evaluation.inputs import load_development_data
from src.evaluation.oof_validation import validate_oof_predictions

ROOT=Path(__file__).resolve().parents[1]

def test_fold_local_scaling_and_unknown_categories():
    train=pd.DataFrame({"number":[0.,2.,4.,6.],"category":["a","a","b","b"]})
    test=pd.DataFrame({"number":[1000.],"category":["unseen"]})
    m=build_pipeline(list(train.columns),["category"],3,None,scaled=True)
    m.fit(train,[0,1,0,1])
    scaler=m.named_steps["preprocess"].named_transformers_["numeric"]
    assert scaler.mean_[0]==3
    assert np.isfinite(m.predict_proba(test)).all()
    assert scaler.mean_[0]==3

def test_saved_nested_evidence_and_holdout_exclusion():
    out=ROOT/"results/models/decision_tree/individual_IT25100429"
    q=pd.read_csv(out/"oof_predictions.csv")
    data,registry=load_development_data(ROOT,usecols=[])
    validate_oof_predictions(q,registry,source_data=data,expected_model_name="Decision Tree")
    evidence=json.loads((out/"selection_evidence.json").read_text())
    assert len(evidence)==5
    for e in evidence:
        assert e["winner"]==max(e["candidates"],key=lambda c:c["inner_ap"])["candidate_id"]
        assert e["inner_train_rows"]==54400 and e["inner_validation_rows"]==9600
        assert e["outer_train_rows"]==64000 and e["outer_validation_rows"]==16000
    summary=json.loads((out/"summary_metrics.json").read_text())
    assert np.isclose(average_precision_score(q.y_true,q.y_probability),summary["average_precision"])
    assert summary["formal_holdout_accessed"] is False
    group=yaml.safe_load((ROOT/"configs/analysis_registry.yaml").read_text())
    assert len(group["models"])==6 and "Decision Tree" not in group["models"]
