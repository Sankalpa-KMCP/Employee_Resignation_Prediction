import numpy as np
import pandas as pd
from src.models.ebm_experiments import CANDIDATES, choose, fit_candidate, predict

def test_equivalence_prefers_simpler_candidate():
    rows=[dict(candidate_id="complex",inner_ap=.101,interactions=10,max_leaves=3),
          dict(candidate_id="simple",inner_ap=.1005,interactions=0,max_leaves=2),
          dict(candidate_id="poor",inner_ap=.095,interactions=0,max_leaves=2)]
    assert choose(rows)["candidate_id"] == "simple"

def test_scaler_is_training_only_and_unseen_category_predicts():
    X=pd.DataFrame({"number":np.arange(100,dtype=float),"category":["a","b"]*50})
    y=np.array([0,1]*50)
    candidate=dict(CANDIDATES[1])
    fitted=fit_candidate(X,y,candidate,["category"])
    assert fitted[1].mean_[0] == 49.5
    probabilities=predict(fitted,pd.DataFrame({"number":[999.],"category":["unseen"]}))
    assert fitted[1].mean_[0] == 49.5
    assert probabilities.shape == (1,) and np.isfinite(probabilities).all()
    assert 0 <= probabilities[0] <= 1
