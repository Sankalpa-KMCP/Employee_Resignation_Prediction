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


def test_config_matches_executed_candidates_and_fixed_parameters():
    import json
    from pathlib import Path
    from src.models.ebm_experiments import CONFIG
    root = Path(__file__).resolve().parents[1]
    metadata = json.loads((root / "results/models/ebm/individual_IT25100429/run_metadata.json").read_text())
    assert CANDIDATES == metadata["candidate_settings"]
    assert CONFIG["fixed_params"] == metadata["fixed_parameters"]
    assert len([c for c in CONFIG["candidates"] if c["role"] == "primary"]) == 3


def test_formal_loader_has_no_historical_fallback(tmp_path):
    import shutil
    from pathlib import Path
    import pytest
    from src.models.ebm_experiments import load_formal_ebm_input
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "configs/models").mkdir(parents=True)
    shutil.copy(root / "configs/models/ebm.yaml", tmp_path / "configs/models/ebm.yaml")
    historical = tmp_path / "results/models/ebm"
    historical.mkdir(parents=True)
    (historical / "oof_predictions.csv").write_text("historical result")
    with pytest.raises(FileNotFoundError):
        load_formal_ebm_input(tmp_path)
