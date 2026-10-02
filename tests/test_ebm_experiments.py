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


import pytest
from pathlib import Path
import shutil
from src.models.ebm_experiments import load_formal_ebm_input, write_formal_manifest, text_sha256

@pytest.fixture
def formal_checkout(tmp_path):
    root = Path(__file__).resolve().parents[1]
    paths = ["configs/models/ebm.yaml", "data/processed/split_registry.csv",
             "results/models/ebm/individual_IT25100429/formal_input_manifest.json",
             "results/models/ebm/individual_IT25100429/oof_predictions.csv",
             "results/models/ebm/individual_IT25100429/fold_metrics.csv",
             "results/models/ebm/individual_IT25100429/summary_metrics.json"]
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    return tmp_path


def to_crlf(root):
    for path in root.rglob("*"):
        if path.is_file():
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))


def test_loader_accepts_windows_line_endings_and_manifest_writer_is_portable(formal_checkout):
    root = formal_checkout
    import json
    before = load_formal_ebm_input(root)
    original_manifest = json.loads((root / "results/models/ebm/individual_IT25100429/formal_input_manifest.json").read_text())
    to_crlf(root)
    after = load_formal_ebm_input(root)
    pd.testing.assert_frame_equal(before[0], after[0])
    pd.testing.assert_frame_equal(before[1], after[1])
    assert before[2] == after[2]
    write_formal_manifest(root)
    generated = json.loads((root / "results/models/ebm/individual_IT25100429/formal_input_manifest.json").read_text())
    assert generated == original_manifest
    # A manifest produced from CRLF also verifies an LF checkout.
    for path in root.rglob("*"):
        if path.is_file():path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    load_formal_ebm_input(root)


@pytest.mark.parametrize("relative,old,new,error", [
    ("configs/models/ebm.yaml", b"n_jobs: 2", b"n_jobs: 3", "config changed"),
    ("results/models/ebm/individual_IT25100429/summary_metrics.json", b"0.099000", b"0.199000", "checksum mismatch"),
])
def test_crlf_loader_still_rejects_content_changes(formal_checkout,relative,old,new,error):
    to_crlf(formal_checkout)
    path = formal_checkout / relative
    contents = path.read_bytes()
    assert old in contents
    path.write_bytes(contents.replace(old,new,1))
    with pytest.raises(ValueError, match=error):load_formal_ebm_input(formal_checkout)


def test_normalization_preserves_whitespace_and_lone_cr(tmp_path):
    p = tmp_path / "file.txt"
    p.write_bytes(b"value\r\n")
    canonical = text_sha256(p)
    p.write_bytes(b"value\n")
    assert text_sha256(p) == canonical
    for changed in (b"value \n", b"value\r"):
        p.write_bytes(changed)
        assert text_sha256(p) != canonical
