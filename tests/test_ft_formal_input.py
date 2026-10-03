from pathlib import Path
import pytest
from src.models.ft_transformer_execution import digest,load_formal_ft_input


def test_ft_hash_ignores_only_crlf(tmp_path):
    path=tmp_path / "artifact.csv"
    path.write_bytes(b"a,b\n1,2\n")
    expected=digest(path)
    path.write_bytes(b"a,b\r\n1,2\r\n")
    assert digest(path)==expected
    path.write_bytes(b"a,b\r\n1,3\r\n")
    assert digest(path)!=expected


def test_formal_ft_loader_does_not_fall_back_to_legacy(tmp_path):
    old=tmp_path / "results/models/ft_transformer/weighted_early_stopping_final"
    old.mkdir(parents=True)
    (old / "ft_transformer_final_oof_predictions.csv").write_text("legacy")
    with pytest.raises(FileNotFoundError):load_formal_ft_input(tmp_path)
