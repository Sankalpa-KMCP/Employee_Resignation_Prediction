"""Load and query the canonical feature contract."""
from __future__ import annotations
from pathlib import Path
import yaml

DEFAULT_CONTRACT = Path(__file__).resolve().parents[2] / "configs" / "feature_contract.yaml"

def load_feature_contract(path: str | Path = DEFAULT_CONTRACT) -> dict:
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(p)
    data=yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data,dict): raise ValueError("Feature contract must be a YAML mapping.")
    return data

def numerical_feature_names(contract: dict | None=None) -> list[str]:
    c=contract or load_feature_contract(); return [x["name"] for x in c.get("numerical_features",[])]

def categorical_feature_names(contract: dict | None=None) -> list[str]:
    c=contract or load_feature_contract(); return [x["name"] for x in c.get("categorical_features",[])]
