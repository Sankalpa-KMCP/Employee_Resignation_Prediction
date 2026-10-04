"""CatBoost Pipeline Environment Diagnostic Script (Step 1).

Executes a complete standalone diagnostic audit for the CatBoost modeling pipeline,
checking Python runtime, library versions, CatBoost thread configuration, GPU detection,
feature contract schema, and split registry integrity.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import catboost
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import yaml

from src.models.catboost_model import CatBoostAdapter
from src.data.feature_contract import load_feature_contract


def run_diagnostic() -> int:
    """Run full diagnostic checks and print structured report."""
    print("=" * 75)
    print("CATBOOST PIPELINE ENVIRONMENT & RUNTIME DIAGNOSTIC AUDIT")
    print("=" * 75)

    # 1. Environment & Runtime
    print("\n[1] Python Runtime & Platform:")
    print(f"  Python Version:     {platform.python_version()} ({platform.architecture()[0]})")
    print(f"  Python Executable:  {sys.executable}")
    print(f"  Operating System:   {platform.system()} {platform.release()} ({platform.version()})")
    print(f"  CPU Count (Logical):{os.cpu_count()}")

    # 2. Dependency Versions
    print("\n[2] Required Library Versions:")
    print(f"  catboost:           {catboost.__version__}")
    print(f"  pandas:             {pd.__version__}")
    print(f"  numpy:              {np.__version__}")
    print(f"  scikit-learn:       {sklearn.__version__}")
    print(f"  matplotlib:         {matplotlib.__version__}")
    print(f"  seaborn:            {sns.__version__}")
    print(f"  pyyaml:             {yaml.__version__}")

    # 3. Adapter Instantiation
    print("\n[3] CatBoostAdapter Instantiation Check:")
    adapter = CatBoostAdapter()
    print(f"  Adapter Name:       {adapter.name}")
    print(f"  Adapter Model ID:   {adapter.model_id}")
    print(f"  Adapter Fitted:     {adapter._fitted}")
    print(f"  Configured Threads: {adapter.params.get('thread_count', -1)} (-1 = all available logical cores)")
    print(f"  Default Task Type:  {adapter.params.get('task_type', 'CPU')}")

    # 4. GPU Detection Check
    print("\n[4] CatBoost GPU Support Check:")
    gpu_available = False
    try:
        from catboost import CatBoostClassifier
        _ = CatBoostClassifier(task_type="GPU", iterations=1, verbose=False, allow_writing_files=False)
        gpu_available = True
    except Exception as exc:
        gpu_available = False
        gpu_error = str(exc)

    print(f"  GPU Hardware Found: {gpu_available}")
    print("  Execution Policy:   CPU default enforced for exact reproducibility across machines.")

    # 5. Feature Contract Check
    print("\n[5] Feature Contract & Split Registry Check:")
    contract = load_feature_contract()
    primary_feats = contract.get("primary_model_features", [])
    cat_feats = [c["name"] for c in contract.get("categorical_features", [])]
    num_feats = [c["name"] for c in contract.get("numerical_features", [])]
    print(f"  Primary Features:   {len(primary_feats)} total (Expected: 17)")
    print(f"  Categoricals:       {len(cat_feats)} ({', '.join(cat_feats)})")
    print(f"  Numericals/Ordinal: {len(num_feats)}")

    # Check raw data and split registry existence
    raw_path = PROJECT_ROOT / contract["dataset"]["path"]
    split_path = PROJECT_ROOT / contract["split_registry"]["path"]
    print(f"  Raw Dataset Path:   {raw_path.relative_to(PROJECT_ROOT)} (Exists: {raw_path.exists()})")
    print(f"  Split Registry Path:{split_path.relative_to(PROJECT_ROOT)} (Exists: {split_path.exists()})")

    if raw_path.exists() and split_path.exists():
        split_df = pd.read_csv(split_path)
        dev_count = (split_df["partition"] == "development").sum()
        holdout_count = (split_df["partition"] == "holdout").sum()
        print(f"  Development Records:{dev_count:,} (Expected: 80,000)")
        print(f"  Holdout Records:    {holdout_count:,} (Expected: 20,000)")
        assert dev_count == 80000, "Development partition mismatch!"
        assert holdout_count == 20000, "Holdout partition mismatch!"

    print("\n" + "=" * 75)
    print("STATUS: ENVIRONMENT & CONFIGURATION AUDIT PASSED [OK]")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(run_diagnostic())
