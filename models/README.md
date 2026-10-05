# Serialized Models

`models/*/*` is git-ignored, so serialized models are **not** in a fresh clone. Restore them locally instead of committing them.

## CatBoost (`models/catboost/`)

The promoted v2.1.0 development model (candidate R09, unweighted, 28 trees, seed 42) is used by `notebooks/team_contributions/models/member_2_catboost/06_m2_catboost.ipynb` (sections 11 and 12b). Rebuild it (about a second, development rows only, deterministic) with:

```bash
python scripts/restore_catboost_model.py          # writes only missing files
python scripts/restore_catboost_model.py --force  # rebuild even if present
```

This writes `models/catboost/model.cbm` and `models/catboost/metadata.json`. The rebuilt model reproduces the originally saved model's predictions exactly.

No other family has a serialized model; no overall "final" production model is frozen because the evidence does not justify an operational winner.
