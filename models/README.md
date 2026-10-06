# Serialized Models

`models/*/*` is git-ignored, so serialized models are **not** in a fresh clone. Rebuild them locally instead of committing them.

## CatBoost — IT25100565 (`models/catboost/`)

`notebooks/team_contributions/models/member_2_catboost/06_m2_catboost.ipynb` uses `models/catboost/catboost_resignation_model.cbm` and `models/catboost/metadata.json` (B21, unweighted, 10 trees, seed 42, all 80,000 development rows). This folder is git-ignored like every model. In a fresh clone the notebook rebuilds the same deterministic model **in memory** during saved review, writing nothing, so no restore step is needed. The committed `results/models/catboost/model_metadata.json` carries the metadata the prediction section needs.

The previous R09 model (`model.cbm`, `catboost_development_model.cbm`) and its restore script were removed in this update; they remain in Git history.

No other family has a serialized model; no overall "final" production model is frozen because the evidence does not justify an operational winner.
