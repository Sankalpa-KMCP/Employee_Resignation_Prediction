# Repository Architecture

## Design goals

The repository is organized to make three things immediately visible during review or viva:

1. shared group work;
2. individually attributable team contributions;
3. reproducible implementation and evidence.

## Folder responsibilities

### `configs/`
Declarative configuration only. Model parameters and data/split contracts belong here.

### `data/`
`raw/` contains the assigned source dataset. `processed/` contains controlled derived artifacts such as the frozen split registry.

### `notebooks/`
The five root notebooks contain shared project foundations: audit, split creation, EDA, baselines and PCA.

`team_contributions/` contains individually attributable preprocessing and model work.

`group/` contains integrated pipeline/evaluation activities owned by the group as a whole.

### `src/`
Canonical reusable Python implementation. Model adapters, metrics, split logic and other reusable functionality belong here. New notebook code should call this layer where practical.

### `results/`
Generated evidence: tables, predictions, figures, tuning records and evaluation summaries. Existing model-result layouts were intentionally preserved during the architecture refactor to avoid rewriting historical experiment evidence.

### `models/`
Reserved for actual serialized/fitted model or preprocessing objects. Do not store metric CSVs or figures here.

### `reports/` and `presentations/`
Final assessment deliverables only.

### `tests/`
Automated repository and methodology validation.

## Team contribution rule

`notebooks/team_contributions/` is intentionally retained as a first-class folder.

- `preprocessing/`: IT-number-named preprocessing contributions.
- `models/`: Member 1–6 headline model notebooks.

The repository does not currently contain an authoritative mapping between the six preprocessing IT numbers and the Member 1–6 model roles. Do not infer that mapping from filenames.

## Group notebook rule

Integrated work belongs under `notebooks/group/`, including the combined pipeline, six-model comparison, fairness analysis and final holdout corroboration.

## Path rule

Moved notebooks now locate the project root by walking upward until they find the canonical feature contract rather than relying on a fixed number of `../` path segments. This allows notebooks to be launched from the project root or their own directories.

## Historical artifact rule

Model result folders were not aggressively normalized in this refactor. Their current model-specific structures encode earlier experiment history and are referenced by stored notebook outputs. Normalize them only when each model's final workflow is frozen and references can be updated together.
