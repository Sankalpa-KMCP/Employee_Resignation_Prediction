"""Centralized model configuration loader, schema validator, and candidate resolver.

This module is the authoritative I/O boundary for model experiment configurations.
Model adapters and training loops must not load or parse YAML files directly;
they must receive validated configuration dictionaries or resolved parameter
mappings produced by this module.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_MODELS_DIR = PROJECT_ROOT / "configs" / "models"

# Semantic class-imbalance strategies supported by the architecture.
# Adapters map these semantic declarations to their underlying estimator parameters.
VALID_CLASS_IMBALANCE_STRATEGIES = frozenset({"none", "balanced", "dynamic_ratio"})

# Top-level sections required in the modernized target schema.
TARGET_REQUIRED_SECTIONS = frozenset(
    {"model", "reproducibility", "inner_validation", "candidates"}
)

# Top-level sections required in legacy/transitional configs.
LEGACY_REQUIRED_SECTIONS = frozenset(
    {"model", "reproducibility", "fixed_params", "validation", "search_space"}
)


def _detect_unresolved_placeholders(obj: Any, path: str = "") -> list[str]:
    """Recursively identify TODO or unpopulated placeholder tokens in data structures."""
    findings: list[str] = []
    if isinstance(obj, str):
        if "TODO" in obj or "PLACEHOLDER" in obj:
            findings.append(f"{path}: {obj!r}")
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else str(key)
            if isinstance(key, str) and ("TODO" in key or "PLACEHOLDER" in key):
                findings.append(f"{current_path} (key): {key!r}")
            findings.extend(_detect_unresolved_placeholders(value, current_path))
    elif isinstance(obj, (list, tuple, set)):
        for idx, item in enumerate(obj):
            current_path = f"{path}[{idx}]"
            findings.extend(_detect_unresolved_placeholders(item, current_path))
    return findings


def validate_model_config(
    config: Mapping[str, Any],
    strict_schema: bool = False,
    model_id: str | None = None,
) -> None:
    """Validate model configuration dictionary against structural and methodological rules.

    Parameters
    ----------
    config : Mapping[str, Any]
        Parsed configuration dictionary.
    strict_schema : bool, default=False
        If True, validates strictly against the modernized target schema:
        requires discrete candidates, inner validation policy, semantic class imbalance,
        and forbids unresolved TODO placeholders.
        If False, supports transitional legacy configs while enforcing basic integrity.
    model_id : str or None, default=None
        Expected model short identifier (e.g., 'xgboost', 'catboost'). If provided,
        ensures metadata matches the expected identifier.

    Raises
    ------
    TypeError
        If config is not a dictionary/mapping.
    ValueError
        If required sections are missing, candidate IDs are duplicated, parameters
        are malformed, or strict-mode checks fail.
    """
    if not isinstance(config, Mapping):
        raise TypeError(f"Model configuration must be a mapping, got {type(config).__name__}")

    if "model" not in config or not isinstance(config["model"], Mapping):
        raise ValueError("Configuration missing required 'model' metadata mapping.")

    model_meta = config["model"]
    for required_meta in ("name", "model_id"):
        if required_meta not in model_meta:
            raise ValueError(f"Section 'model' missing required attribute '{required_meta}'.")

    cfg_model_id = str(model_meta["model_id"]).strip()
    if model_id is not None and cfg_model_id != model_id:
        # Allow prefix matching e.g. m3_xgboost vs xgboost
        if not (cfg_model_id.endswith(model_id) or model_id.endswith(cfg_model_id)):
            raise ValueError(
                f"Config model_id '{cfg_model_id}' does not match expected model_id '{model_id}'."
            )

    if "reproducibility" not in config or not isinstance(config["reproducibility"], Mapping):
        raise ValueError("Configuration missing required 'reproducibility' mapping.")

    repro = config["reproducibility"]
    if not any(k in repro for k in ("seed", "random_seed", "random_state")):
        raise ValueError("Section 'reproducibility' must define 'seed', 'random_seed', or 'random_state'.")

    # -------------------------------------------------------------------------
    # Target Strict Schema Validation
    # -------------------------------------------------------------------------
    if strict_schema:
        missing_sections = sorted(TARGET_REQUIRED_SECTIONS - set(config.keys()))
        if missing_sections:
            raise ValueError(f"Strict target schema missing required sections: {missing_sections}")

        # Strict Model Metadata
        for strict_meta in ("id", "implementation"):
            if strict_meta not in model_meta:
                raise ValueError(f"Strict target schema requires 'model.{strict_meta}'.")

        # Feature Contract Authority Check
        # Model configs must not duplicate the list of 17 predictor names.
        if "preprocessing" in config and isinstance(config["preprocessing"], Mapping):
            preproc = config["preprocessing"]
            for feat_key in ("features", "predictor_names", "numerical_features"):
                val = preproc.get(feat_key)
                if isinstance(val, (list, tuple)) and len(val) == 17:
                    raise ValueError(
                        f"Found full predictor list under 'preprocessing.{feat_key}'. "
                        "Do not duplicate the 17 predictor names in model configs; "
                        "configs/feature_contract.yaml remains authoritative."
                    )

        # Inner Validation Policy
        inner_val = config["inner_validation"]
        if not isinstance(inner_val, Mapping):
            raise ValueError("Section 'inner_validation' must be a mapping.")
        if "fraction" not in inner_val:
            raise ValueError("Section 'inner_validation' must define 'fraction'.")
        try:
            fraction = float(inner_val["fraction"])
            if not 0.0 < fraction < 0.5:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"'inner_validation.fraction' must be a float between 0.0 and 0.5, got {inner_val.get('fraction')!r}"
            ) from exc

        # Semantic Class Imbalance Strategy
        if "class_imbalance" in config:
            ci = config["class_imbalance"]
            if not isinstance(ci, Mapping):
                raise ValueError("Section 'class_imbalance' must be a mapping.")
            strategy = ci.get("strategy")
            if strategy not in VALID_CLASS_IMBALANCE_STRATEGIES:
                raise ValueError(
                    f"Invalid class imbalance strategy {strategy!r}. "
                    f"Allowed semantic strategies: {sorted(VALID_CLASS_IMBALANCE_STRATEGIES)}"
                )

        # Candidates Validation
        candidates = config["candidates"]
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise ValueError("Section 'candidates' must be a non-empty sequence of candidate definitions.")
        if len(candidates) == 0:
            raise ValueError("Section 'candidates' must declare at least one candidate configuration.")

        candidate_ids: set[str] = set()
        for idx, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ValueError(f"Candidate at index {idx} must be a mapping.")
            cid = candidate.get("id")
            if not cid or not isinstance(cid, str):
                raise ValueError(f"Candidate at index {idx} must define a non-empty string 'id'.")
            if cid in candidate_ids:
                raise ValueError(f"Duplicate candidate id '{cid}' detected in 'candidates'.")
            candidate_ids.add(cid)

            # Params must exist and be non-empty mapping
            if "params" in candidate:
                params = candidate["params"]
                if not isinstance(params, Mapping) or len(params) == 0:
                    raise ValueError(f"Candidate '{cid}' has no model parameters declared.")
            else:
                # Check for legacy TabNet format where parameters are top-level on candidate dict
                sub_params = {
                    k: v for k, v in candidate.items() if k not in ("id", "description")
                }
                if len(sub_params) == 0:
                    raise ValueError(f"Candidate '{cid}' has no model parameters declared.")

        # Strict Check: No unresolved TODO placeholders permitted in strict target schema
        todos = _detect_unresolved_placeholders(config)
        if todos:
            raise ValueError(
                f"Strict schema rejects unresolved TODO/placeholder items:\n  - "
                + "\n  - ".join(todos[:5])
            )

    # -------------------------------------------------------------------------
    # Transitional Legacy Schema Validation
    # -------------------------------------------------------------------------
    else:
        # Note: Transitional legacy support allows unmigrated v1.0/v4.4 YAMLs to parse
        # until all six headline models are migrated to the target candidate schema.
        if "candidates" in config and isinstance(config["candidates"], Sequence):
            candidate_ids = set()
            for idx, candidate in enumerate(config["candidates"]):
                if isinstance(candidate, Mapping) and "id" in candidate:
                    cid = str(candidate["id"])
                    if cid in candidate_ids:
                        raise ValueError(f"Duplicate candidate id '{cid}' detected in 'candidates'.")
                    candidate_ids.add(cid)


def load_model_config(
    model_id_or_path: str | Path,
    strict_schema: bool = False,
) -> dict[str, Any]:
    """Load, parse, and validate a model YAML configuration file.

    Parameters
    ----------
    model_id_or_path : str or Path
        Either a model identifier (e.g., 'xgboost', 'ft_transformer') which resolves
        to configs/models/{model_id}.yaml, or a direct file Path.
    strict_schema : bool, default=False
        Whether to enforce the strict modernized target schema.

    Returns
    -------
    dict[str, Any]
        Parsed and validated configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist on disk.
    ValueError
        If the YAML is malformed or fails schema validation.
    """
    raw_path = Path(model_id_or_path)
    if raw_path.suffix in (".yaml", ".yml") and raw_path.exists():
        config_path = raw_path
    elif (CONFIGS_MODELS_DIR / f"{model_id_or_path}.yaml").exists():
        config_path = CONFIGS_MODELS_DIR / f"{model_id_or_path}.yaml"
    elif raw_path.is_absolute():
        config_path = raw_path
    else:
        config_path = CONFIGS_MODELS_DIR / f"{model_id_or_path}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Model configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed YAML in configuration file {config_path}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Configuration file {config_path} did not parse into a top-level dictionary.")

    validate_model_config(parsed, strict_schema=strict_schema, model_id=config_path.stem)
    return parsed


def resolve_candidate(
    config: Mapping[str, Any],
    candidate_id: str | None = None,
) -> dict[str, Any]:
    """Resolve fixed parameters and candidate overrides into a deterministic execution dictionary.

    Does NOT mutate the input configuration mapping.

    Parameters
    ----------
    config : Mapping[str, Any]
        Model configuration dictionary.
    candidate_id : str or None, default=None
        Candidate ID to resolve. If None and 'candidates' exists, resolves the first declared candidate.

    Returns
    -------
    dict[str, Any]
        Resolved execution dictionary containing:
        - 'model': model identity metadata
        - 'reproducibility': reproducibility settings
        - 'inner_validation': inner-validation policy
        - 'class_imbalance': semantic class imbalance configuration
        - 'candidate_id': resolved candidate ID (or 'default')
        - 'params': merged estimator hyperparameter mapping
        - 'raw_config': copy of the full validated configuration
    """
    # Defensive deep copy ensures source mapping is never mutated
    cfg_copy = copy.deepcopy(dict(config))

    model_meta = cfg_copy.get("model", {})
    repro = cfg_copy.get("reproducibility", {})
    inner_val = cfg_copy.get("inner_validation", cfg_copy.get("validation", {}))
    class_imbalance = cfg_copy.get("class_imbalance", {})
    fixed_params = cfg_copy.get("fixed_params", {})

    merged_params: dict[str, Any] = copy.deepcopy(fixed_params)
    resolved_id = "default"

    # Resolve from discrete candidates if declared
    if "candidates" in cfg_copy and isinstance(cfg_copy["candidates"], Sequence):
        candidates = list(cfg_copy["candidates"])
        if len(candidates) == 0:
            raise ValueError("Config declares 'candidates' but the list is empty.")

        available_ids = [str(c.get("id")) for c in candidates if isinstance(c, Mapping)]

        if candidate_id is None:
            chosen = candidates[0]
            resolved_id = str(chosen.get("id", "C01"))
        else:
            matches = [c for c in candidates if isinstance(c, Mapping) and str(c.get("id")) == str(candidate_id)]
            if not matches:
                raise ValueError(
                    f"Unknown candidate_id {candidate_id!r}. Available candidates: {available_ids}"
                )
            chosen = matches[0]
            resolved_id = str(chosen.get("id"))

        # Candidate parameters can be under 'params' sub-mapping or top-level on candidate mapping
        if "params" in chosen and isinstance(chosen["params"], Mapping):
            candidate_params = copy.deepcopy(chosen["params"])
        else:
            candidate_params = {
                k: copy.deepcopy(v)
                for k, v in chosen.items()
                if k not in ("id", "description")
            }

        merged_params.update(candidate_params)

    elif candidate_id is not None:
        raise ValueError(
            f"Cannot resolve candidate_id {candidate_id!r}: configuration does not define 'candidates'."
        )

    return {
        "model": model_meta,
        "reproducibility": repro,
        "inner_validation": inner_val,
        "class_imbalance": class_imbalance,
        "candidate_id": resolved_id,
        "params": merged_params,
        "raw_config": cfg_copy,
    }
