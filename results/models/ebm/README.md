# EBM result provenance

Canonical group input: `individual_IT25100429/oof_predictions.csv`. Its sibling `fold_metrics.csv` and `summary_metrics.json` describe the raw-candidate nested evaluation. `formal_input_manifest.json` binds paths and hashes to the authoritative config.

Files directly in this directory are historical default-only evidence. Never use top-level `oof_predictions.csv` or `ebm_oof_predictions.csv` for formal group comparison. The six-variety candidate table is descriptive; it is not the primary nested OOF.

Manifest text checksums normalize CRLF to LF before hashing. All other bytes remain significant. The manifest writer and loader use the same convention (`hash_normalization: crlf_to_lf`), so Git Windows checkout conversion does not invalidate the formal inputs.
