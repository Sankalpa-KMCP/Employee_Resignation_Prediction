"""Create isolated experiment destinations and describe artifacts in this checkout."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib

import pandas as pd


def create_run_directory(root, model, output_dir=None):
    """Create a new directory beneath results/models/<model>/reruns.

    Explicit destinations must also be new directories beneath that boundary.
    Refuse existing paths and paths outside it, including the canonical results.
    """
    root = Path(root).resolve()
    boundary = (root / "results/models" / model / "reruns").resolve()
    if not boundary.is_relative_to(root / "results/models"):
        raise ValueError("Model run boundary must stay inside results/models.")
    if output_dir is None:
        output = boundary / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    else:
        output = Path(output_dir)
        output = (output if output.is_absolute() else root / output).resolve()
    if output == boundary or not output.is_relative_to(boundary):
        raise ValueError("New experiment output must be a directory beneath the model's reruns directory.")
    output.mkdir(parents=True, exist_ok=False)
    return output


def refresh_artifact_inventory(root, manifest_path):
    """Refresh an existing CSV inventory from available files, without creating models.

    Missing artifacts retain their path and role but have no current size or hash.
    The manifest's own size and checksum remain blank to avoid self-reference.
    Both the shared ``artifact`` and legacy ``relative_path`` schemas are
    supported, including inventories that originally had no size/hash fields.
    Where both occur, ``artifact`` is a display label and ``relative_path`` is
    the location. Stored Windows separators are normalized for portability.
    """
    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = pd.read_csv(manifest_path)
    path_column = "relative_path" if "relative_path" in manifest else "artifact"
    if path_column not in manifest:
        raise ValueError("Artifact inventory requires artifact or relative_path.")
    if "size_bytes" not in manifest:
        manifest["size_bytes"] = pd.Series(pd.NA, index=manifest.index, dtype="Int64")
    else:
        manifest["size_bytes"] = manifest["size_bytes"].astype("Int64")
    for index, row in manifest.iterrows():
        path = (root / str(row[path_column]).replace("\\", "/")).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Artifact inventory path escapes the project root.")
        exists = path.is_file()
        manifest.loc[index, "exists"] = exists
        self_reference = path == manifest_path
        manifest.loc[index, "size_bytes"] = path.stat().st_size if exists and not self_reference else pd.NA
        manifest.loc[index, "sha256"] = hashlib.sha256(path.read_bytes()).hexdigest() if exists and not self_reference else pd.NA
        manifest.loc[index, "availability_note"] = (
            "self-reference: checksum and size intentionally omitted" if self_reference
            else "available in this checkout" if exists else "not available in this checkout"
        )
    manifest.to_csv(manifest_path, index=False)
    return manifest
