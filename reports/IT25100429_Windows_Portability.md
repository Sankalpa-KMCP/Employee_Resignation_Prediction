# EBM formal-input line-ending portability

The manifest writer and loader hash bytes after replacing CRLF with LF. They do not normalize whitespace, numbers, JSON formatting, encoding or lone CR bytes. LF hashes remain unchanged, and prior LF manifests remain compatible. The manifest declares `hash_normalization: crlf_to_lf`. Unsupported conventions are rejected. The writer reads the configuration from the supplied project root.

## Verification performed
- 76 relevant tests passed on macOS, including CRLF loading, CRLF manifest generation followed by LF loading, changed-config rejection, changed-artifact rejection, and byte preservation beyond CRLF.
- A fresh local Git clone of a temporary snapshot containing the patch was checked out with actual `core.autocrlf=true` conversion. Config, prediction CSV, fold metrics CSV and summary JSON had CRLF bytes; their raw hashes differed from the manifest, and their normalized hashes matched. The formal loader returned 80,000 OOF records and mean fold AP .0995577376800461.
- The frozen split registry retained LF as required by `.gitattributes`.

These checks ran on macOS. Native Windows execution has not yet been performed because this session has no Windows machine or available Windows runner.

## Native Windows verification
The manual GitHub Actions workflow `.github/workflows/ebm-windows-portability.yml` creates a fresh checkout on `windows-latest`, enables `core.autocrlf=true` before checkout, asserts native Windows and CRLF bytes, loads the formal inputs, and runs the regression suite. It deliberately does not regenerate the manifest before loading it.

After the changes are pushed to GitHub, open Actions, select **EBM Windows formal-input portability**, and choose **Run workflow** on the updated branch. Only a successful run establishes native Windows verification. The workflow has been saved locally, not dispatched.

## Local branch integration verification — 2026-10-02

The EBM changes were imported into `refactor/file-architecture` on Windows with Python 3.14. The 96 targeted EBM/configuration/notebook/evaluation tests passed, including CRLF loading, manifest regeneration, and rejection of altered content. The manifest-bound formal loader returned 80,000 OOF rows and five folds; the six candidate files contain 480,000 validated development-only rows in total. Four group analysis notebooks executed successfully with the canonical EBM input. Full EBM retraining and the fresh Windows GitHub Actions checkout workflow were not run during this integration.
