# Final Report

`FINAL_REPORT.docx` is the requested Word report covering the nine required IT2011 report sections. It includes each model's measured variation results, the group comparison, calibration, fairness, the capacity simulation, limitations and linked references. No submission PDF is generated.

`FINAL_REPORT_DRAFT.md` is its editable content source. `scripts/build_final_report.py` creates the DOCX using Python and `python-docx`; run it with a document-authoring environment that provides this library, then render and inspect every page before delivery. Changes made directly in Word are not imported back into Markdown by this script.

`INDIVIDUAL_MODEL_EVIDENCE.md` gives each member's preprocessing and parameter comparison, measured metrics, validation method, conclusions and limitations. The short salary-masking exercise for Members 1, 2, 4 and 6 is exploratory; the six-model headline comparison uses the formal development OOF files.

Before formal submission the group must:
1. supply the verified Group ID;
2. review member identities/ownership and AI-use declaration;
3. retain the disclosed FT fixed-candidate scope and missing original per-fold adapter diagnostics;
4. retain the report's limit of no more than 15 pages after replacing placeholders;
5. update the title and filename using the confirmed Group ID and check the assessment submission instructions. The current deliverable uses Word as requested by the user.

Do not invent the Group ID from repository filenames.

## Completed formal six-model comparison update

FT-Transformer has now completed all 40 inner candidate evaluations and five outer winner refits. The registry uses `results/models/ft_transformer/strict_nested/oof_predictions.csv`; the older screening estimates remain historical. All six registry entries have completed tuning evidence. Group ranking uses mean outer-fold Average Precision; pooled AP is separate. Current numerical results and performance discussion are in `reports/Six_Model_Comparison.md`, which supersedes older FT-incomplete status statements and tables. No final holdout evaluation or deployment freeze has been performed.
