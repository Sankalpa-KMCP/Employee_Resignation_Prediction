# Team Contributions

This folder contains individually attributable work.

## Preprocessing contributions

`preprocessing/` preserves the six IT-number-named preprocessing notebooks exactly as individual contribution evidence, with only path-handling changes where needed for the refactored repository.

- `IT24100879_Outlier_Analysis.ipynb`
- `IT25100142_PCA.ipynb`
- `IT25100385_Categorical_Encoding.ipynb`
- `IT25100429_Standardization.ipynb`
- `IT25100445_Feature_Selection.ipynb`
- `IT25100565_Data_Quality.ipynb`

The verified Member 5 EBM contribution identifies IT25100429 as the standardization owner. The remaining IT-to-model relationships are not established here and should be verified by the group.

## Model contributions

`models/` contains the six member-owned headline model notebooks:

- Member 1 — FT-Transformer
- Member 2 — CatBoost
- Member 3 — XGBoost
- Member 4 — LightGBM
- Member 5 — EBM
- Member 6 — TabNet

Each member should be able to explain model suitability, implementation, tuning/variants, validation, metrics, comparison, limitations and code/output during the viva.

Member 5 also has a supplementary Decision Tree demonstration. The standalone
Regularized Logistic Regression demonstration belongs to IT24100879.
`models/supplementary_logistic_regression/11_IT24100879_logistic_regression_extra.ipynb`
reviews nested L2 tuning, calibration, coefficients and risk × value-proxy ranking.
These demonstrations do not replace the assigned models or the shared PCA benchmark.

`reports/INDIVIDUAL_MODEL_EVIDENCE.md` collects compact comparison tables for all six members. The short paired preprocessing checks for Members 1, 2, 4 and 6 are exploratory and are clearly distinguished from their formal five-fold OOF results.

Member 1's `models/member_1_ft_transformer/05_m1_ft_transformer.ipynb` contains the complete formal evaluation in 24 cells (12 Markdown and 12 code), with six inline figure groups and a compact file-derived historical summary. The executable companion `05_m1_ft_transformer_historical.ipynb` preserves all 15 historical code cells. Both default to saved review without training or artifact export. Explicit historical FT02 or formal nested repetitions write to isolated rerun directories; the group registry continues to use the canonical strict nested evidence.
