# Project: Employee Resignation Prediction

## Tech Stack
- Python, pandas, scikit-learn
- Jupyter Notebooks for analysis

## Project Structure
- `data/` - Contains raw and processed datasets. Never commit large data files.
- `notebooks/` - Jupyter Notebooks for data exploration and modeling.
- `docs/` - Documentation and reports.
- `presentations/` - Presentation assets and slides.
- `results/` - Output models, metrics, and generated figures.

## Code Conventions
- Use `pathlib.Path` for all file path operations (e.g., `Path("..").resolve()`) instead of os.path or hardcoded strings.
- Always set pandas display options when reviewing data (e.g., `pd.set_option("display.max_columns", None)`).
- Provide detailed Markdown cells in notebooks to explain the rationale behind data transformations and modeling choices.
- Keep notebooks clean: Restart and run all before committing to ensure reproducibility.

## Boundaries
- Never commit actual sensitive employee data to version control.
- Ensure all paths are relative to the project root so notebooks run on any machine.
- Ask before adding new heavy dependencies.

## Data Processing Rules
- Log shapes before and after filtering/dropping rows.
- Verify target distribution (`Resigned`) and explicitly mention handling of class imbalance if found.
