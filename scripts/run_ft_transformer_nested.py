"""Execute/resume the authoritative FT01–FT08 nested evaluation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
if __name__ == "__main__":
    from src.models.ft_transformer_execution import run
    run(ROOT)
