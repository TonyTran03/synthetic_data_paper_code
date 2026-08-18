"""Generate Table 1: dataset sizes, feature counts, and class counts."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.revision.data_io import initialize_datasets


OUTPUT_DIR = PROJECT_ROOT / "table_outputs"

datasets, _ = initialize_datasets()
table = pd.DataFrame(
    [
        {
            "Dataset": dataset,
            "n": len(data["y"]),
            "p": len(data["feature_names"]),
            "n0": int((data["y"] == 0).sum()),
            "n1": int((data["y"] == 1).sum()),
        }
        for dataset, data in datasets.items()
    ]
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
table.to_csv(OUTPUT_DIR / "table_1_dataset_summary.csv", index=False)
(OUTPUT_DIR / "table_1_dataset_summary.tex").write_text(
    "% Requires \\usepackage{booktabs}\n"
    + table.to_latex(
        index=False,
        escape=False,
        caption="Binary classification datasets used for synthetic-data evaluation.",
        label="tab:datasets",
        column_format="lrrrr",
    ),
    encoding="utf-8",
)
print(table.to_string(index=False))
