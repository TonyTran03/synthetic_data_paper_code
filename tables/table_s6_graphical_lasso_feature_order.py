"""Generate Table S6: real-data Graphical Lasso matrix feature ordering."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import synthetic_data_fidelity as analysis
from src.revision.data_io import initialize_datasets
from src.revision.graphical_lasso_plots import (
    fit_glasso_precision,
    get_real_structure_order,
    precision_to_partial_corr,
)


OUTPUT_DIR = PROJECT_ROOT / "table_outputs"

datasets, dataset_summary = initialize_datasets()
rows = []
for dataset, data in datasets.items():
    precision = fit_glasso_precision(
        data["X"], alpha=analysis.GRAPHICAL_LASSO_ALPHAS[dataset]
    )
    partial_correlation = precision_to_partial_corr(precision)
    order = get_real_structure_order(partial_correlation)
    for matrix_index, original_index in enumerate(order, start=1):
        rows.append(
            {
                "Dataset": dataset,
                "Matrix index": matrix_index,
                "Original index": int(original_index) + 1,
                "Feature": data["feature_names"][int(original_index)],
            }
        )

table = pd.DataFrame(rows)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
table.to_csv(OUTPUT_DIR / "table_s6_graphical_lasso_feature_order.csv", index=False)
(OUTPUT_DIR / "table_s6_graphical_lasso_feature_order.tex").write_text(
    (
        "% Requires \\usepackage{booktabs,longtable}\n"
        + table.to_latex(
            index=False,
            longtable=True,
            escape=True,
            caption=(
                "Feature ordering used in the Graphical Lasso edge-comparison "
                "matrices. Matrix positions were obtained by average-linkage "
                "clustering of the absolute real-data partial-correlation structure."
            ),
            label="tab:glasso_feature_order",
            column_format="lrrp{0.48\\textwidth}",
        )
    ),
    encoding="utf-8",
)
print(table.to_string(index=False))
