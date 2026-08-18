"""Generate Table S5: class-specific GMM AIC and BIC model selection."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.revision.data_io import initialize_datasets


SEED = 42
CANDIDATE_COMPONENTS = (2, 3, 4, 5)
OUTPUT_DIR = PROJECT_ROOT / "table_outputs"

datasets, dataset_summary = initialize_datasets()
rows = []
for dataset, data in datasets.items():
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"], dtype=int)
    for class_label in np.unique(y):
        class_values = X[y == class_label]
        scaled_values = StandardScaler().fit_transform(class_values)
        candidate_rows = []
        for components in CANDIDATE_COMPONENTS:
            model = GaussianMixture(
                n_components=components,
                covariance_type="full",
                reg_covar=1e-4,
                n_init=5,
                random_state=SEED,
            ).fit(scaled_values)
            candidate_rows.append(
                {
                    "Dataset": dataset,
                    "Class": int(class_label),
                    "n": len(class_values),
                    "K": components,
                    "AIC": model.aic(scaled_values),
                    "BIC": model.bic(scaled_values),
                }
            )
        best_aic = min(candidate_rows, key=lambda row: row["AIC"])["K"]
        best_bic = min(candidate_rows, key=lambda row: row["BIC"])["K"]
        for row in candidate_rows:
            row["AIC-selected K"] = best_aic
            row["BIC-selected K"] = best_bic
            rows.append(row)

long_table = pd.DataFrame(rows)
score_table = long_table.pivot_table(
    index=["Dataset", "Class", "n"],
    columns="K",
    values=["AIC", "BIC"],
    aggfunc="first",
).reset_index()
score_table.columns = [
    column
    if isinstance(column, str)
    else column[0]
    if column[1] == ""
    else f"{column[0]} (K={column[1]})"
    for column in score_table.columns
]
selections = long_table.groupby(["Dataset", "Class"], as_index=False).first()[
    ["Dataset", "Class", "AIC-selected K", "BIC-selected K"]
]
table = score_table.merge(selections, on=["Dataset", "Class"])
table["AIC/BIC agree"] = table["AIC-selected K"] == table["BIC-selected K"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
table.to_csv(OUTPUT_DIR / "table_s5_gmm_aic_bic_selection.csv", index=False)
latex = table.to_latex(
        index=False,
        escape=False,
        float_format="%.2f",
        caption=(
            "AIC and BIC model-selection results for class-specific Gaussian "
            "mixture models."
        ),
        label="tab:gmm_aic_bic",
    )
latex = latex.replace(
    "\\begin{tabular}",
    "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}",
).replace("\\end{tabular}", "\\end{tabular}%\n}")
(OUTPUT_DIR / "table_s5_gmm_aic_bic_selection.tex").write_text(
    "% Requires \\usepackage{booktabs,graphicx}\n" + latex,
    encoding="utf-8",
)
print(table.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
