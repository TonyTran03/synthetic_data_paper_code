"""Generate Table 2 from 50 AUC and 20 predictive-utility repetitions."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import synthetic_data_fidelity as analysis
from src.revision.data_io import initialize_datasets


SEED = 42
CVAE_EPOCHS = 200
AUC_REPETITIONS = 50
UTILITY_REPETITIONS = 20
OUTPUT_DIR = PROJECT_ROOT / "table_outputs"
METHOD_ORDER = [
    "Bootstrap",
    "Column-wise",
    "GMM",
    "CVAE",
    "SMOTE",
    "GMM-guided SMOTE",
]

datasets, dataset_summary = initialize_datasets()
synthetic_cohorts = analysis.generate_cohorts(
    datasets,
    methods=analysis.METHOD_ORDER,
    seed=SEED,
    cvae_epochs=CVAE_EPOCHS,
)
auc_results = analysis.compute_origin_auc(
    datasets,
    synthetic_cohorts,
    repeats=AUC_REPETITIONS,
    seed=SEED,
)
utility_results = analysis.compute_tstr_runs(
    datasets,
    synthetic_cohorts,
    repeats=UTILITY_REPETITIONS,
    seed=SEED,
)
utility_results["utility_gap_abs"] = (
    utility_results["trtr_f1"] - utility_results["tstr_f1"]
).abs()

auc_summary = (
    auc_results.groupby(["dataset", "method"], as_index=False)
    .agg(AUC=("separability_auc", "mean"))
)
utility_summary = (
    utility_results.groupby(["dataset", "method"], as_index=False)
    .agg(
        TSTR_F1=("tstr_f1", "mean"),
        utility_gap=("utility_gap_abs", "mean"),
    )
)
table = auc_summary.merge(utility_summary, on=["dataset", "method"])
table["dataset"] = pd.Categorical(table["dataset"], list(datasets), ordered=True)
table["method"] = pd.Categorical(table["method"], METHOD_ORDER, ordered=True)
table = table.sort_values(["dataset", "method"]).rename(
    columns={
        "dataset": "Dataset",
        "method": "Method",
        "TSTR_F1": "TSTR F1",
        "utility_gap": "|TRTR-TSTR|",
    }
)
table["Method"] = table["Method"].astype(str).replace(
    {"GMM-guided SMOTE": "GMM-SMOTE"}
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
table.to_csv(OUTPUT_DIR / "table_2_discriminator_utility_summary.csv", index=False)
(OUTPUT_DIR / "table_2_discriminator_utility_summary.tex").write_text(
    "% Requires \\usepackage{booktabs}\n"
    + table.to_latex(
        index=False,
        escape=False,
        float_format="%.2f",
        caption=(
            "Main summary of discriminator and utility metrics across datasets "
            "and methods. AUC values are means across 50 discriminator "
            "repetitions; utility metrics are means across 20 repetitions."
        ),
        label="tab:main_metrics",
        column_format="llccc",
    ),
    encoding="utf-8",
)
print(table.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
