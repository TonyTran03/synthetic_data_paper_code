"""Generate the summary and complete feature-level marginal tables S1-S4."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, ttest_ind

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import synthetic_data_fidelity as analysis
from src.revision.data_io import initialize_datasets


SEED = 42
CVAE_EPOCHS = 200
OUTPUT_DIR = PROJECT_ROOT / "table_outputs"
METHODS = ["Bootstrap", "Column-wise", "CVAE", "GMM"]
TABLE_NUMBERS = {"HIV": "s2", "Breast Cancer": "s3", "Diabetes": "s4"}

datasets, dataset_summary = initialize_datasets()
synthetic_cohorts = analysis.generate_cohorts(
    datasets,
    methods=analysis.METHOD_ORDER,
    seed=SEED,
    cvae_epochs=CVAE_EPOCHS,
)

rows = []
for dataset, data in datasets.items():
    X_real = np.asarray(data["X"], dtype=float)
    for method in METHODS:
        X_synthetic = np.asarray(synthetic_cohorts[dataset][method][0], dtype=float)
        for feature_number, feature_name in enumerate(data["feature_names"], start=1):
            real_values = X_real[:, feature_number - 1]
            synthetic_values = X_synthetic[:, feature_number - 1]
            t_result = ttest_ind(
                real_values,
                synthetic_values,
                equal_var=False,
                nan_policy="omit",
            )
            ks_result = ks_2samp(real_values, synthetic_values)
            real_mean = float(np.nanmean(real_values))
            synthetic_mean = float(np.nanmean(synthetic_values))
            rows.append(
                {
                    "Dataset": dataset,
                    "Method": method,
                    "No.": feature_number,
                    "Feature": feature_name,
                    "n real": int(np.isfinite(real_values).sum()),
                    "n synthetic": int(np.isfinite(synthetic_values).sum()),
                    "Real mean": real_mean,
                    "Synthetic mean": synthetic_mean,
                    "Difference": synthetic_mean - real_mean,
                    "Welch t": float(t_result.statistic),
                    "p-value": float(t_result.pvalue),
                    "KS": float(ks_result.statistic),
                    "p < 0.05": "Yes" if t_result.pvalue < 0.05 else "No",
                }
            )

complete_table = pd.DataFrame(rows)
summary_table = (
    complete_table.assign(significant=complete_table["p-value"] < 0.05)
    .groupby(["Dataset", "Method"], as_index=False)
    .agg(
        Features=("Feature", "count"),
        **{
            "p < 0.05": ("significant", "sum"),
            "Median p": ("p-value", "median"),
            "Mean KS": ("KS", "mean"),
        },
    )
)
summary_table["Dataset"] = pd.Categorical(
    summary_table["Dataset"], list(datasets), ordered=True
)
summary_table["Method"] = pd.Categorical(
    summary_table["Method"], METHODS, ordered=True
)
summary_table = summary_table.sort_values(["Dataset", "Method"])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
summary_table.to_csv(OUTPUT_DIR / "table_s1_marginal_summary.csv", index=False)
(OUTPUT_DIR / "table_s1_marginal_summary.tex").write_text(
    "% Requires \\usepackage{booktabs}\n"
    + summary_table.to_latex(
        index=False,
        escape=False,
        float_format="%.3f",
        caption=(
            "Summary of feature-level marginal comparisons for the three "
            "datasets and four original synthesis methods. Raw Welch "
            "two-sample t-test p-values are reported without multiplicity "
            "adjustment; KS denotes the two-sample Kolmogorov-Smirnov statistic."
        ),
        label="tab:marginal_summary",
        column_format="llrrrr",
    ),
    encoding="utf-8",
)

for dataset, table_number in TABLE_NUMBERS.items():
    dataset_table = complete_table.loc[
        complete_table["Dataset"] == dataset
    ].drop(columns="Dataset")
    filename_dataset = dataset.lower().replace(" ", "_")
    dataset_table.to_csv(
        OUTPUT_DIR / f"table_{table_number}_{filename_dataset}_marginal_comparisons.csv",
        index=False,
    )
    (OUTPUT_DIR / f"table_{table_number}_{filename_dataset}_marginal_comparisons.tex").write_text(
        "% Requires \\usepackage{booktabs,longtable}\n"
        + dataset_table.to_latex(
            index=False,
            longtable=True,
            escape=True,
            float_format="%.4g",
            caption=(
                f"Complete feature-level marginal comparisons for {dataset}. "
                "Difference is synthetic minus real. Welch p-values are raw "
                "and unadjusted. KS is the two-sample Kolmogorov-Smirnov statistic."
            ),
            label=f"tab:{table_number}_marginal_{filename_dataset}",
            column_format="lrp{0.22\\textwidth}rrrrrrrrl",
        ),
        encoding="utf-8",
    )

print(summary_table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
