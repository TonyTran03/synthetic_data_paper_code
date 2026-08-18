"""Compare fixed K=2 GMMs with class-specific AIC and BIC selections."""

from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.gmm import sample_gmm
from src import synthetic_data_fidelity as analysis
from src.revision.common import class_counts
from src.revision.data_io import initialize_datasets


SEED = 42
GMM_SEED = SEED
AUC_REPETITIONS = int(os.environ.get("GMM_AUC_REPETITIONS", "50"))
UTILITY_REPETITIONS = int(os.environ.get("GMM_UTILITY_REPETITIONS", "20"))
CANDIDATE_COMPONENTS = (2, 3, 4, 5)
TABLE_OUTPUT_DIR = Path(
    os.environ.get("GMM_TABLE_OUTPUT_DIR", PROJECT_ROOT / "table_outputs")
)

SPECIFICATION_COLORS = {
    "2 components": "#6A5ACD",
    "AIC-selected": "#009E73",
    "BIC-selected": "#D55E00",
}


def estimate_component_criteria(datasets):
    """Fit the candidate full-covariance GMMs separately within each class."""
    rows = []
    for dataset, data in datasets.items():
        X = np.asarray(data["X"], dtype=float)
        y = np.asarray(data["y"], dtype=int)
        for class_label in np.unique(y):
            class_values = X[y == class_label]
            scaled_values = StandardScaler().fit_transform(class_values)
            class_fits = []
            for components in CANDIDATE_COMPONENTS:
                model = GaussianMixture(
                    n_components=components,
                    covariance_type="full",
                    reg_covar=1e-4,
                    n_init=5,
                    random_state=SEED,
                ).fit(scaled_values)
                class_fits.append(
                    (components, model.aic(scaled_values), model.bic(scaled_values))
                )
            minimum_aic = min(class_fits, key=lambda values: values[1])[0]
            minimum_bic = min(class_fits, key=lambda values: values[2])[0]
            for components, aic, bic in class_fits:
                rows.append(
                    {
                        "dataset": dataset,
                        "class": int(class_label),
                        "n_samples": len(class_values),
                        "components": components,
                        "AIC": aic,
                        "BIC": bic,
                        "AIC_selected": components == minimum_aic,
                        "BIC_selected": components == minimum_bic,
                    }
                )
    return pd.DataFrame(rows)


def selected_components(criteria, criterion):
    selected = criteria.loc[criteria[f"{criterion}_selected"]]
    return {
        dataset: tuple(
            selected.loc[selected["dataset"] == dataset]
            .sort_values("class")["components"]
            .astype(int)
        )
        for dataset in criteria["dataset"].unique()
    }


def generate_gmm_cohorts(datasets, component_specification):
    cohorts = {}
    for dataset, data in datasets.items():
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=int)
        n0, n1 = class_counts(y)
        cohorts[dataset] = {
            "GMM": sample_gmm(
                X,
                y,
                n0,
                n1,
                seed=GMM_SEED,
                n_components=component_specification[dataset],
                covariance_type="full",
                reg_covar=1e-4,
            )
        }
    return cohorts


datasets, dataset_summary = initialize_datasets()
component_criteria = estimate_component_criteria(datasets)
aic_components = selected_components(component_criteria, "AIC")
bic_components = selected_components(component_criteria, "BIC")
component_specifications = {
    "2 components": {dataset: (2, 2) for dataset in datasets},
    "AIC-selected": aic_components,
    "BIC-selected": bic_components,
}

auc_tables = []
kld_tables = []
utility_tables = []
for specification, component_counts in component_specifications.items():
    cohorts = generate_gmm_cohorts(datasets, component_counts)

    auc = analysis.compute_origin_auc(
        datasets, cohorts, repeats=AUC_REPETITIONS, seed=SEED
    ).assign(specification=specification)
    kld = analysis.compute_feature_kld_table(
        datasets, cohorts
    ).assign(specification=specification)
    utility = analysis.compute_tstr_runs(
        datasets, cohorts, repeats=UTILITY_REPETITIONS, seed=SEED
    ).assign(specification=specification)
    utility["utility_gap_abs"] = (
        utility["trtr_f1"] - utility["tstr_f1"]
    ).abs()

    auc_tables.append(auc)
    kld_tables.append(kld)
    utility_tables.append(utility)

auc_results = pd.concat(auc_tables, ignore_index=True)
kld_results = pd.concat(kld_tables, ignore_index=True)
utility_results = pd.concat(utility_tables, ignore_index=True)

metric_sources = {
    "AUC": (auc_results, "separability_auc"),
    "KLD": (kld_results, "kld"),
    "Utility gap": (utility_results, "utility_gap_abs"),
}

fig, axes = plt.subplots(3, 3, figsize=(9.1, 10.4), sharey="row", squeeze=False)
for row, (metric_label, (table, value_column)) in enumerate(metric_sources.items()):
    for column, dataset in enumerate(datasets):
        ax = axes[row, column]
        for position, specification in enumerate(component_specifications):
            values = table.loc[
                (table["dataset"] == dataset)
                & (table["specification"] == specification),
                value_column,
            ].dropna().to_numpy(dtype=float)
            color = SPECIFICATION_COLORS[specification]
            if np.ptp(values) > 0:
                violin = ax.violinplot(
                    [values],
                    positions=[position],
                    widths=0.72,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                violin["bodies"][0].set_facecolor(color)
                violin["bodies"][0].set_edgecolor(color)
                violin["bodies"][0].set_alpha(0.45)
            else:
                ax.hlines(values[0], position - 0.28, position + 0.28, color=color, linewidth=3)
            ax.errorbar(
                position,
                values.mean(),
                yerr=values.std(ddof=1),
                fmt="o",
                color=color,
                markerfacecolor="white",
                markersize=5,
                capsize=3,
                linewidth=1.4,
                zorder=4,
            )

        if row == 0:
            ax.axhline(0.5, color="#777777", linewidth=1)
            ax.set_ylim(0.47, 1.02)
            ax.set_title(dataset, color="black", weight="bold")
        else:
            ax.set_ylim(bottom=0)
        if column == 0:
            ax.set_ylabel(metric_label)
        ax.set_xticks(range(3))
        ax.set_xticklabels(component_specifications, rotation=24, ha="right")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.75, alpha=0.55)

fig.subplots_adjust(left=0.09, right=0.99, top=0.955, bottom=0.10, wspace=0.16, hspace=0.44)
analysis.label_figure_panels(fig)

# Table S5 and a compact performance comparison are generated as CSV and LaTeX.
wide_criteria = component_criteria.pivot_table(
    index=["dataset", "class", "n_samples"],
    columns="components",
    values=["AIC", "BIC"],
    aggfunc="first",
).reset_index()
wide_criteria.columns = [
    column
    if isinstance(column, str)
    else column[0]
    if column[1] == ""
    else f"{column[0]} (K={column[1]})"
    for column in wide_criteria.columns
]
wide_criteria["AIC-selected K"] = [
    aic_components[dataset][int(class_label)]
    for dataset, class_label in zip(wide_criteria["dataset"], wide_criteria["class"])
]
wide_criteria["BIC-selected K"] = [
    bic_components[dataset][int(class_label)]
    for dataset, class_label in zip(wide_criteria["dataset"], wide_criteria["class"])
]
wide_criteria["criteria agree"] = (
    wide_criteria["AIC-selected K"] == wide_criteria["BIC-selected K"]
)

comparison_rows = []
for dataset in datasets:
    for specification, counts in component_specifications.items():
        auc_values = auc_results.query(
            "dataset == @dataset and specification == @specification"
        )["separability_auc"]
        kld_values = kld_results.query(
            "dataset == @dataset and specification == @specification"
        )["kld"]
        utility_values = utility_results.query(
            "dataset == @dataset and specification == @specification"
        )["utility_gap_abs"]
        comparison_rows.append(
            {
                "Dataset": dataset,
                "GMM specification": specification,
                "K0/K1": f"{counts[dataset][0]}/{counts[dataset][1]}",
                "AUC": f"{auc_values.mean():.3f} +/- {auc_values.std(ddof=1):.3f}",
                "KLD": f"{kld_values.mean():.3f} +/- {kld_values.std(ddof=1):.3f}",
                "Utility gap": f"{utility_values.mean():.3f} +/- {utility_values.std(ddof=1):.3f}",
            }
        )
comparison_table = pd.DataFrame(comparison_rows)

TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
wide_criteria.to_csv(TABLE_OUTPUT_DIR / "table_s5_gmm_aic_bic_selection.csv", index=False)
comparison_table.to_csv(TABLE_OUTPUT_DIR / "gmm_fixed2_aic_bic_comparison.csv", index=False)
(TABLE_OUTPUT_DIR / "table_s5_gmm_aic_bic_selection.tex").write_text(
    wide_criteria.to_latex(index=False, escape=False, float_format="%.2f"),
    encoding="utf-8",
)
(TABLE_OUTPUT_DIR / "gmm_fixed2_aic_bic_comparison.tex").write_text(
    comparison_table.to_latex(
        index=False,
        escape=False,
        caption="Comparison of fixed two-component, AIC-selected, and BIC-selected GMM specifications. Values are mean $\\pm$ standard deviation.",
        label="tab:gmm_component_sensitivity",
        column_format="lllccc",
    ),
    encoding="utf-8",
)
print(comparison_table.to_string(index=False))
plt.show()
