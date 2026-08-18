"""Synthetic-data generation, fidelity metrics, and manuscript plots."""

from __future__ import annotations

import contextlib
import io
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from models.bootstrap import sample_bootstrap
from models.cvae import sample_cvae
from models.gmm import AIC_COMPONENTS_BY_DATASET, sample_gmm
from models.iid_columnwise import sample_columnwise
from models.smote import sample_gmm_guided_smote, sample_smote
from src.revision.common import (
    Config,
    DATASET_COLORS,
    add_confidence_ellipse,
    apply_manuscript_figure_style,
    class_counts,
    standardize_pair,
)
from src.revision.graphical_lasso_plots import (
    STATUS_COLORS,
    build_edge_status_matrix,
    compute_edge_recovery,
    compute_frobenius_deviation,
    compute_synthetic_only_rate,
    fit_glasso_precision,
    get_edge_set,
    get_real_structure_order,
    precision_to_partial_corr,
)
from src.revision.stats import (
    ablation_grid,
    ks_by_feature,
    mean_kld_by_feature,
    nn_distance_mean,
    one_run_origin_auc,
    rank_discriminating_features,
    stratified_subsample,
    tstr_values,
)

METHOD_ORDER = [
    "Bootstrap",
    "Column-wise",
    "GMM",
    "SMOTE",
    "GMM-guided SMOTE",
    "CVAE",
]
METHOD_COLORS = {
    "Bootstrap": "#6A5ACD",
    "Column-wise": "#CC79A7",
    "GMM": "#009E73",
    "SMOTE": "#0072B2",
    "GMM-guided SMOTE": "#56B4E9",
    "CVAE": "#D55E00",
}

GRAPHICAL_LASSO_ALPHAS = {
    "HIV": 0.504,
    "Breast Cancer": 0.502,
    "Diabetes": 0.0159,
}

PANEL_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def label_figure_panels(fig, axes=None, start=0):
    """Give every supplied Matplotlib axes its own sequential panel letter."""
    axes = list(fig.axes if axes is None else axes)
    if start < 0 or start + len(axes) > len(PANEL_LABELS):
        raise ValueError("Panel labels must fit within A-Z.")

    # Replace older group-level labels so a composite cannot contain duplicate
    # letters after this per-axes labelling pass.
    for ax in fig.axes:
        for text_artist in list(ax.texts):
            if text_artist.get_text() in set(PANEL_LABELS):
                text_artist.remove()

    for offset, ax in enumerate(axes):
        panel_label = ax.text(
            -0.055,
            1.045,
            PANEL_LABELS[start + offset],
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.5,
            weight="bold",
            color="#111111",
            clip_on=False,
            zorder=20,
        )
        panel_label.set_gid("panel-letter")
    return fig

def sample_method(
    X,
    y,
    method,
    seed=42,
    cvae_epochs=50,
    dataset=None,
):
    n0, n1 = class_counts(y)
    if method == "Bootstrap":
        return sample_bootstrap(X, y, n0, n1, seed=seed)
    if method == "Column-wise":
        return sample_columnwise(X, y, n0, n1, seed=seed)
    if method == "GMM":
        components = AIC_COMPONENTS_BY_DATASET.get(dataset, 2)
        return sample_gmm(X, y, n0, n1, seed=seed, n_components=components)
    if method == "SMOTE":
        return sample_smote(X, y, n0, n1, seed=seed)
    if method == "GMM-guided SMOTE":
        return sample_gmm_guided_smote(X, y, n0, n1, seed=seed)
    if method == "CVAE":
        with contextlib.redirect_stdout(io.StringIO()):
            return sample_cvae(
                X,
                y,
                n0,
                n1,
                seed=seed,
                cfg=Config(
                    seed=seed,
                    epochs=cvae_epochs,
                    x_transform="none",
                    latent_prior="normal",
                ),
            )
    raise ValueError(f"Unknown method: {method}")

def generate_cohorts(
    datasets,
    methods=METHOD_ORDER,
    seed=42,
    cvae_epochs=50,
):
    cohorts = {}
    for dataset, data in datasets.items():
        cohorts[dataset] = {}
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=int)
        for offset, method in enumerate(methods):
            print(f"[generate] {dataset} - {method}")
            cohorts[dataset][method] = sample_method(
                X,
                y,
                method,
                seed=seed + 101 * offset,
                cvae_epochs=cvae_epochs,
                dataset=dataset,
            )
    return cohorts

def compute_origin_auc(datasets, cohorts, repeats=5, seed=42):
    rows = []
    for dataset, method_data in cohorts.items():
        X_real = np.asarray(datasets[dataset]["X"])
        y_real = np.asarray(datasets[dataset]["y"], dtype=int)
        for method, (X_syn, y_syn) in method_data.items():
            for repeat in range(int(repeats)):
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "run": repeat,
                        "separability_auc": one_run_origin_auc(
                            X_real,
                            y_real,
                            X_syn,
                            y_syn,
                            seed=seed + 1009 * repeat,
                        ),
                    }
                )
    return pd.DataFrame(rows)

def compute_feature_kld_table(datasets, cohorts):
    """Return one KLD value per dataset, method, and feature."""

    rows = []
    for dataset, method_data in cohorts.items():
        X_real = np.asarray(datasets[dataset]["X"], dtype=float)
        names = list(
            datasets[dataset].get(
                "feature_names",
                [f"feature_{index + 1}" for index in range(X_real.shape[1])],
            )
        )
        for method, (X_syn, _) in method_data.items():
            values = mean_kld_by_feature(X_real, X_syn)
            for feature_index, value in enumerate(values):
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "feature_index": feature_index,
                        "feature": names[feature_index],
                        "kld": float(value),
                    }
                )
    return pd.DataFrame(rows)

def plot_fidelity_auc_kld_utility_gap(
    auc_runs,
    feature_kld,
    marginal_tests,
    tstr_runs,
    dataset_order=None,
    method_order=None,
    jitter_seed=42,
):
    """Plot AUC, feature KLD, and utility-gap distributions in a 3 x 3 grid."""
    dataset_order = list(dataset_order or dict.fromkeys(auc_runs["dataset"]))
    method_order = [
        method
        for method in (method_order or METHOD_ORDER)
        if method in set(auc_runs["method"])
    ]
    if len(dataset_order) != 3:
        raise ValueError("Figure 1 requires exactly three datasets")
    if "utility_gap_abs" not in tstr_runs.columns:
        tstr_runs = tstr_runs.assign(
            utility_gap_abs=(tstr_runs["trtr_f1"] - tstr_runs["tstr_f1"]).abs()
        )

    row_specs = [
        (auc_runs, "separability_auc", "AUC"),
        (feature_kld, "kld", "KLD"),
        (tstr_runs, "utility_gap_abs", "Utility gap"),
    ]
    # Portrait proportions allow the complete 3 x 3 figure and its caption to
    # occupy an A4 journal page when included at \textwidth.
    fig, axes = plt.subplots(3, 3, figsize=(9.1, 10.4), squeeze=False)

    def consistent_jitter(count, width=0.105):
        if count <= 1:
            return np.zeros(count, dtype=float)
        offsets = np.linspace(-width, width, count)
        return offsets[np.random.default_rng(jitter_seed + count).permutation(count)]

    for row, (table, value_column, row_label) in enumerate(row_specs):
        for col, dataset in enumerate(dataset_order):
            ax = axes[row, col]
            subset = table[table["dataset"] == dataset]
            values = [
                subset.loc[subset["method"] == method, value_column]
                .dropna()
                .to_numpy(dtype=float)
                for method in method_order
            ]
            boxes = ax.boxplot(
                values,
                positions=np.arange(len(method_order)),
                widths=0.62,
                patch_artist=True,
                showmeans=True,
                showfliers=False,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "white",
                    "markeredgecolor": "#333333",
                    "markeredgewidth": 0.8,
                    "markersize": 4.2,
                },
                medianprops={"color": "#111111", "linewidth": 1.35},
                whiskerprops={"color": "#666666", "linewidth": 1.0},
                capprops={"color": "#666666", "linewidth": 1.0},
            )
            for box, method in zip(boxes["boxes"], method_order):
                box.set_facecolor(METHOD_COLORS[method])
                box.set_edgecolor(METHOD_COLORS[method])
                box.set_alpha(0.58)
                box.set_linewidth(1.25)

            for position, method_values, method in zip(
                np.arange(len(method_order)), values, method_order
            ):
                ax.scatter(
                    position + consistent_jitter(len(method_values)),
                    method_values,
                    s=11,
                    color=METHOD_COLORS[method],
                    edgecolors="white",
                    linewidths=0.22,
                    alpha=0.38,
                    zorder=3,
                )

            if row == 0:
                ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.0)
                ax.set_ylim(0.47, 1.02)
                ax.set_title(
                    dataset,
                    color=DATASET_COLORS[dataset],
                    fontsize=12.5,
                    weight="bold",
                    pad=10,
                )
            elif row == 1:
                ax.set_ylim(bottom=0.0)
            elif row == 2:
                ax.set_ylim(0.0, 0.40)
                ax.set_yticks(np.arange(0.0, 0.401, 0.10))
                ax.tick_params(axis="y", labelleft=col == 0)

            if col == 0:
                ax.set_ylabel(row_label, fontsize=11, weight="semibold")
            else:
                ax.set_ylabel("")
            ax.set_xticks(np.arange(len(method_order)))
            ax.set_xticklabels(method_order, rotation=45, ha="center", fontsize=8.0)
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.75, alpha=0.55)
            ax.tick_params(direction="out", width=0.8)
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_color("#333333")

    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        top=0.955,
        bottom=0.105,
        wspace=0.16,
        hspace=0.48,
    )
    for letter, ax in zip("ABCDEFGHI", axes.ravel()):
        position = ax.get_position()
        fig.text(
            position.x0 - 0.027,
            position.y1 + 0.006,
            letter,
            ha="left",
            va="bottom",
            fontsize=14,
            weight="bold",
        )
    return apply_manuscript_figure_style(fig)

def compute_tstr_runs(datasets, cohorts, repeats=3, seed=42):
    """Return repeat-level TSTR and TRTR F1 values for plotting and summaries."""
    rows = []
    for dataset, method_data in cohorts.items():
        X_real = np.asarray(datasets[dataset]["X"])
        y_real = np.asarray(datasets[dataset]["y"], dtype=int)
        for method, (X_syn, y_syn) in method_data.items():
            tstr, trtr = tstr_values(
                X_real,
                y_real,
                X_syn,
                y_syn,
                seed=seed,
                repeats=repeats,
            )
            rows.extend(
                {
                    "dataset": dataset,
                    "method": method,
                    "repeat": repeat,
                    "tstr_f1": float(tstr_value),
                    "trtr_f1": float(trtr_value),
                    "utility_gap_abs": float(abs(trtr_value - tstr_value)),
                }
                for repeat, (tstr_value, trtr_value) in enumerate(zip(tstr, trtr))
            )
    return pd.DataFrame(rows)

def _fixed_real_pca_payloads(X_real, method_data, methods, seed=42):
    """Project every method onto one PCA basis fitted to standardized real data.

    The returned real and synthetic percentages are variance fractions measured
    within each dataset along the fixed real-data loading vectors. Thus the
    coordinates remain directly comparable while the displayed percentages can
    differ by method.
    """
    X_real = np.asarray(X_real, dtype=np.float64)
    Xr, _ = standardize_pair(X_real, X_real)
    pca = PCA(n_components=2, random_state=seed).fit(Xr)
    Zr = pca.transform(Xr)

    def projected_ratios(X_standardized, coordinates):
        total_variance = float(
            np.sum(np.var(X_standardized, axis=0, ddof=1))
        )
        if not np.isfinite(total_variance) or total_variance <= 0.0:
            return np.full(coordinates.shape[1], np.nan, dtype=float)
        return np.var(coordinates, axis=0, ddof=1) / total_variance

    real_ratios = projected_ratios(Xr, Zr)
    payloads = {}
    for method in methods:
        X_syn = np.asarray(method_data[method][0], dtype=np.float64)
        _, Xs = standardize_pair(X_real, X_syn)
        Zs = pca.transform(Xs)
        synthetic_ratios = projected_ratios(Xs, Zs)
        payloads[method] = (Zr, Zs, real_ratios, synthetic_ratios)
    return payloads

def _fixed_pca_axis_label(component, synthetic_ratio):
    """Label a fixed real-data PC direction by synthetic projected variance."""
    return f"PC{component + 1} ({100 * synthetic_ratio:.1f}%)"

def compute_marginal_tests(datasets, cohorts):
    rows = []
    for dataset, method_data in cohorts.items():
        X_real = np.asarray(datasets[dataset]["X"], dtype=float)
        names = datasets[dataset].get(
            "feature_names", [f"feature_{i + 1}" for i in range(X_real.shape[1])]
        )
        for method, (X_syn, _) in method_data.items():
            ks_values = ks_by_feature(X_real, X_syn)
            for index, name in enumerate(names):
                test = ttest_ind(
                    X_real[:, index],
                    np.asarray(X_syn)[:, index],
                    equal_var=False,
                    nan_policy="omit",
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "feature": name,
                        "t_statistic": float(test.statistic),
                        "p_value": float(test.pvalue),
                        "ks_statistic": float(ks_values[index]),
                    }
                )
    return pd.DataFrame(rows)

def plot_marginal_distribution_grid(
    datasets,
    cohorts,
    marginal_tests,
    dataset="HIV",
    method_order=None,
    top_n=8,
    feature_start=0,
):
    """Plot one title-free supplement page of HIV marginal overlaps.

    Features are ranked by their largest KS statistic across methods. Each row
    uses one real-derived standardization and one pooled 1st--99th percentile
    range so the six synthetic-method panels are directly comparable. Use
    ``feature_start`` to paginate the complete ranked feature list.
    """
    method_order = [
        method
        for method in (method_order or METHOD_ORDER)
        if method in cohorts[dataset]
    ]
    if len(method_order) != 6:
        raise ValueError(
            "The HIV marginal-overlap supplement requires all six methods."
        )
    if top_n < 1:
        raise ValueError("top_n must be positive.")
    if feature_start < 0:
        raise ValueError("feature_start cannot be negative.")

    X_real = np.asarray(datasets[dataset]["X"], dtype=np.float64)
    feature_names = list(
        datasets[dataset].get(
            "feature_names",
            [f"feature_{index + 1}" for index in range(X_real.shape[1])],
        )
    )
    name_to_index = {str(name): index for index, name in enumerate(feature_names)}
    subset = marginal_tests[marginal_tests["dataset"] == dataset].copy()
    ranked_features = (
        subset.groupby("feature", sort=False)["ks_statistic"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()[feature_start:feature_start + int(top_n)]
    )
    if not ranked_features:
        raise ValueError(f"No marginal-test results are available for {dataset!r}.")

    # The reciprocal dimensions finish at true A4 portrait after the shared
    # The shared manuscript style applies its 1.18 figure-size multiplier.
    fig, axes = plt.subplots(
        len(ranked_features),
        len(method_order),
        figsize=(8.27 / 1.18, 11.69 / 1.18),
        squeeze=False,
        sharex="row",
        sharey="row",
    )

    for row, feature in enumerate(ranked_features):
        feature_index = name_to_index[str(feature)]
        real_values = X_real[:, feature_index]
        real_values = real_values[np.isfinite(real_values)]
        center = float(np.mean(real_values))
        scale = float(np.std(real_values, ddof=1))
        if not np.isfinite(scale) or np.isclose(scale, 0.0):
            scale = 1.0
        real_z = (real_values - center) / scale

        synthetic_z = {}
        for method in method_order:
            values = np.asarray(
                cohorts[dataset][method][0][:, feature_index], dtype=np.float64
            )
            values = values[np.isfinite(values)]
            synthetic_z[method] = (values - center) / scale

        pooled = np.concatenate([real_z, *synthetic_z.values()])
        x_low, x_high = np.quantile(pooled, [0.01, 0.99])
        x_low, x_high = float(x_low), float(x_high)
        if np.isclose(x_low, x_high):
            pad = max(0.5, abs(x_low) * 0.05)
            x_low, x_high = x_low - pad, x_high + pad
        visible = pooled[(pooled >= x_low) & (pooled <= x_high)]
        candidate_edges = np.histogram_bin_edges(visible, bins="fd")
        n_bins = int(np.clip(len(candidate_edges) - 1, 10, 22))
        edges = np.linspace(x_low, x_high, n_bins + 1)
        axis_pad = 0.07 * (x_high - x_low)

        for col, method in enumerate(method_order):
            ax = axes[row, col]
            method_color = METHOD_COLORS[method]
            ax.hist(
                real_z,
                bins=edges,
                density=True,
                histtype="stepfilled",
                color="#777777",
                alpha=0.34,
                linewidth=0,
            )
            ax.hist(
                synthetic_z[method],
                bins=edges,
                density=True,
                histtype="stepfilled",
                color=method_color,
                alpha=0.34,
                linewidth=0,
            )
            ax.hist(
                real_z,
                bins=edges,
                density=True,
                histtype="step",
                color="#333434",
                linewidth=0.85,
            )
            ax.hist(
                synthetic_z[method],
                bins=edges,
                density=True,
                histtype="step",
                color=method_color,
                linewidth=0.85,
            )
            ks_values = subset.loc[
                (subset["method"] == method) & (subset["feature"] == feature),
                "ks_statistic",
            ]
            if not ks_values.empty:
                ax.text(
                    0.97,
                    0.09,
                    f"KS {float(ks_values.iloc[0]):.2f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=4.8,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.80,
                        "pad": 0.7,
                    },
                )

            if row == 0:
                ax.set_title(method, fontsize=7.3, weight="semibold", pad=4.0)
            if col == 0:
                feature_label = str(feature).replace("_", " ")
                if len(feature_label) > 21:
                    feature_label = feature_label[:18] + "..."
                ax.set_ylabel(
                    feature_label,
                    fontsize=3.8,
                    weight="semibold",
                    labelpad=2.0,
                )
            else:
                ax.tick_params(axis="y", labelleft=False)
            if row == len(ranked_features) - 1:
                ax.set_xlabel("Standardized value", fontsize=5.7)
            else:
                ax.tick_params(axis="x", labelbottom=False)

            ax.set_xlim(x_low - axis_pad, x_high + axis_pad)
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.45)
            ax.tick_params(labelsize=4.8, width=0.6, length=2.0, pad=1.0)
            for spine in ax.spines.values():
                spine.set_linewidth(0.65)
                spine.set_color("#555555")

    fig.subplots_adjust(
        left=0.072,
        right=0.998,
        top=0.955,
        bottom=0.075,
        wspace=0.035,
        hspace=0.12,
    )
    return apply_manuscript_figure_style(fig)

def compute_noise_sensitivity(
    datasets,
    cohorts,
    sigmas=(0.0, 0.2, 0.5, 1.0),
    repeats=3,
    seed=42,
):
    rows = []
    for dataset, method_data in cohorts.items():
        X_real = np.asarray(datasets[dataset]["X"], dtype=float)
        y_real = np.asarray(datasets[dataset]["y"], dtype=int)
        scale = np.where(X_real.std(axis=0) == 0, 1.0, X_real.std(axis=0))
        for method, (base_syn, y_syn) in method_data.items():
            for sigma in sigmas:
                values = []
                for repeat in range(int(repeats)):
                    rng = np.random.default_rng(seed + 1009 * repeat)
                    X_syn = np.asarray(base_syn, dtype=float)
                    if sigma:
                        X_syn = X_syn + rng.normal(size=X_syn.shape) * scale * sigma
                    values.append(
                        one_run_origin_auc(
                            X_real, y_real, X_syn, y_syn, seed + 101 * repeat
                        )
                    )
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "sigma": float(sigma),
                        "sep_mean": float(np.mean(values)),
                        "sep_sd": float(np.std(values)),
                    }
                )
    return pd.DataFrame(rows)

def plot_noise_sensitivity_summary(
    noise_table,
    dataset="HIV",
    method_order=None,
):
    """Render the HIV noise-sensitivity analysis as an A4 supplement page."""
    method_order = [
        method
        for method in (method_order or METHOD_ORDER)
        if method in set(noise_table["method"])
    ]
    if len(method_order) != 6:
        raise ValueError(
            "The HIV noise-sensitivity supplement requires all six methods."
        )

    # The reciprocal dimensions finish at true A4 landscape after the shared
    # The shared manuscript style applies its 1.18 figure-size multiplier.
    fig, ax = plt.subplots(figsize=(11.69 / 1.18, 8.27 / 1.18))
    display_names = {"GMM-guided SMOTE": "GMM-SMOTE"}
    for method in method_order:
        values = noise_table.query(
            "dataset == @dataset and method == @method"
        ).sort_values("sigma")
        if values.empty:
            continue
        color = METHOD_COLORS[method]
        ax.plot(
            values["sigma"],
            values["sep_mean"],
            color=color,
            marker="o",
            markersize=4.2,
            linewidth=1.8,
            label=display_names.get(method, method),
        )
        if "sep_sd" in values:
            lower = np.clip(values["sep_mean"] - values["sep_sd"], 0.0, 1.0)
            upper = np.clip(values["sep_mean"] + values["sep_sd"], 0.0, 1.0)
            ax.fill_between(
                values["sigma"],
                lower,
                upper,
                color=color,
                alpha=0.14,
                linewidth=0,
            )

    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.15)
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel(r"Noise level $\sigma$", fontsize=11.0)
    ax.set_ylabel(r"$\langle\mathrm{AUC}\rangle$", fontsize=11.0)
    ax.tick_params(axis="both", labelsize=9.0, direction="out")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.75, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="center right",
        frameon=False,
        fontsize=8.2,
        ncol=2,
        columnspacing=1.0,
        handlelength=2.0,
    )
    fig.subplots_adjust(left=0.10, right=0.985, top=0.965, bottom=0.10)
    return apply_manuscript_figure_style(fig)

def permute_class_conditional_dependence(X, y, proportion, seed=42):
    """Partially break cross-feature alignment without changing marginals.

    Within each outcome class and feature, ``proportion`` of row positions are
    selected and the values occupying those positions are randomly permuted.
    Consequently, every class-conditional feature contains exactly the same
    multiset of values before and after perturbation.
    """
    if not 0.0 <= float(proportion) <= 1.0:
        raise ValueError("proportion must be between 0 and 1")

    X = np.asarray(X)
    y = np.asarray(y)
    perturbed = X.copy()
    if float(proportion) == 0.0:
        return perturbed

    rng = np.random.default_rng(seed)
    for outcome in np.unique(y):
        class_indices = np.flatnonzero(y == outcome)
        n_selected = int(np.rint(float(proportion) * len(class_indices)))
        if n_selected < 2:
            continue
        for feature_index in range(X.shape[1]):
            selected = rng.choice(class_indices, size=n_selected, replace=False)
            perturbed[selected, feature_index] = perturbed[
                rng.permutation(selected), feature_index
            ]
    return perturbed

def paired_origin_auc(X_real, X_perturbed, y, seed=42):
    """Origin AUC with matched row pairs kept in the same data split."""
    X_real = np.asarray(X_real, dtype=np.float64)
    X_perturbed = np.asarray(X_perturbed, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    if X_real.shape != X_perturbed.shape or len(y) != len(X_real):
        raise ValueError("real, perturbed, and outcome arrays must align by row")

    row_indices = np.arange(len(y))
    train_indices, test_indices = train_test_split(
        row_indices,
        test_size=0.25,
        stratify=y,
        random_state=seed,
    )
    Xr, Xp = standardize_pair(X_real, X_perturbed)
    X_train = np.vstack((Xr[train_indices], Xp[train_indices]))
    origin_train = np.r_[
        np.zeros(len(train_indices), dtype=int),
        np.ones(len(train_indices), dtype=int),
    ]
    X_test = np.vstack((Xr[test_indices], Xp[test_indices]))
    origin_test = np.r_[
        np.zeros(len(test_indices), dtype=int),
        np.ones(len(test_indices), dtype=int),
    ]
    discriminator = RandomForestClassifier(
        n_estimators=500,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )
    discriminator.fit(X_train, origin_train)
    auc = roc_auc_score(
        origin_test, discriminator.predict_proba(X_test)[:, 1]
    )
    return float(max(auc, 1.0 - auc))

def compute_dependence_permutation_sensitivity(
    datasets,
    proportions=tuple(np.linspace(0.0, 1.0, 11)),
    repeats=10,
    seed=42,
    alphas=None,
    edge_threshold=1e-7,
):
    """Measure separability and edge preservation after marginal-safe shuffling."""
    proportions = tuple(float(value) for value in proportions)
    if not proportions or any(value < 0.0 or value > 1.0 for value in proportions):
        raise ValueError("proportions must contain values between 0 and 1")
    if int(repeats) < 2:
        raise ValueError("repeats must be at least 2 to estimate uncertainty")

    selected_alphas = dict(GRAPHICAL_LASSO_ALPHAS)
    if alphas is not None:
        selected_alphas.update(alphas)

    rows = []
    for dataset_index, (dataset, data) in enumerate(datasets.items()):
        X_real = np.asarray(data["X"], dtype=np.float64)
        y_real = np.asarray(data["y"], dtype=int)
        alpha = float(selected_alphas[dataset])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            real_precision = fit_glasso_precision(X_real, alpha)
        real_partial = precision_to_partial_corr(real_precision)
        real_edges = get_edge_set(real_partial, threshold=edge_threshold)

        for level_index, proportion in enumerate(proportions):
            for repeat in range(int(repeats)):
                run_seed = (
                    int(seed)
                    + 100_003 * dataset_index
                    + 10_007 * level_index
                    + 1_009 * repeat
                )
                X_perturbed = permute_class_conditional_dependence(
                    X_real, y_real, proportion, seed=run_seed
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    perturbed_precision = fit_glasso_precision(X_perturbed, alpha)
                perturbed_partial = precision_to_partial_corr(perturbed_precision)
                perturbed_edges = get_edge_set(
                    perturbed_partial, threshold=edge_threshold
                )
                edge_union = real_edges | perturbed_edges
                edge_jaccard = (
                    len(real_edges & perturbed_edges) / len(edge_union)
                    if edge_union
                    else 1.0
                )
                edge_recovery = compute_edge_recovery(
                    real_edges, perturbed_edges
                )
                discriminator_auc = paired_origin_auc(
                    X_real, X_perturbed, y_real, seed=run_seed + 503
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "proportion_permuted": proportion,
                        "percent_permuted": 100.0 * proportion,
                        "repeat": repeat,
                        "discriminator_auc": discriminator_auc,
                        "edge_jaccard": float(edge_jaccard),
                        "edge_recovery": float(edge_recovery),
                        "n_real_edges": len(real_edges),
                        "n_perturbed_edges": len(perturbed_edges),
                        "glasso_alpha": alpha,
                    }
                )
    return pd.DataFrame(rows)

def plot_dependence_permutation_sensitivity(table, dataset_order=None):
    """Overlay the AUC trend on the area under the edge-recovery curve."""
    dataset_order = list(dataset_order or dict.fromkeys(table["dataset"]))
    perturbation_levels = sorted(table["percent_permuted"].unique())
    positions = np.asarray(perturbation_levels, dtype=float)

    # Both metrics retain their original 0--1 values. The recovery area shows
    # remaining structure, while the AUC line shows distinguishability.
    fig, axes = plt.subplots(
        len(dataset_order),
        1,
        figsize=(8.0 / 1.18, 9.2 / 1.18),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row, dataset in enumerate(dataset_order):
        ax = axes[row, 0]
        subset = table.loc[table["dataset"].eq(dataset)]
        auc_summary = (
            subset.groupby("percent_permuted", sort=False)["discriminator_auc"]
            .agg(["mean", "std"])
            .reindex(perturbation_levels)
        )
        auc_mean = auc_summary["mean"].to_numpy(dtype=float)
        auc_sd = auc_summary["std"].fillna(0.0).to_numpy(dtype=float)
        recovery_summary = (
            subset.groupby("percent_permuted", sort=False)["edge_recovery"]
            .agg(["mean", "std"])
            .reindex(perturbation_levels)
        )
        recovery_mean = recovery_summary["mean"].to_numpy(dtype=float)
        color = DATASET_COLORS[dataset]

        ax.fill_between(
            positions,
            0.0,
            recovery_mean,
            color=color,
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            positions,
            recovery_mean,
            color=color,
            linestyle="-",
            linewidth=1.15,
            alpha=0.72,
            zorder=3,
        )
        ax.errorbar(
            positions,
            auc_mean,
            yerr=auc_sd,
            color=color,
            marker="o",
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.55,
            markersize=4.6,
            linewidth=2.0,
            capsize=2.4,
            elinewidth=0.9,
            solid_capstyle="round",
            solid_joinstyle="round",
            clip_on=False,
            zorder=4,
        )

        ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.9, zorder=0)
        ax.set_xlim(0.0, float(max(perturbation_levels)))
        ax.set_ylim(0.0, 1.02)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{level:g}" for level in perturbation_levels])
        ax.set_title(dataset, color="#222222", fontsize=11.5, weight="bold", pad=7)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.65, alpha=0.52)
        ax.grid(axis="x", color="#E4E4E4", linewidth=0.65, alpha=0.62)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#666666")
        ax.spines["bottom"].set_color("#666666")
        ax.spines["left"].set_linewidth(0.95)
        ax.spines["bottom"].set_linewidth(0.95)
        ax.tick_params(axis="both", width=0.8, length=3.0)
        if row == 1:
            ax.set_ylabel("Metric value", fontsize=10.5, weight="semibold")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#666666",
            marker="o",
            markerfacecolor="#666666",
            markeredgecolor="#666666",
            linewidth=1.75,
            markersize=4.4,
            label="AUC",
        ),
        Patch(
            facecolor="#888888",
            edgecolor="#888888",
            alpha=0.20,
            label="Edge recovery",
        ),
    ]
    axes[0, 0].legend(
        handles=legend_handles,
        frameon=False,
        fontsize=8.0,
        loc="lower left",
        handlelength=1.7,
        handletextpad=0.5,
    )
    for letter, ax in zip("ABC", axes[:, 0]):
        ax.text(
            -0.10,
            1.05,
            letter,
            transform=ax.transAxes,
            fontsize=13,
            weight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
        )
    fig.supxlabel(
        "Within-class permutation (%)",
        fontsize=10.5,
        weight="semibold",
        y=0.025,
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.96,
        bottom=0.09,
        hspace=0.28,
    )
    return apply_manuscript_figure_style(fig)

def compute_reverse_ablation(datasets, cohorts, repeats=3, seed=42):
    rows = []
    for dataset, method_data in cohorts.items():
        X_real = np.asarray(datasets[dataset]["X"])
        y_real = np.asarray(datasets[dataset]["y"], dtype=int)
        for method, (X_syn, y_syn) in method_data.items():
            ranking = rank_discriminating_features(X_real, X_syn, seed=seed)
            for removed in ablation_grid(X_real.shape[1]):
                keep = ranking[int(removed):]
                values = [
                    one_run_origin_auc(
                        X_real[:, keep],
                        y_real,
                        np.asarray(X_syn)[:, keep],
                        y_syn,
                        seed=seed + 1009 * repeat,
                    )
                    for repeat in range(int(repeats))
                ]
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "n_features_removed": int(removed),
                        "percent_removed": 100 * removed / X_real.shape[1],
                        "auc_mean": float(np.mean(values)),
                        "auc_sd": float(np.std(values)),
                    }
                )
    return pd.DataFrame(rows)

def plot_graphical_lasso_ablation_pca(
    datasets,
    cohorts,
    ablation_table,
    edge_status,
    dataset="Breast Cancer",
):
    """Compose edge-status matrices, reverse ablation, and fixed-basis PCA."""
    methods = [method for method in METHOD_ORDER if method in cohorts[dataset]]
    structures = edge_status.structures[dataset]
    fig = plt.figure(figsize=(8.27 / 1.18, 10.15 / 1.18), facecolor="white")
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.35, 0.86],
        left=0.055,
        right=0.995,
        top=0.965,
        bottom=0.075,
        hspace=0.25,
    )

    # A: the structural result receives the full figure width.
    matrix_grid = outer[0].subgridspec(2, 3, wspace=0.08, hspace=0.17)
    real = structures["real"]
    order = get_real_structure_order(real["partial"])
    n_features = real["partial"].shape[0]
    ticks = np.arange(0, n_features, 10 if n_features <= 70 else 20)
    tick_labels = [str(value + 1) for value in ticks]
    status_cmap = ListedColormap([
        STATUS_COLORS["absent"],
        STATUS_COLORS["preserved"],
        STATUS_COLORS["real_only"],
        STATUS_COLORS["synthetic_only"],
    ])
    matrix_axes = []
    for index, method in enumerate(methods):
        row, col = divmod(index, 3)
        ax = fig.add_subplot(matrix_grid[row, col])
        matrix_axes.append(ax)
        syn_edges = structures["synthetic"][method]["edges"]
        status = build_edge_status_matrix(real["edges"], syn_edges, n_features)
        ax.imshow(
            status[np.ix_(order, order)],
            cmap=status_cmap,
            vmin=-0.5,
            vmax=3.5,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(
            "GMM-SMOTE" if method == "GMM-guided SMOTE" else method,
            color=METHOD_COLORS[method],
            fontsize=9,
            weight="bold",
            pad=2.5,
        )
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels if row == 1 else [], fontsize=5.0)
        ax.set_yticklabels(tick_labels if col == 0 else [], fontsize=5.0)
        ax.tick_params(axis="both", length=1.7, width=0.6, pad=0.8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.65)
            spine.set_color("#444444")
    matrix_axes[0].text(
        -0.13, 1.08, "A", transform=matrix_axes[0].transAxes,
        ha="left", va="bottom", fontsize=11.0, weight="bold", clip_on=False,
    )
    status_handles = [
        Patch(facecolor=STATUS_COLORS["preserved"], edgecolor="#333333", label="Preserved"),
        Patch(facecolor=STATUS_COLORS["real_only"], edgecolor="#333333", label="Real-only"),
        Patch(facecolor=STATUS_COLORS["synthetic_only"], edgecolor="#333333", label="Synthetic-only"),
        Patch(facecolor=STATUS_COLORS["absent"], edgecolor="#C9CDD2", label="Absent"),
    ]
    fig.legend(
        handles=status_handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.42),
        ncol=4,
        frameon=False,
        fontsize=7,
        handlelength=1.25,
        handletextpad=0.35,
        columnspacing=0.9,
    )

    lower = outer[1].subgridspec(1, 2, width_ratios=[0.38, 0.62], wspace=0.27)

    # B: reverse ablation.
    ablation_ax = fig.add_subplot(lower[0, 0])
    for method in methods:
        values = ablation_table.query(
            "dataset == @dataset and method == @method"
        ).sort_values("n_features_removed")
        color = METHOD_COLORS[method]
        ablation_ax.plot(
            values["n_features_removed"], values["auc_mean"],
            color=color, marker="o", markersize=2.3, linewidth=1.05,
            label="GMM-SMOTE" if method == "GMM-guided SMOTE" else method,
        )
        if "auc_sd" in values:
            lower_auc = np.clip(values["auc_mean"] - values["auc_sd"], 0.0, 1.0)
            upper_auc = np.clip(values["auc_mean"] + values["auc_sd"], 0.0, 1.0)
            ablation_ax.fill_between(
                values["n_features_removed"], lower_auc, upper_auc,
                color=color, alpha=0.14, linewidth=0,
            )
    ablation_ax.axhline(0.5, color="#777777", linestyle="--", linewidth=0.85)
    feature_ticks = values["n_features_removed"].to_numpy(dtype=int)
    ablation_ax.set_xticks(feature_ticks)
    ablation_ax.set_xlabel("Features removed", fontsize=7.0)
    ablation_ax.set_ylabel("AUC", fontsize=7.0)
    ablation_ax.tick_params(axis="both", labelsize=6.2)
    ablation_ax.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.65)
    ablation_ax.legend(
        loc="best", frameon=False, fontsize=5.4, ncol=2,
        columnspacing=0.7, handlelength=1.4,
    )
    ablation_ax.spines["top"].set_visible(False)
    ablation_ax.spines["right"].set_visible(False)
    ablation_ax.text(
        -0.15, 1.035, "B", transform=ablation_ax.transAxes,
        ha="left", va="bottom", fontsize=11.0, weight="bold", clip_on=False,
    )

    # C: compact 2 x 3 PCA comparison.
    pca_grid = lower[0, 1].subgridspec(2, 3, wspace=0.22, hspace=0.52)
    X_real = np.asarray(datasets[dataset]["X"])
    pca_payloads = _fixed_real_pca_payloads(
        X_real, cohorts[dataset], methods, seed=42
    )
    all_pca_coordinates = []
    for method in methods:
        Zr, Zs, _, _ = pca_payloads[method]
        all_pca_coordinates.extend((Zr, Zs))

    combined_pca = np.vstack(all_pca_coordinates)
    pca_x_min, pca_y_min = np.nanmin(combined_pca, axis=0)
    pca_x_max, pca_y_max = np.nanmax(combined_pca, axis=0)
    pca_x_pad = max((pca_x_max - pca_x_min) * 0.08, 0.5)
    pca_y_pad = max((pca_y_max - pca_y_min) * 0.08, 0.5)
    pca_x_limits = (pca_x_min - pca_x_pad, pca_x_max + pca_x_pad)
    pca_y_limits = (pca_y_min - pca_y_pad, pca_y_max + pca_y_pad)

    pca_axes = []
    for index, method in enumerate(methods):
        row, col = divmod(index, 3)
        ax = fig.add_subplot(pca_grid[row, col])
        pca_axes.append(ax)
        Zr, Zs, real_ratio, synthetic_ratio = pca_payloads[method]
        ax.scatter(
            Zr[:, 0], Zr[:, 1], s=3.2, facecolors="none",
            edgecolors="#777777", linewidths=0.4, alpha=0.45, label="Real",
        )
        ax.scatter(
            Zs[:, 0], Zs[:, 1], s=3.2, color=METHOD_COLORS[method],
            edgecolors="none", alpha=0.58, label=method,
        )
        add_confidence_ellipse(ax, Zr, "#777777", linewidth=0.8)
        add_confidence_ellipse(ax, Zs, METHOD_COLORS[method], linewidth=0.95)
        ax.set_title(
            "GMM-SMOTE" if method == "GMM-guided SMOTE" else method,
            color=METHOD_COLORS[method], fontsize=7, weight="bold", pad=1.5,
        )
        ax.set_xlabel(
            _fixed_pca_axis_label(0, synthetic_ratio[0]),
            fontsize=5.6, labelpad=0.6,
        )
        ax.set_ylabel(
            _fixed_pca_axis_label(1, synthetic_ratio[1])
            if col == 0 else "",
            fontsize=5.6, labelpad=0.6,
        )
        ax.set_xlim(*pca_x_limits)
        ax.set_ylim(*pca_y_limits)
        ax.tick_params(axis="both", labelsize=3.9, length=1.4, pad=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.legend(
            loc="upper left", frameon=True, fontsize=5, facecolor="white",
            markerscale=2, handletextpad=0.2, borderaxespad=0.15, ncol=2, columnspacing=0.8,  framealpha=0.9,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
            spine.set_color("#444444")
    pca_axes[0].text(
        -0.16, 1.12, "C", transform=pca_axes[0].transAxes,
        ha="left", va="bottom", fontsize=11.0, weight="bold", clip_on=False,
    )
    return apply_manuscript_figure_style(fig)

def plot_graphical_lasso_regularization_path(edge_status):
    """Plot every HIV Graphical Lasso edge path for the supplement."""
    path_table = edge_status.regularization_path
    if path_table is None or path_table.empty:
        raise ValueError("edge_status must contain a regularization path.")
    fig = plt.figure(figsize=(8.27 / 1.18, 5.05 / 1.18), facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=[3.25, 1.15], hspace=0.08)
    ax = fig.add_subplot(grid[0])
    survival_ax = fig.add_subplot(grid[1], sharex=ax)
    edge_columns = ["feature_a_matrix_index", "feature_b_matrix_index"]
    grouped_paths = list(path_table.groupby(edge_columns, sort=False))
    selected_alpha = float(path_table["selected_alpha"].iloc[0])
    if "selected_nonzero" not in path_table.columns:
        selected_flags = {}
        for edge_key, values in grouped_paths:
            nearest = values.iloc[(values["alpha"] - selected_alpha).abs().argmin()]
            selected_flags[edge_key] = abs(float(nearest["precision_coefficient"])) > 1e-7
    else:
        selected_flags = {
            edge_key: bool(values["selected_nonzero"].iloc[0])
            for edge_key, values in grouped_paths
        }

    total_edges = len(grouped_paths)
    retained_edges = sum(selected_flags.values())
    removed_edges = total_edges - retained_edges
    retained_pct = 100.0 * retained_edges / total_edges
    removed_pct = 100.0 * removed_edges / total_edges

    # Draw retained paths first, then place the much more numerous paths that
    # are zero at the selected lambda on top.  A warm retained-path color keeps
    # this panel distinct from the blue retained-area encoding below.
    for selected_state in (True, False):
        for edge_key, values in grouped_paths:
            if selected_flags[edge_key] != selected_state:
                continue
            values = values.sort_values("alpha")
            ax.plot(
                np.log(values["alpha"].to_numpy(dtype=float)),
                values["precision_coefficient"],
                color="#C46A2D" if selected_state else "#6F7782",
                linewidth=0.72 if selected_state else 0.48,
                alpha=0.48 if selected_state else 0.28,
                zorder=1 if selected_state else 2,
            )
    ax.axvline(
        np.log(selected_alpha),
        color="#222222",
        linestyle="--",
        linewidth=1.2,
        zorder=3,
    )
    ax.axhline(0, color="#777777", linewidth=0.7, alpha=0.75)
    ax.set_ylabel("Precision coefficient", fontsize=8.0)
    ax.tick_params(axis="both", labelsize=6.8)
    ax.tick_params(axis="x", labelbottom=False)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.38)

    # Show the complete sparsification process directly rather than relying on
    # the overlapping coefficient paths to communicate how many edges survive.
    remaining_by_alpha = {
        float(alpha): int(np.count_nonzero(values.to_numpy(dtype=float) > 1e-7))
        for alpha, values in path_table.assign(
            absolute_coefficient=path_table["precision_coefficient"].abs()
        ).groupby("alpha")["absolute_coefficient"]
    }
    remaining_by_alpha[selected_alpha] = retained_edges
    survival_alphas = np.array(sorted(remaining_by_alpha), dtype=float)
    survival_pct = np.array(
        [100.0 * remaining_by_alpha[alpha] / total_edges for alpha in survival_alphas],
        dtype=float,
    )
    log_survival_alphas = np.log(survival_alphas)
    selected_log_alpha = np.log(selected_alpha)
    # The gray envelope is the complete candidate-edge set (100%); blue is the
    # portion still nonzero.  The gray area left above the curve therefore
    # directly shows the edges removed by regularization.
    survival_ax.fill_between(
        log_survival_alphas, 0, 100,
        color="#D9DEE3", alpha=0.78, linewidth=0, zorder=0,
    )
    survival_ax.fill_between(
        log_survival_alphas, 0, survival_pct,
        color="#C46A2D", alpha=0.78, linewidth=0, zorder=1,
    )
    survival_ax.plot(
        log_survival_alphas, survival_pct,
        color="#C46A2D", linewidth=1.15, zorder=2,
    )
    survival_ax.axvline(
        selected_log_alpha, color="#222222", linestyle="--", linewidth=1.2,
    )
    survival_ax.scatter(
        [selected_log_alpha], [retained_pct], s=24,
        color="#C46A2D", edgecolor="white", linewidth=0.7, zorder=4,
    )
    survival_ax.annotate(
        f"{retained_edges:,} retained ({retained_pct:.1f}%)\n"
        f"{removed_edges:,} removed ({removed_pct:.1f}%)",
        xy=(selected_log_alpha, retained_pct),
        xytext=(selected_log_alpha + 0.22, 58),
        ha="left", va="center", fontsize=6.2, color="#333333",
        bbox=dict(facecolor="white", edgecolor="#D6DADF", alpha=0.94, pad=2.2),
        arrowprops=dict(arrowstyle="-", color="#C46A2D", linewidth=0.8),
    )
    survival_ax.set_ylim(0, 100)
    survival_ax.set_yticks([0, 25, 50, 75, 100])
    survival_ax.set_ylabel("Edges retained (%)", fontsize=7.1)
    survival_ax.set_xlabel(r"$\log(\lambda)$", fontsize=8.0)
    survival_ax.tick_params(axis="both", labelsize=6.6)
    survival_ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.36)

    survival_ax.spines["top"].set_visible(False)
    survival_ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#6F7782", linewidth=1.1, alpha=0.85, label=r"Zero at selected $\lambda$"),
            Line2D([0], [0], color="#C46A2D", linewidth=1.7, label=r"Non-zero at selected $\lambda$"),
            Line2D([0], [0], color="#222222", linewidth=1.2, linestyle="--", label=rf"Selected $\lambda={selected_alpha:g}$"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.60),
        frameon=False,
        fontsize=6.2,
        ncol=3,
        handlelength=1.8,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.965, bottom=0.22)
    return apply_manuscript_figure_style(fig)
