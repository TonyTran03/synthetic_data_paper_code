from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn import covariance
from sklearn.preprocessing import StandardScaler


EDGE_COLORS = {
    "preserved": "#2F6DB3",
    "real_only": "#C43C39",
    "synthetic_only": "#E88925",
}

METRIC_LABELS = {
    "frobenius_deviation": r"Frobenius deviation, $||\Theta_R-\Theta_S||_F$",
    "edge_recovery": r"Edge recovery, $|E_R \cap E_S| / |E_R|$",
    "synthetic_only_rate": r"Synthetic-only edge rate, $|E_S \setminus E_R| / |E_S|$",
}

METHOD_PRESERVATION_COLORS = {
    "Bootstrap": "#6A5ACD",
    "Column-wise": "#CC79A7",
    "SMOTE": "#0072B2",
    "GMM": "#009E73",
    "GMM-guided SMOTE": "#56B4E9",
    "CVAE": "#D55E00",
}

METHOD_PRESERVATION_PASTELS = {
    "Bootstrap": "#C7C2F4",
    "Column-wise": "#E8B4D2",
    "SMOTE": "#9ECAE1",
    "GMM": "#A8DEC9",
    "GMM-guided SMOTE": "#B9E2F5",
    "CVAE": "#F2B49B",
}

@dataclass
class EdgeStatusResult:
    fig: plt.Figure
    metrics: pd.DataFrame
    anchor: int
    anchor_feature: str
    structures: dict
    edge_recovery: pd.DataFrame | None = None
    feature_index: pd.DataFrame | None = None
    regularization_path: pd.DataFrame | None = None

def _prepare_glasso_input(X):
    X = np.asarray(X, dtype=np.float64)
    X = np.where(np.isfinite(X), X, np.nan)
    col_medians = np.nanmedian(X, axis=0)
    col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
    missing = np.where(~np.isfinite(X))
    if len(missing[0]):
        X = X.copy()
        X[missing] = np.take(col_medians, missing[1])
    Xs = StandardScaler().fit_transform(X)
    return np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

def fit_glasso_precision(X, alpha):
    """Fit Graphical Lasso on standardized features and return a precision matrix."""
    Xs = _prepare_glasso_input(X)
    if alpha is None:
        cv = min(5, max(2, Xs.shape[0] // 3))
        try:
            model = covariance.GraphicalLassoCV(
                alphas=8, cv=cv, max_iter=1000, n_refinements=3
            ).fit(Xs)
            alpha = float(model.alpha_)
        except Exception:
            alpha = 0.08
    try:
        model = covariance.GraphicalLasso(alpha=float(alpha), max_iter=1000).fit(Xs)
        return np.asarray(model.precision_, dtype=float)
    except Exception:
        empirical = covariance.EmpiricalCovariance().fit(Xs)
        return np.linalg.pinv(empirical.covariance_)

def _estimate_glasso_alpha(X, cv_folds=5, max_iter=1000):
    Xs = _prepare_glasso_input(X)
    cv = min(cv_folds, max(2, Xs.shape[0] // 3))
    model = covariance.GraphicalLassoCV(
        alphas=8, cv=cv, max_iter=max_iter, n_refinements=3
    ).fit(Xs)
    return float(model.alpha_)

def precision_to_partial_corr(theta):
    """Convert a precision matrix to a partial-correlation matrix."""
    theta = np.asarray(theta, dtype=np.float64)
    diag = np.clip(np.diag(theta), 1e-12, None)
    denom = np.sqrt(np.outer(diag, diag))
    partial = -theta / denom
    np.fill_diagonal(partial, 0.0)
    partial[np.abs(partial) < 1e-12] = 0.0
    return np.clip(partial, -1.0, 1.0)

def get_edge_set(partial_corr, threshold=1e-7):
    """Return undirected off-diagonal edges with absolute partial correlation above threshold."""
    partial_corr = np.asarray(partial_corr, dtype=float)
    edges = set()
    for i in range(partial_corr.shape[0]):
        for j in range(i + 1, partial_corr.shape[1]):
            if abs(float(partial_corr[i, j])) > threshold:
                edges.add((i, j))
    return edges

def compute_edge_recovery(real_edges, synthetic_edges):
    return float(len(real_edges & synthetic_edges) / len(real_edges)) if real_edges else np.nan

def compute_synthetic_only_rate(real_edges, synthetic_edges):
    return float(len(synthetic_edges - real_edges) / len(synthetic_edges)) if synthetic_edges else np.nan

def compute_frobenius_deviation(theta_real, theta_syn):
    theta_real = np.asarray(theta_real, dtype=float).copy()
    theta_syn = np.asarray(theta_syn, dtype=float).copy()
    np.fill_diagonal(theta_real, 0.0)
    np.fill_diagonal(theta_syn, 0.0)
    return float(np.linalg.norm(theta_real - theta_syn, ord="fro"))

def _fit_structures(real_data, synthetic_data, alphas=None, threshold=1e-7, dataset_order=None, method_order=None):
    dataset_order = list(dataset_order or real_data.keys())
    method_order = list(method_order or synthetic_data[dataset_order[0]].keys())
    structures = {}
    rows = []

    for dataset in dataset_order:
        alpha = None if alphas is None else alphas.get(dataset)
        if alpha is None:
            alpha = _estimate_glasso_alpha(real_data[dataset])
        theta_real = fit_glasso_precision(real_data[dataset], alpha)
        real_partial = precision_to_partial_corr(theta_real)
        real_edges = get_edge_set(real_partial, threshold)
        structures[dataset] = {
            "real": {"theta": theta_real, "partial": real_partial, "edges": real_edges},
            "synthetic": {},
            "alpha": alpha,
        }
        for method in method_order:
            theta_syn = fit_glasso_precision(synthetic_data[dataset][method], alpha)
            syn_partial = precision_to_partial_corr(theta_syn)
            syn_edges = get_edge_set(syn_partial, threshold)
            structures[dataset]["synthetic"][method] = {
                "theta": theta_syn, "partial": syn_partial, "edges": syn_edges,
            }
            rows.append({
                "dataset": dataset,
                "method": method,
                "frobenius_deviation": compute_frobenius_deviation(theta_real, theta_syn),
                "edge_recovery": compute_edge_recovery(real_edges, syn_edges),
                "synthetic_only_rate": compute_synthetic_only_rate(real_edges, syn_edges),
                "n_real_edges": len(real_edges),
                "n_synthetic_edges": len(syn_edges),
            })

    return structures, pd.DataFrame(rows)

STATUS_CODES = {
    "absent": 0,
    "preserved": 1,
    "real_only": 2,
    "synthetic_only": 3,
}

STATUS_COLORS = {
    "absent": "#F3F5F7",
    "preserved": "#1F5A93",
    "real_only": "#C83F3F",
    "synthetic_only": "#E98A2A",
}

def get_real_structure_order(real_partial):
    """Order features by hierarchical clustering on the real partial-correlation structure."""
    structure = np.abs(np.asarray(real_partial, dtype=float))
    np.fill_diagonal(structure, 1.0)
    distance = 1.0 - np.clip(structure, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    if np.allclose(condensed, condensed[0] if len(condensed) else 0.0):
        return np.arange(structure.shape[0])
    linkage = hierarchy.linkage(condensed, method="average")
    return hierarchy.leaves_list(linkage)

def build_edge_status_matrix(real_edges, synthetic_edges, n_features):
    """Build a symmetric categorical matrix comparing real and synthetic edge sets."""
    status = np.zeros((n_features, n_features), dtype=int)
    all_edges = real_edges | synthetic_edges
    for edge in all_edges:
        i, j = edge
        if edge in real_edges and edge in synthetic_edges:
            code = STATUS_CODES["preserved"]
        elif edge in real_edges:
            code = STATUS_CODES["real_only"]
        else:
            code = STATUS_CODES["synthetic_only"]
        status[i, j] = code
        status[j, i] = code
    np.fill_diagonal(status, STATUS_CODES["absent"])
    return status

def make_feature_index_table(feature_names, order):
    return pd.DataFrame({
        "matrix_index": np.arange(1, len(order) + 1),
        "feature_original_index": np.asarray(order, dtype=int) + 1,
        "feature_name": [feature_names[i] for i in order],
    })

def plot_edge_status_matrices(
    real_data,
    synthetic_data,
    feature_names,
    alphas=None,
    dataset_name="HIV",
    dataset_order=None,
    method_order=None,
    comparison_methods=None,
    threshold=1e-7,
    save_path=None,
):
    """Combine a 2 x 3 HIV edge-status grid with a full-width solution path."""
    dataset_order = list(dataset_order or real_data.keys())
    if dataset_name not in dataset_order:
        raise ValueError(f"{dataset_name!r} is not present in dataset_order.")
    method_order = list(method_order or synthetic_data[dataset_name].keys())
    comparison_methods = list(comparison_methods or method_order)
    preferred_order = [
        "Bootstrap",
        "Column-wise",
        "SMOTE",
        "GMM",
        "GMM-guided SMOTE",
        "CVAE",
    ]
    comparison_methods = [
        method for method in preferred_order
        if method in comparison_methods and method in method_order
    ]
    if len(comparison_methods) != 6:
        raise ValueError(
            "The main structural mosaic requires Bootstrap, Column-wise, GMM, "
            "SMOTE, GMM-guided SMOTE, and CVAE."
        )

    structures, metrics = _fit_structures(
        real_data,
        synthetic_data,
        alphas=alphas,
        threshold=threshold,
        dataset_order=dataset_order,
        method_order=method_order,
    )

    # Reciprocal dimensions finish at 8.27 x 9.7 inches after the shared
    # The shared manuscript style applies its 1.18 size multiplier.
    fig = plt.figure(figsize=(8.27 / 1.18, 9.7 / 1.18), facecolor="white")
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.0, 1.0, 0.78],
        left=0.085,
        right=0.985,
        top=0.965,
        bottom=0.155,
        wspace=0.13,
        hspace=0.30,
    )

    cmap = ListedColormap([
        STATUS_COLORS["absent"],
        STATUS_COLORS["preserved"],
        STATUS_COLORS["real_only"],
        STATUS_COLORS["synthetic_only"],
    ])
    real = structures[dataset_name]["real"]
    real_edges = real["edges"]
    order = get_real_structure_order(real["partial"])
    names = list(
        feature_names[dataset_name]
        if isinstance(feature_names, Mapping)
        else feature_names
    )
    n_features = real["partial"].shape[0]
    tick_step = 1 if n_features <= 12 else 5 if n_features <= 35 else 10
    ticks = np.arange(0, n_features, tick_step)
    tick_labels = [str(index + 1) for index in ticks]
    panel_letters = list("ABCDEFG")
    display_names = {"GMM-guided SMOTE": "GMM-SMOTE"}

    for index, method in enumerate(comparison_methods):
        row, col = divmod(index, 3)
        ax = fig.add_subplot(grid[row, col])
        synthetic_edges = structures[dataset_name]["synthetic"][method]["edges"]
        status = build_edge_status_matrix(real_edges, synthetic_edges, n_features)
        ordered_status = status[np.ix_(order, order)]
        ax.imshow(
            ordered_status,
            cmap=cmap,
            vmin=-0.5,
            vmax=3.5,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_xticks(ticks)
        if row == 1:
            ax.set_xticklabels(tick_labels, fontsize=7.0)
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_yticks(ticks)
            ax.set_yticklabels(tick_labels, fontsize=7.0)
        else:
            ax.set_yticks([])
        ax.tick_params(axis="both", length=2.0, width=0.75, pad=1.5)
        ax.set_title(
            display_names.get(method, method),
            fontsize=10.2,
            weight="semibold",
            color=METHOD_PRESERVATION_COLORS.get(method, "#222222"),
            pad=4.0,
        )
        ax.text(
            -0.10,
            1.035,
            panel_letters[index],
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.8,
            weight="bold",
            color="#111111",
            clip_on=False,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#333333")

    # Match the heatmap feature numbering in the regularization-path labels.
    X = np.asarray(real_data[dataset_name], dtype=np.float64)
    Xs = _prepare_glasso_input(X)
    emp_cov = Xs.T @ Xs / Xs.shape[0]
    selected_alpha = float(structures[dataset_name]["alpha"])
    off_diagonal_cov = emp_cov.copy()
    np.fill_diagonal(off_diagonal_cov, 0.0)
    zero_solution_alpha = float(np.max(np.abs(off_diagonal_cov)))
    alpha_grid = np.unique(np.append(
        np.geomspace(
            selected_alpha * 0.1,
            max(selected_alpha * 2.5, zero_solution_alpha * 1.05),
            num=40,
        ),
        selected_alpha,
    ))
    edge_i, edge_j = np.triu_indices(Xs.shape[1], k=1)
    path = np.empty((len(alpha_grid), len(edge_i)), dtype=float)
    for alpha_index, alpha in enumerate(alpha_grid):
        if alpha >= zero_solution_alpha:
            precision = np.diag(1.0 / np.diag(emp_cov))
        else:
            mode = "lars" if alpha >= 0.5 * zero_solution_alpha else "cd"
            _, precision = covariance.graphical_lasso(
                emp_cov,
                alpha=float(alpha),
                mode=mode,
                max_iter=1000,
                tol=1e-3,
            )
        path[alpha_index] = precision[edge_i, edge_j]

    selected_flags = np.array([
        (min(int(i), int(j)), max(int(i), int(j))) in real_edges
        for i, j in zip(edge_i, edge_j)
    ])
    # Retained trajectories go down first; zero-at-selected trajectories are
    # drawn last so the dominant sparse outcome remains visible.
    plotted_edges = np.concatenate([
        np.flatnonzero(selected_flags),
        np.flatnonzero(~selected_flags),
    ])
    heatmap_number = np.empty(n_features, dtype=int)
    heatmap_number[order] = np.arange(1, n_features + 1)
    path_ax = fig.add_subplot(grid[2, :])
    path_rows = []
    for edge_index in plotted_edges:
        i, j = int(edge_i[edge_index]), int(edge_j[edge_index])
        edge_a, edge_b = int(heatmap_number[i]), int(heatmap_number[j])
        selected_nonzero = bool(selected_flags[edge_index])
        path_ax.plot(
            np.log(alpha_grid),
            path[:, edge_index],
            color="#C46A2D" if selected_nonzero else "#6F7782",
            linewidth=0.85 if selected_nonzero else 0.42,
            alpha=0.58 if selected_nonzero else 0.24,
            zorder=1 if selected_nonzero else 2,
        )
        path_rows.extend(
            {
                "dataset": dataset_name,
                "alpha": float(alpha),
                "selected_alpha": selected_alpha,
                "feature_a_matrix_index": edge_a,
                "feature_b_matrix_index": edge_b,
                "feature_a": names[i],
                "feature_b": names[j],
                "selected_nonzero": bool(selected_nonzero),
                "precision_coefficient": float(coefficient),
            }
            for alpha, coefficient in zip(alpha_grid, path[:, edge_index])
        )
    path_ax.axvline(
        np.log(selected_alpha),
        color="#222222",
        linestyle="--",
        linewidth=1.25,
        label=rf"Selected $\lambda={selected_alpha:g}$",
    )
    path_ax.axhline(0, color="#777777", linewidth=0.75, alpha=0.75)
    path_ax.set_xlabel(r"$\log(\lambda)$", fontsize=8.2)
    path_ax.set_ylabel("Precision coefficient", fontsize=8.2)
    path_ax.tick_params(axis="both", labelsize=7.0)
    path_ax.grid(True, linestyle="--", linewidth=0.65, alpha=0.38)
    path_ax.text(
        -0.035,
        1.035,
        panel_letters[6],
        transform=path_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.8,
        weight="bold",
        color="#111111",
        clip_on=False,
    )
    path_ax.legend(
        handles=[
            Line2D([0], [0], color="#6F7782", linewidth=1.0, alpha=0.80, label=r"Zero at selected $\lambda$"),
            Line2D([0], [0], color="#C46A2D", linewidth=1.5, label=r"Nonzero at selected $\lambda$"),
            Line2D([0], [0], color="#222222", linewidth=1.25, linestyle="--", label=rf"Selected $\lambda={selected_alpha:g}$"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.27),
        frameon=False,
        fontsize=6.4,
        ncol=3,
        handlelength=1.8,
        columnspacing=0.9,
    )

    legend_handles = [
        Patch(facecolor=STATUS_COLORS["preserved"], edgecolor="#333333", label="Preserved edge"),
        Patch(facecolor=STATUS_COLORS["real_only"], edgecolor="#333333", label="Real-only / lost"),
        Patch(facecolor=STATUS_COLORS["synthetic_only"], edgecolor="#333333", label="Synthetic-only"),
        Patch(facecolor=STATUS_COLORS["absent"], edgecolor="#C9CDD2", label="Absent in both"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.365),
        ncol=4,
        frameon=False,
        fontsize=8.2,
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=1.0,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")

    feature_index = make_feature_index_table(names, order).assign(dataset=dataset_name)
    result = EdgeStatusResult(
        fig=fig,
        metrics=metrics,
        anchor=-1,
        anchor_feature="",
        structures=structures,
        feature_index=feature_index,
        regularization_path=pd.DataFrame(path_rows),
    )
    empty_group_summary = pd.DataFrame(columns=[
        "cluster_id",
        "method",
        "n_features",
        "n_features_matching_method",
        "prominent_features",
        "feature_indices",
        "center_x",
        "center_y",
    ])
    result.preserve_group_summary = empty_group_summary.copy()
    result.lost_group_summary = empty_group_summary.copy()
    result.synthetic_only_group_summary = empty_group_summary.copy()
    result.neighborhood_summary = result.preserve_group_summary
    return result
