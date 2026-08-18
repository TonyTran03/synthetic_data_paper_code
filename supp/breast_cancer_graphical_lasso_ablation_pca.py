"""Breast Cancer Graphical Lasso, reverse ablation, and fixed-basis PCA."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import synthetic_data_fidelity as analysis
from src.revision import graphical_lasso_plots as glasso
from src.revision.data_io import initialize_datasets


SEED = 42
CVAE_EPOCHS = 200
ABLATION_REPETITIONS = 20
DATASET = "Breast Cancer"
METHODS = analysis.METHOD_ORDER


datasets, dataset_summary = initialize_datasets()
synthetic_cohorts = analysis.generate_cohorts(
    datasets, methods=METHODS, seed=SEED, cvae_epochs=CVAE_EPOCHS
)
ablation_results = analysis.compute_reverse_ablation(
    datasets, synthetic_cohorts, repeats=ABLATION_REPETITIONS, seed=SEED
)
real_matrices = {name: values["X"] for name, values in datasets.items()}
synthetic_matrices = {
    name: {method: sample[0] for method, sample in cohorts.items()}
    for name, cohorts in synthetic_cohorts.items()
}
feature_names = {
    name: values["feature_names"] for name, values in datasets.items()
}
edge_status = glasso.plot_edge_status_matrices(
    real_data=real_matrices,
    synthetic_data=synthetic_matrices,
    feature_names=feature_names,
    alphas=analysis.GRAPHICAL_LASSO_ALPHAS,
    dataset_name=DATASET,
    dataset_order=list(datasets),
    method_order=METHODS,
    comparison_methods=METHODS,
)
plt.close(edge_status.fig)
figure = analysis.plot_graphical_lasso_ablation_pca(
    datasets, synthetic_cohorts, ablation_results, edge_status, dataset=DATASET
)
analysis.label_figure_panels(figure)
plt.show()
