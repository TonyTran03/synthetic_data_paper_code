"""HIV Graphical Lasso, reverse feature ablation, and fixed-basis PCA."""

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
DATASET = "HIV"
METHODS = analysis.METHOD_ORDER


datasets, dataset_summary = initialize_datasets()
synthetic_cohorts = analysis.generate_cohorts(
    datasets,
    methods=METHODS,
    seed=SEED,
    cvae_epochs=CVAE_EPOCHS,
)

# Rank features by their contribution to origin discrimination, then remove
# them cumulatively and recompute AUC after every removal step.
ablation_results = analysis.compute_reverse_ablation(
    datasets,
    synthetic_cohorts,
    repeats=ABLATION_REPETITIONS,
    seed=SEED,
)

# Fit the real and synthetic Graphical Lasso models using the dataset-specific
# regularization values reported in the manuscript.
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

# PCA directions are fitted once to standardized real HIV data. Synthetic
# observations are projected onto those same directions; percentages shown for
# synthetic panels are variance captured after projection, not refitted axes.
figure = analysis.plot_graphical_lasso_ablation_pca(
    datasets,
    synthetic_cohorts,
    ablation_results,
    edge_status,
    dataset=DATASET,
)
analysis.label_figure_panels(figure)
plt.show()
