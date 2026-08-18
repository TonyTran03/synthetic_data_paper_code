"""Feature-level real and synthetic marginal distributions for HIV."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import synthetic_data_fidelity as analysis
from src.revision.data_io import initialize_datasets


SEED = 42
CVAE_EPOCHS = 200
FEATURES_PER_PAGE = 16
METHODS = analysis.METHOD_ORDER


datasets, dataset_summary = initialize_datasets()
synthetic_cohorts = analysis.generate_cohorts(
    datasets,
    methods=METHODS,
    seed=SEED,
    cvae_epochs=CVAE_EPOCHS,
)
marginal_tests = analysis.compute_marginal_tests(datasets, synthetic_cohorts)

feature_count = len(datasets["HIV"]["feature_names"])
for first_feature in range(0, feature_count, FEATURES_PER_PAGE):
    analysis.plot_marginal_distribution_grid(
        datasets,
        synthetic_cohorts,
        marginal_tests,
        dataset="HIV",
        method_order=METHODS,
        top_n=FEATURES_PER_PAGE,
        feature_start=first_feature,
    )
plt.show()
