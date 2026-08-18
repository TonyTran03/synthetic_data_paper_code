"""Real-synthetic AUC, marginal KLD, and predictive utility analysis."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import synthetic_data_fidelity as analysis
from src.revision.data_io import initialize_datasets


SEED = 42
CVAE_EPOCHS = 200
AUC_REPETITIONS = 50
UTILITY_REPETITIONS = 20
METHODS = analysis.METHOD_ORDER


# Load the three real datasets and generate one class-conditional synthetic
# cohort per method. Each synthetic cohort has the same class sizes as its
# corresponding real dataset.
datasets, dataset_summary = initialize_datasets()
synthetic_cohorts = analysis.generate_cohorts(
    datasets,
    methods=METHODS,
    seed=SEED,
    cvae_epochs=CVAE_EPOCHS,
)

# Repeated origin-classification AUC (real versus synthetic).
auc_results = analysis.compute_origin_auc(
    datasets,
    synthetic_cohorts,
    repeats=AUC_REPETITIONS,
    seed=SEED,
)

# Feature-level marginal divergence and the KS tests used to annotate the
# distributional comparison.
feature_kld = analysis.compute_feature_kld_table(datasets, synthetic_cohorts)
marginal_tests = analysis.compute_marginal_tests(datasets, synthetic_cohorts)

# Train-on-synthetic, test-on-real utility compared with the real-data
# reference model.
utility_results = analysis.compute_tstr_runs(
    datasets,
    synthetic_cohorts,
    repeats=UTILITY_REPETITIONS,
    seed=SEED,
)

figure = analysis.plot_fidelity_auc_kld_utility_gap(
    auc_results,
    feature_kld,
    marginal_tests,
    utility_results,
    dataset_order=list(datasets),
    method_order=METHODS,
    jitter_seed=SEED,
)
plt.show()
