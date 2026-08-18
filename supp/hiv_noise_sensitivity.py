"""HIV sensitivity analysis using independent additive Gaussian noise."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import synthetic_data_fidelity as analysis
from src.revision.data_io import initialize_datasets


SEED = 42
CVAE_EPOCHS = 200
REPETITIONS = 5
NOISE_LEVELS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00)
METHODS = analysis.METHOD_ORDER


datasets, dataset_summary = initialize_datasets()
synthetic_cohorts = analysis.generate_cohorts(
    datasets,
    methods=METHODS,
    seed=SEED,
    cvae_epochs=CVAE_EPOCHS,
)
noise_results = analysis.compute_noise_sensitivity(
    datasets,
    synthetic_cohorts,
    sigmas=NOISE_LEVELS,
    repeats=REPETITIONS,
    seed=SEED,
)
figure = analysis.plot_noise_sensitivity_summary(
    noise_results,
    dataset="HIV",
    method_order=METHODS,
)
plt.show()
