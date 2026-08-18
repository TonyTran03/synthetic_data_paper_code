"""Marginal-preserving within-class dependence perturbation analysis."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import synthetic_data_fidelity as analysis
from src.revision.data_io import initialize_datasets


SEED = 42
REPETITIONS = 10
PERMUTATION_PROPORTIONS = tuple(percent / 100 for percent in range(0, 101, 10))


datasets, dataset_summary = initialize_datasets()

# At each level, independently permute the selected observations within every
# feature and outcome class. This preserves each class-conditional marginal
# distribution while progressively disrupting cross-feature dependence.
permutation_results = analysis.compute_dependence_permutation_sensitivity(
    datasets,
    proportions=PERMUTATION_PROPORTIONS,
    repeats=REPETITIONS,
    seed=SEED,
)

# The plotted AUC compares real with perturbed observations. Edge recovery is
# the fraction of real-data Graphical Lasso edges retained after perturbation.
figure = analysis.plot_dependence_permutation_sensitivity(
    permutation_results,
    dataset_order=list(datasets),
)
plt.show()
