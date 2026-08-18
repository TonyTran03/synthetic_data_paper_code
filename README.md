# Reproducible synthetic-data analyses

Figure 1 is a conceptual workflow diagram, so there is deliberately no
`figure_1.py` file.

## Scripts corresponding to the paper

```text
main_figure/
  fidelity_auc_kld_utility_gap.py
      Figure 2: discriminator AUC, marginal KLD, and utility gap

  hiv_graphical_lasso_ablation_pca.py
      Figure 3: HIV edge preservation, reverse ablation, and fixed-basis PCA

supp/
  hiv_noise_sensitivity.py
      Supplementary Figure S1

  hiv_marginal_distributions.py
      Supplementary Figure S2

  breast_cancer_graphical_lasso_ablation_pca.py
      Supplementary Figure S3

  diabetes_graphical_lasso_ablation_pca.py
      Supplementary Figure S4

  hiv_graphical_lasso_regularization_path.py
      Supplementary Graphical Lasso regularization-path analysis

  within_class_dependence_permutation.py
      New marginal-preserving dependence sensitivity analysis
```

The scripts are intentionally ordinary top-to-bottom analysis files. They show
the seeds, repetition counts, model generation, metric computation, and plot
export directly. Reusable scientific functions are in `src/`; generator
implementations are in `models/`.

## Run

Create an environment and install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run a single analysis from this directory:

```powershell
python main_figure/fidelity_auc_kld_utility_gap.py
python supp/within_class_dependence_permutation.py
```

Each script opens its Matplotlib figure with `plt.show()`. It does not save a
PDF, PNG, JPEG, or any other rendered artifact. `cache/` is retained only for
legacy low-level helpers and can be deleted safely between runs.

## Data

`data/allSyntheticData.RData` contains the HIV input used by the paper. Breast
Cancer is loaded from scikit-learn. Diabetes is loaded from OpenML with the
existing scikit-learn fallback retained for offline execution.
