# CAC-2026 Data Challenge — Olive Oil Musty-Defect Detection

Classifying olive oil samples for the **musty sensory defect** (present/absent)
by **mid-level data fusion** of three spectroscopic blocks — headspace-MS
(HS-MS), mid-infrared (MIR/ATR-FTIR), and UV-vis. Per-block PLS-DA scores are
fused and passed to a final classifier, evaluated by **F1** under nested
cross-validation. Extends the *3MLpls* strategy of Borràs et al. (2016).

## Results

The selected model is **mid-level PLS fusion + SVM-RBF**, chosen from a five-head
sweep (PLS-DA, LDA, QDA, SVM-RBF, Random Forest):

| | Honest nested-CV (5×3) |
|---|---|
| **F1** | **0.78 ± 0.03** |
| Precision | 0.80 |
| Recall | 0.76 |

Key findings:

- **HS-MS is the dominant block** (~73 % of fused-score importance), consistent
  with mustiness being an aroma defect that headspace-MS measures directly.
- **Widening the MIR and UV-vis regions** was the most effective single change
  (nested-CV F1 ≈ 0.69 → 0.76).
- The performance ceiling is **feature-bound**: five classifier families and
  several preprocessing variants all plateau near F1 ≈ 0.78.

Full methodology, settings, and the complete results table are in
**[Modelling Pipeline Description](modeling_pipeline_description.md)**.

## Pipeline (in brief)

```
HS-MS ─┐
MIR  ──┼─ per-block PLS-DA scores ─ hstack ─ [scale] ─ classifier ─ musty / non-musty
UV-vis─┘
```

Per-block latent-variable counts and the final classifier's hyperparameters are
tuned **jointly** in one nested-CV search. See the
[pipeline description](modeling_pipeline_description.md) for the rationale,
architecture diagrams, and validation design.

## Repository structure

```
.
├── src/olive_oil/            # installable package
│   ├── data.py               # Excel loading, replicate averaging, label join
│   ├── preprocessing.py      # region select, SNV, SG-derivative, log, row-profile, PQN, mean-center
│   ├── blocks.py             # BlockConfig + prepare_blocks (region, recipe, outliers, alignment)
│   ├── outliers.py           # PCA Hotelling's T² / Q-residual screening
│   ├── models.py             # BlockSet, PLSScorer, PLSDAClassifier, mid-level fusion estimator
│   ├── evaluation.py         # nested_cv, tune_final_model, predict_test
│   └── visualization.py      # spectra / score / loading plots
├── Notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02a_Classification_with_PLSDA.ipynb     # final classifier sweep …
│   ├── 02b_Classification_with_LDA.ipynb
│   ├── 02c_Classification_with_QDA.ipynb
│   ├── 02d_Classification_with_SVM.ipynb       # selected model
│   ├── 02e_Classification_with_RandomForest.ipynb
│   └── 02f_SVM_threshold_tuning.ipynb          # operating-point finalisation
├── tests/                    # pytest suite for data/preprocessing
├── Docs/                     # reference paper (Borràs et al., 2016)
├── Data/                     # dataset xlsx (gitignored — not committed)
├── modeling_pipeline_description.md            # full methodology & results
├── pipeline.py               # standalone fusion-estimator smoke test
└── pyproject.toml
```

## Setup

```bash
conda create -n olive-oil python=3.13 && conda activate olive-oil   # requires Python >= 3.10
pip install -e ".[dev,viz,notebooks]"   # package + pytest + plotting + Jupyter
```

Place the challenge dataset at `Data/CAC2026_Data_challenge.xlsx` (the `Data/`
folder is gitignored). Then run the notebooks (`02d` is the selected model) or the
tests:

```bash
pytest -q
```

## Reference

Borràs, E., et al. (2016). *Olive oil sensory defects classification with data
fusion of instrumental techniques and multivariate analysis (PLS-DA).* Food
Chemistry, 203, 314–322. https://doi.org/10.1016/j.foodchem.2016.02.038
