# DES Viscosity Prediction with System-Disjoint Nested Cross-Validation

Code and data accompanying the manuscript

> **Machine Learning with System-Disjoint Nested Cross-Validation for DES Viscosity Prediction: GFN2-xTB-Derived and Compositional Descriptors with SHAP Interpretability**
> Udayakumar Mani, Lavanya Priyadarshini Ramalingam, Senthilkumar Rathinasamy
> Green Separation Engineering Laboratory, SASTRA Deemed to be University, Thanjavur, India
> Manuscript COMPTC-D-26-01938 (revised)

This repository reports **exactly the same pipeline run as the revised manuscript**. Every number, table and figure in the manuscript can be regenerated from the files here.

---

## 1. Overview

The pipeline predicts the dynamic viscosity of deep eutectic solvents (DESs) from 22 descriptors (9 derived from GFN2-xTB calculations, 13 compositional, temperature, molecular-size and hydrogen-bond-topology descriptors). Models are evaluated with **fully system-disjoint nested cross-validation**: all measurements of a DES are held out together, in both the outer evaluation loop and the inner hyperparameter search, so performance reflects prediction for DES systems not seen during training.

## 2. Repository contents

| File | Description |
|---|---|
| `DES_Experimental_Data.xlsx` | Original measurement records: 140 DES formulations; density, viscosity and conductivity at 25, 30, 35, 40, 45 and 50 °C |
| `DES_viscosity_CORRECTED.xlsx` | Rebuilt dataset: every formulation with a unique ID and every viscosity at its measured temperature (840 rows); descriptor status of each formulation; data-quality flags |
| `viscosity_dataset_CORRECTED_ml_ready.csv` | Modelling dataset used in the manuscript: 121 DES × 6 temperatures (726 rows) with descriptors |
| `viscosity_pipeline_v3_CORRECTED.py` | Complete analysis pipeline (settings used for the manuscript) |
| `viscosity_output_CORRECTED/` | All outputs of the manuscript run (figures, tables, reports, fitted models) |

## 3. Data

### 3.1 Measurements
Viscosity, density and conductivity were measured for 140 DES formulations (natural, protic, amphiphilic, surfactant-based and amino-acid-based) at six temperatures, 25–50 °C. Measurements for each formulation were made sequentially on the same sample from 25 to 50 °C.

### 3.2 Modelling dataset
The modelling dataset was assembled directly from the measurement records:

* each formulation has a **unique identifier** (e.g., `CC:LA` = choline chloride : lactic acid and `CC:Lac` = choline chloride : lactose; `BBC:L(1:1)` and `BBC:L(2:1)` for the two molar ratios);
* each viscosity value is assigned to the temperature at which it was measured;
* **19 formulations were excluded**: 13 whose descriptor sets could not be assigned unambiguously and 6 without descriptor calculations (listed in Table 1 of the manuscript and in `DES_viscosity_CORRECTED.xlsx`, sheet `Descriptor_status`);
* **1 measurement was excluded**: L:P400 at 40 °C (1753 cP), the only non-monotonic value in the dataset.

**Final dataset: 725 measurements for 121 DES systems at 25–50 °C** (3.1–2528 cP; median 89 cP).

### 3.3 Note on the earlier release
The dataset released with the original submission (`viscosity_dataset_filled.csv`) was assembled incorrectly: viscosity values were shifted by one temperature step, the 25 °C position contained a density value, the 50 °C viscosity was missing, and formulations sharing an abbreviated name (e.g., L = lactic acid and lactose; M = malic acid and maltose) were merged. **That file should not be used.** It is superseded by the files above.

### 3.4 Known limitations of the data
* For almost all systems, viscosity decreases by a median factor of about 2.2 per 5 °C step between 25 and 40 °C but only about 1.1 per step between 40 and 50 °C (see `reports/data_qc_step_ratio_by_temperature.csv` and Figure 2 of the manuscript).
* For several chloride-salt HBAs combined with the same HBD, the descriptor vectors are identical, and for some systems the component-difference descriptors take a common filled-in value. These are discussed as limitations in Section 4.4 of the manuscript.

## 4. Requirements

The manuscript run used:

| Package | Version |
|---|---|
| Python | 3.10.11 |
| scikit-learn | 1.7.1 |
| xgboost | 3.2.0 |
| shap | 0.48.0 |
| numpy, pandas, scipy, matplotlib, seaborn, joblib | recent versions |

```bash
pip install numpy pandas scipy matplotlib seaborn joblib scikit-learn==1.7.1 xgboost==3.2.0 shap==0.48.0
```

> **Note:** GroupKFold fold membership depends on the scikit-learn version. To reproduce the per-fold results (Tables 3–6) exactly, use scikit-learn 1.7.1.

## 5. Running the pipeline

Place `viscosity_pipeline_v3_CORRECTED.py` and `viscosity_dataset_CORRECTED_ml_ready.csv` in the same folder and run:

```bash
python viscosity_pipeline_v3_CORRECTED.py
```

Runtime is about 90 minutes on a desktop PC (most of it is the ten fully re-tuned repeated partitions). For a 2-minute smoke test with small grids:

```bash
FAST=1 python viscosity_pipeline_v3_CORRECTED.py        # Linux / macOS
set FAST=1 && python viscosity_pipeline_v3_CORRECTED.py # Windows
```

### Main settings (`Config` class at the top of the script)

| Setting | Value used | Meaning |
|---|---|---|
| `data_path` | `viscosity_dataset_CORRECTED_ml_ready.csv` | Input data |
| `value_corrections` | `{("L:P400", 40): None}` | Excludes the single non-monotonic measurement |
| `drop_temperatures` | `[]` | All six temperatures used |
| `outer_folds`, `inner_folds` | 5, 4 | Nested GroupKFold cross-validation |
| `corr_threshold` | 0.95 | Correlation pruning, refitted inside every training fold |
| `tuning_scoring` | `r2` (on ln(1 + η)) | Inner-loop selection criterion |
| `n_repeats`, `repeat_tuning` | 10, `full` | Repeated system-disjoint partitions with full re-tuning |
| `random_state` | 42 | Seed for all stochastic estimators |

## 6. Method summary

1. **Target:** models are trained on ln(1 + η); predictions are back-transformed with exp(ŷ) − 1.
2. **Preprocessing (inside every training fold):** median imputation → correlation pruning (|r| > 0.95; 27 candidate descriptors → 22, identical in all folds) → standardisation.
3. **Outer loop:** 5-fold GroupKFold by DES system (25, 24, 24, 24, 24 systems per test fold).
4. **Inner loop:** 4-fold GroupKFold GridSearchCV, also system-disjoint.
5. **Models:** Random Forest, Gradient Boosting and XGBoost (identical nested tuning); Decision Tree, SVR, KNN, Ridge, ElasticNet and Linear Regression (fixed settings); cross-validated Arrhenius population baseline on the same folds.
6. **Statistics:** Wilcoxon signed-rank tests on the main partition; Nadeau–Bengio corrected resampled t-tests over 10 repeated partitions.
7. **Interpretation:** SHAP (TreeExplainer) on held-out samples of each outer fold, for Random Forest and GBM; partial dependence; learning curve over the number of training systems.
8. **Metrics:** RMSLE (primary), R², RMSE and MAE; fold results are mean ± sample SD (n − 1).

## 7. Key results (manuscript run)

| Model | R² (mean ± SD) | RMSLE (mean ± SD) |
|---|---|---|
| XGBoost | 0.715 ± 0.125 | 0.572 ± 0.106 |
| Gradient Boosting | 0.711 ± 0.082 | 0.555 ± 0.111 |
| **Random Forest (primary)** | 0.673 ± 0.114 | **0.542 ± 0.112** |
| Decision Tree | 0.620 ± 0.103 | 0.765 ± 0.143 |
| SVR (RBF) | 0.516 ± 0.193 | 0.667 ± 0.101 |
| ElasticNet | 0.486 ± 0.151 | 0.814 ± 0.107 |
| Ridge | 0.485 ± 0.151 | 0.812 ± 0.104 |
| Linear Regression | 0.474 ± 0.192 | 0.823 ± 0.110 |
| KNN (k = 5) | 0.420 ± 0.080 | 0.699 ± 0.057 |
| Arrhenius baseline | 0.368 ± 0.105 | 1.101 ± 0.092 |

* **Repeated partitions (10 × 5-fold, full re-tuning):** Random Forest had the lowest RMSLE in 7 of 10 repeats; no pairwise difference among the three ensembles was significant (all p ≥ 0.28).
* **Leading SHAP descriptors (Random Forest):** 1000/T, weighted LUMO energy, T × solvation energy, and hydrogen-bond contact counts; Random Forest–GBM rank agreement Spearman ρ = 0.76.
* **Learning curve:** R² rises from 0.38 (≈19 training systems) to 0.67 (≈97 systems) without a clear plateau.

## 8. Outputs (`viscosity_output_CORRECTED/`)

| Path | Content | Manuscript |
|---|---|---|
| `figures/fig1_data_landscape.png` | Viscosity landscape | Figure 1 |
| `figures/figS4_data_qc_diagnostic.png` | Viscosity ratio per 5 °C step | Figure 2 |
| `figures/fig2_arrhenius_vs_ml.png` | Arrhenius baseline vs Random Forest | Figure 3 |
| `figures/fig3_nested_cv_performance.png` | Random Forest nested-CV performance | Figure 4 |
| `figures/fig4_model_comparison.png` | Nine-model benchmark | Figure 5 |
| `figures/figS3_repeated_cv.png` | Repeated partitions | Figure 6 |
| `figures/fig5_shap_summary.png` | SHAP summary (Random Forest) | Figure 7 |
| `figures/figS1_shap_rf_vs_gbm.png` | SHAP: Random Forest vs GBM | Figure 8 |
| `figures/fig6_shap_dependence.png` | SHAP dependence plots | Figure 9 |
| `figures/fig7_error_anatomy.png` | Error analysis | Figure 10 |
| `figures/fig8_partial_dependence.png` | Partial dependence | Figure 11 |
| `figures/figS2_learning_curve.png` | Learning curve | Figure 12 |
| `model_comparison_nested_cv.csv`, `fold_metrics_all_models.csv` | Benchmark and per-fold metrics for all models | Tables 5–7 |
| `significance_tests.csv` | Wilcoxon tests (main partition) | Table 8 |
| `reports/repeated_cv_*.csv` | Repeated-partition scores, rank stability, corrected t-tests | Table 9 |
| `reports/tableS1_shap_fold_variance_rf.csv`, `..._gbm.csv` | Fold-level SHAP variability | Tables 10–11 |
| `shap_importance_rf.csv`, `shap_importance_gbm.csv`, `shap_rf_vs_gbm_rank_comparison.csv` | SHAP importances and rank comparison | Tables 12–13 |
| `reports/tableS2_fold_membership.csv` | DES systems in each outer fold | Table 3 |
| `reports/tableS3_fold_descriptor_profile.csv` | Descriptor profile of each fold | Table 4 |
| `reports/learning_curve_*.csv` | Learning curve | Figure 12 |
| `oof_predictions_rf.csv`, `oof_predictions_gbm.csv` | Out-of-fold predictions with system, temperature and fold | — |
| `per_system_errors_rf.csv` | System-level errors | Section 4.2 |
| `reports/data_qc_*.csv`, `reports/removed_rows.csv` | Data-quality checks and excluded measurement | Section 2.1 |
| `reports/model_summary.json` | Summary of the run (versions, settings, headline metrics) | — |
| `models/*.pkl` | Fitted pipelines for each outer fold (Random Forest, GBM) | — |

(Output file names such as `figS4_…` are kept from the pipeline; the table above maps them to the figure numbers in the manuscript.)

## 9. Citation

If you use this code or data, please cite the article (details to be added on publication) and the companion density study:

> Mani, U.; Ramalingam, L. P.; Rathinasamy, S. Deep eutectic solvent density prediction: machine learning from quantum chemical descriptors reveals electronic and stoichiometric controls. *J. Mol. Liq.* 2026, 457, 129715.

## 10. License and contact

License: [to be added]
Contact: Udayakumar Mani, Senthilkumar Rathinasamy — Green Separation Engineering Laboratory, SASTRA Deemed to be University, Thanjavur, Tamil Nadu 613 401, India.
