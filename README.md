# Probabilistic Building Energy Prediction

**Gaussian Process Regression, uncertainty diagnostics and reproducible evaluation across 768 simulated building configurations.**

[![tests](https://github.com/jacobemanuel/building-energy-gpr/actions/workflows/tests.yml/badge.svg)](https://github.com/jacobemanuel/building-energy-gpr/actions/workflows/tests.yml)

Can a probabilistic surrogate provide accurate load estimates while also indicating when its predictions should not be trusted?

I built a leakage-safe evaluation pipeline for heating and cooling load using the [UCI Energy Efficiency benchmark](https://doi.org/10.24432/C51307). The study compares ridge regression, a 300-tree random forest and exact Gaussian Process Regression under the same held-out splits. Point accuracy is only half of the analysis: native Gaussian predictive intervals are checked empirically, then contrasted with a separate split-conformal coverage check.

> Independent M.Sc. research project, Technical University of Munich, Management and Technology.

## At a glance

- **Data:** 768 simulated designs, 8 input variables, 2 regression targets, no missing or duplicate rows
- **Targets:** heating load (`Y1`) and cooling load (`Y2`)
- **Models:** RidgeCV, Random Forest and exact Gaussian Process Regression
- **Primary split:** fixed 80/20 holdout with seed 42
- **Stability check:** 10 repeated random 80/20 holdouts with seeds 1 to 10
- **Uncertainty:** empirical diagnostics for native Gaussian intervals plus a separate split-conformal check
- **Reproducibility:** persisted split indices, configuration, predictions, metrics and generated figures
- **Verification:** 11 automated tests covering data, metrics, pipeline behavior and artifacts

## Experimental design

The dataset is a structured Ecotect simulation grid rather than an observational sample of real buildings. Inputs describe relative compactness, surface and wall area, roof area, overall height, orientation, glazing area and glazing-area distribution.

Orientation and glazing distribution are nominal variables. The main pipeline therefore one-hot encodes them instead of treating their numeric labels as meaningful distances. Continuous variables are standardized. Every preprocessing step remains inside a scikit-learn `Pipeline` and is fitted on the training partition only.

The same outer split is shared across models and targets. This makes the point-error comparison direct and prevents preprocessing leakage. Split indices, model configuration, held-out predictions and evaluation metrics are saved so every public number can be traced back to a generated artifact.

![Heating and cooling load across the structured design grid](outputs/figures/target_joint_by_x8.png)

*Heating and cooling loads form a strongly structured design grid; color marks glazing-area distribution.*

## Models

- **RidgeCV** provides a regularized linear baseline.
- **Random Forest** provides a strong nonlinear point-prediction baseline.
- **One-hot GPR** is the principled probabilistic model reported in the main analysis.
- **Numeric-encoding GPR** is retained as an ablation. It improves some point metrics but imposes an unjustified ordinal geometry on nominal categories, so it is not treated as the primary model.

The Gaussian process uses a constant multiplied by a Matern-3/2 kernel, plus a `WhiteKernel` residual component. On deterministic simulation data, that component should be read as model lack of fit or numerical tolerance, not as an empirical decomposition of real-world aleatoric uncertainty.

## Held-out results

Fixed 80/20 holdout, seed 42:

| Target | Model | MAE | RMSE | R2 | Native 90% coverage | Native 95% coverage |
|---|---|---:|---:|---:|---:|---:|
| Heating load (`Y1`) | Random Forest | **0.367** | **0.533** | **0.9973** | n/a | n/a |
| Heating load (`Y1`) | GPR | 0.518 | 0.703 | 0.9953 | 96.1% | 98.7% |
| Cooling load (`Y2`) | Random Forest | 1.270 | 2.035 | 0.9553 | n/a | n/a |
| Cooling load (`Y2`) | GPR | **1.038** | **1.497** | **0.9758** | 91.6% | 92.9% |

The ranking is target-dependent. Random Forest produced the lowest heating-load point error. GPR performed better on cooling load and additionally supplied model-based uncertainty estimates. I kept that trade-off visible instead of presenting one model as a universal winner.

![Gaussian-process predictions against held-out values](outputs/figures/gpr_pred_vs_true.png)

*Held-out predictions from the principled one-hot GPR. The orange line is the identity reference, not a fitted trend.*

## Stability across repeated holdouts

The primary result was checked across 10 repeated random 80/20 holdouts. Values are mean plus or minus standard deviation for the one-hot GPR.

| Target | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Heating load (`Y1`) | 0.536 +/- 0.044 | 0.719 +/- 0.063 | 0.995 +/- 0.001 |
| Cooling load (`Y2`) | 1.084 +/- 0.114 | 1.502 +/- 0.233 | 0.974 +/- 0.008 |

These are repeated holdouts, not k-fold cross-validation. The distinction is recorded because the evaluation design determines what the variation means.

## Uncertainty, checked rather than assumed

Predictive standard deviations from GPR were evaluated through empirical coverage, reliability curves and standardized residuals. Native intervals were conservative for heating load. For cooling load, 90% coverage was close to nominal while 95% coverage reached 92.9%, indicating mild undercoverage at that level.

![Empirical reliability of native Gaussian predictive intervals](outputs/figures/gpr_reliability.png)

*Empirical coverage of native Gaussian intervals. Points above the diagonal indicate conservative intervals; points below it indicate undercoverage.*

![Standardized residual diagnostics](outputs/figures/gpr_std_residuals.png)

A separate split-conformal sensitivity check used an absolute-residual calibration set drawn from the outer training partition. On the untouched test set it produced:

| Target | 90% nominal | 95% nominal |
|---|---:|---:|
| Heating load | 94.8% empirical coverage | 97.4% empirical coverage |
| Cooling load | 89.6% empirical coverage | 95.5% empirical coverage |

These are finite-sample marginal coverage checks under the conformal exchangeability assumption. They are not a claim of conditional calibration, causal validity or readiness for deployment on real buildings.

<details>
<summary>Additional view: feature and target distributions</summary>

![Feature and target distributions](outputs/figures/feature_target_histograms.png)

</details>

## Reproduce the analysis

The reported pipeline runs on CPU and was validated from a clean checkout.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

python3 -m src.eda
python3 -m src.train
python3 -m src.evaluate
python3 scripts/conformal_check.py
python3 scripts/repeated_splits.py --repeats 10 --start-seed 1

python3 -m pytest tests/ -q
```

The optional GPyTorch backend is experimental and excluded from the reported scikit-learn results.

## Repository map

```text
src/                 data loading, preprocessing, training and evaluation
scripts/             robustness, conformal and artifact-generation utilities
tests/               data, metrics, integration and artifact checks
outputs/metrics/     persisted configurations and numerical results
outputs/predictions/ held-out predictions for every model and target
outputs/figures/     generated diagnostics used in this README and report
report/              public technical report and supplementary appendix
```

## Reports

- [Technical report](report/report.pdf)
- [Supplementary appendix](report/appendix.pdf)

The public PDFs preserve the academic content while omitting private university identifiers. This README provides a compact, reproducible index to the reported result set.

## Scope and limitations

Random holdouts primarily test interpolation within the controlled simulation grid, not generalization to unseen real buildings. The benchmark is useful for comparing pipelines and uncertainty behavior, but it does not reproduce sensor noise, occupancy shifts, weather uncertainty or changes in building stock.

Exact GPR is practical here because `n = 768`; its cubic training cost will not scale unchanged to very large datasets. The model comparison is deliberately compact and reproducible rather than a state-of-the-art hyperparameter competition.

## Why it belongs in my research portfolio

The domain differs from my current XR work, but the research pattern is the same: define a measurable question, build a traceable apparatus, prevent leakage, compare honest baselines, inspect uncertainty and document enough for another person to reproduce the result.

## Data attribution

Tsanas, A. and Xifara, A. (2012). *Energy Efficiency* [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C51307](https://doi.org/10.24432/C51307). Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Author

**Jakub Majewski**

[github.com/jacobemanuel](https://github.com/jacobemanuel)
