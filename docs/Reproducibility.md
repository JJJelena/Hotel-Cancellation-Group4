# Verification and retraining comparison

The submitted report describes the saved reference experiments. Verification of those files and retraining from source answer different questions.

## Saved reference verification

```bash
python src/run_project.py --stage verify --check-models
python src/verify_e.py --project-root .
```

These checks align saved predictions to the shared row IDs, outcomes and folds, recompute their metrics, and check the saved estimators and probability-assessment outputs. They do not repeat every hyperparameter search. Historical six-decimal exports are evaluated at their documented precision. A successful check supports the consistency of the submitted evidence, not a guarantee that fresh fitting on any machine produces the same trees or probabilities.

## A separate training run

Use the full environment described in the project README for all training and SHAP stages. Preserve `results/` as the report's reference record.

```bash
python src/run_project.py --stage all --output-root runs/reproduction
python src/compare_reproduction.py --reference-root . --run-root runs/reproduction
```

`compare_reproduction.py` aligns each common evaluation subset by `row_id`. It checks outcomes, hotel and fold labels, then reports reference and rerun precision, recall, F1, AP, ROC-AUC, Brier and confusion counts. It also reports maximum, mean and RMS probability differences and the number of changed decisions. Structural mismatches stop a valid comparison. Numeric differences are reported without an unsupported acceptance threshold. A `comparison_completed` status means the comparison ran, not that the models are identical or an observed difference is scientifically negligible.

A stage-specific rerun may legitimately contain only some reference prediction files. The comparison lists absent files explicitly. The independent chronological forest is compared separately from the shared chronological refit. If parameters change after a new search, interpret the results together with each run's selected-parameter files.

## Environment and execution record

`src/reproduction_metadata.py` captures execution facts before fitting: Python implementation and compiler, OS and architecture, installed package versions, explicit estimator thread settings, relevant OpenMP/BLAS environment variables, loaded numeric-library versions, and SHA256 digests of data, fixed splits, source files and requirements files. This observed run record does not replace the historical creating-environment records.

The reference protocol uses seed 42, 160 forest trees with `n_jobs=10`, and histogram XGBoost with `n_jobs=6`. The revision retains these settings rather than silently changing the benchmark to one thread. Single-thread fitting can be a separately recorded diagnostic, but would not establish cross-platform equality by itself. XGBoost's documentation identifies floating-point summation order and multithreading as possible sources of small differences. Scikit-learn requires a fixed `random_state` to resolve random-forest fitting randomness, while version and platform changes still need explicit validation.

## Revision checks that were actually run

On the current Windows environment with Python 3.11.2, scikit-learn 1.9.1 and XGBoost 3.2.0, the full three-candidate, five-fold XGBoost experiment and its random/chronological refits were rerun. The selected configuration remained depth 7, 300 trees and learning rate 0.05. All 140,700 exported probabilities matched the reference values after CSV parsing, and AP, F1, Brier and decisions matched for each subset.

The saved independent-chronological forest configuration was also refitted on the 96,428 earlier-arrival records without repeating its five-window parameter search. Its 22,136 later-arrival probabilities differed by at most 3.33e-16, with no changed decisions and identical F1 and AP. These are same-platform checks only. They do not invalidate a colleague's observed platform drift, verify macOS fitting, or prove the full historical forest search was rerun.

The compact evidence is in `results/audit/revision_retraining_checks/`. The report's original predictions and models were not replaced by these diagnostics. Cross-version or cross-platform bitwise retraining is not claimed.

## Fresh full-environment check on Python 3.12

A new Python 3.12.14 environment with scikit-learn 1.9.1, XGBoost 3.2.0 and SHAP 0.54.1 was also tested on Windows. The full XGBoost candidate/fold experiment and both final refits again matched all 140,700 reference CSV probabilities, decisions and AP/F1/Brier values. The selected strict-time forest refit, without repeating parameter search, had a maximum probability difference of 3.33e-16 and no decision, AP or F1 changes; its parsed-CSV Brier differed only in the last displayed decimal place. These are measured results on this environment, not a tolerance or a cross-platform promise.

For a fresh SHAP API check without overwriting the frozen 400-row E analysis:

```bash
python src/verify_shap_api.py --output-root runs/shap_api_check --sample-size 20
```

The check uses the saved random forest, the original 100-row training background and the first 20 rows of the original 400-row test sampling order. It executes `TreeExplainer.shap_values` with the existing float32 threshold-routing compatibility policy and checks class-1 probability additivity. On the tested Python 3.12 environment the maximum reconstruction error was 3.9643e-8, below the existing 1e-5 additivity tolerance, and the saved model file's SHA256 was unchanged. This confirms API execution and additivity on those 20 rows. It does not recompute the full 400-row global ranking, bootstrap stability or every local explanation. Loading the historical forest emitted scikit-learn's expected 1.9.0-to-1.9.1 version warning; a successful check is not a general model-persistence compatibility guarantee.

The compact Python 3.12 evidence is in `results/audit/revision_retraining_checks/clean_python312/` and includes the actual environment records. The frozen model and reference prediction/SHAP files remain unchanged.

## Primary documentation

- XGBoost FAQ, slightly different results between runs: https://xgboost.readthedocs.io/en/release_3.2.0/faq.html
- RandomForestClassifier fitting notes: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- Scikit-learn model persistence and environment compatibility: https://scikit-learn.org/stable/model_persistence.html
