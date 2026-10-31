# Final revision — Group 4

## Corrected identity and repository

The correct course group is **Group 4**. Report affiliation, course notice, metadata, README and final handoff filenames are corrected. The actual GitHub URL is included in the compiled report. The existing repository URL is intentionally retained.

## B patch: four files

- `notebooks/B_02_Clustering.ipynb`
- `results/tables/07_cluster_profile.csv`
- `results/figures/08_cluster_cancellation_rate.png`
- `results/figures/11_cluster_cancellation_by_hotel.png`

Apply percentages/rounding per output column directly from unrounded quantities. The final notebook also corrects a numerical comment illustrating double rounding. Cluster sizes/rates and report conclusions are unchanged. The B README is not copied over the integrated README.

## Installation and retraining

Separate base, full SHAP and notebook requirements. Recommend Python 3.12 for the complete workflow, explain macOS OpenMP, and add an environment checker. Retain historical creating/checking metadata instead of making incompatible histories appear identical. Add a resolved Windows/Python 3.12 environment record.

Capture new-run environment and source/data hashes. Add row-aligned comparison of reference and retrained probabilities/metrics. Save new verification reports under `runs/verification/` by default so routine checking does not overwrite historical audit records. Preserve RF/XGBoost parameters, folds, seeds and thread settings. Same-platform fitting checks and clean-environment artifact checks are documented with their actual scope; no cross-platform bitwise guarantee is made.

## Unchanged scientific reference

The reference split, labels, classifier inputs, model predictions, selected parameters and main report findings are retained. The full independent-chronological forest parameter search was not repeated. Revision training diagnostics are separate from the report's reference results. The required Zoom recording remains a course-submission item to complete.
