# Validation history — Group 4

## Historical integration through 31 October 2026

### Corrected identity and repository

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

## Submission fixes (31 Oct)

- Course code corrected to KM6312 in the report and README; group number remains Group 4.
- Repository URL set with `src/set_repository.py`; `report/main.pdf` recompiled so the PDF matches the source (Group 4, KM6312, repository link).
- `src/verification.py` now writes routine reports to `runs/verification/` by default, so running the check no longer overwrites the submitted `results/audit/integration_verification.json`.
- Added outputs that the A_01, B_01 and B_02 notebooks generate but were not previously committed (missing-value report, EDA/clustering figures and tables). All committed CSV results were re-run and reproduced unchanged.


## Final source review on 4 November 2026

The online Group 4 repository at commit `54273e5e0f264bf32410b61c5704991506e904e9` was cloned into a separate review workspace. Its complete `--stage all` command finished: A/B, Dummy, all 12 logistic candidates, both six-candidate forest searches, main and agent refits, 400-row SHAP, all three XGBoost candidates, and OOF threshold/calibration assessment. Saved-result checks and a separate verification of all three rerun probability-assessment sources passed. The 19-subset comparison retains measured numeric differences rather than claiming bitwise equality.

The reference scientific files and the four B correction files remain unchanged. The source fixes separate the optional SHAP/notebook requirements, preflight the complete environment, preserve historical verification records, let E verification read the entire same-run prediction directory, protect reference SHAP outputs, and record new fitting environments. Model settings, seeds, folds and selection rules are unchanged. The verified academic report body and reference figures remain unchanged. PDF Subject metadata identify the course project. The full rerun and observed LR differences are documented separately in `docs/Full_Validation_20261104.md` and the dated audit records.

No online commit was made by this review. The repository URL in the report remains the actual Group 4 address. Presentation recording details belong in the course's designated submission channel; the report contains no blank recording fields.
