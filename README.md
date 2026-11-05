# Hotel Booking Cancellation Prediction and Booking Segmentation

Group 4 · KM6312 Information Mining and Analysis

HE CAI · LI YUECHAO · LIANG SIMAN · NIU JINGQI · YOU YIRAN

Repository: https://github.com/JJJelena/Hotel-Cancellation-Group4

The project connects data preparation, exploratory analysis, training-fitted booking segmentation, cancellation classification, random-forest explanations, and training-derived probability assessment under one data and evaluation contract.

## Main results

The cleaned data contain 118,564 bookings, including 44,176 cancellations. All classifiers use the same 24 inputs, fixed row-ID split and five training folds. At threshold 0.50, random-test average precision is 0.373 for Dummy, 0.801 for logistic regression, 0.911 for random forest and 0.883 for XGBoost. The selected forest has F1 0.795; its training-selected threshold of 0.45 gives test F1 0.803 as a separate assessment.

The random benchmark includes repeated input profiles. Forest F1 is 0.580 on unseen random-test profiles and 0.539 in the separately selected chronological experiment. The shared later-arrival refit reuses settings selected on random training and is reported separately from independent temporal selection. These results support a retrospective benchmark, with field availability, future transfer and intervention benefits still requiring operational validation. Segmentation and SHAP describe associations, not causal effects.

## Project files

- `data/`: source and cleaned CSVs, dictionary, fixed split IDs and training folds.
- `src/`: preprocessing, metrics, training, probability assessment and verification.
- `notebooks/00_Integrated_Project.ipynb`: common analytical entry point; `A_*.ipynb` through `E_*.ipynb` retain stage-specific analysis.
- `results/`: the report's reference predictions, models, metrics, tuning records, tables, figures, SHAP and audit records.
- `report/main.pdf`: compiled final report. The accompanying `.tex`, bibliography, figures and official ICML style files form the complete editable source.
- `environments/`: current installation profiles and historical software provenance.
- `docs/Reproducibility.md`: verification and fitting scope; `docs/COLLABORATION.md`: team workflow.

## Install

Use **Python 3.12** for the complete workflow, including SHAP 0.54.1. In a clean virtual environment, from this directory:

```bash
python -m venv .venv
# Windows PowerShell: .venv/Scripts/Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-full.txt
python src/check_environment.py --full
python -m pip check
```

The base `requirements.txt` profile supports Python 3.11 and 3.12 for classifier fitting and saved-result checks without SHAP. Add `requirements-notebooks.txt` for JupyterLab and Seaborn. SHAP 0.54.1 requires Python >=3.12; it is not installed by the base or optional notebook profile. Install both optional files under Python 3.12 for the complete notebook environment.

On macOS, install OpenMP using `brew install libomp` before importing XGBoost and keep Python/Homebrew architectures consistent. See [environment details](environments/README.md). The measured execution environment is Windows; macOS execution is not certified by that check.

## Check the submitted evidence

```bash
python src/run_project.py --stage verify --check-models
python src/verify_e.py --project-root .
```

These commands align rows, labels and folds, recompute metrics, check saved-model predictions, and validate the threshold/calibration outputs. Reports are written under `runs/verification/`, preserving historical audit records. The checkers also accept an explicit `--output`. Matching saved evidence does not mean all parameter searches have been rerun.

## Run the analysis from source

```bash
python src/run_project.py --stage all --output-root runs/reproduction
python src/compare_reproduction.py --reference-root . --run-root runs/reproduction
python src/verify_e.py --project-root . --e-root runs/reproduction --prediction-root runs/reproduction --output runs/reproduction/results/audit/probability_verification.json
```

The full workflow runs A/B analysis, Dummy, logistic regression selection, random-forest random and independent temporal selection, 400-row forest SHAP, XGBoost selection, and OOF threshold/calibration assessment. Stages can also run individually with `--stage ab`, `dummy`, `logistic`, `forest`, `explain`, `xgboost`, or `probability`. The probability stage needs LR, RF and XGBoost predictions in the same run directory. The `all` and `explain` stages check the full environment before fitting.

To recompute probability assessment from the submitted reference predictions alone:

```bash
python src/threshold_calibration.py --project-root . --output-root runs/probability_check
```

Keep new outputs under a separate run directory. The comparison aligns reference and fitted probabilities by row ID, reports metric deltas and changed decisions, and treats independent temporal results separately. Historical model-creating and later checking environments are retained independently. Fixed seeds and package pins do not guarantee identical tree fitting across operating systems, numeric libraries or package builds. Source-built preprocessing avoids depending on the old fitted-template pickle for new fitting.

## Report

The report uses the official ICML 2026 template, 10pt Times-family text on Letter paper, with the course affiliation and actual GitHub URL. Compile `report/main.tex` with pdfLaTeX/BibTeX or upload the complete report source to Overleaf. The ICML style files are unchanged; the PDF identifies this as a KM6312 course project.

## Data attribution

António, N., de Almeida, A., and Nunes, L. (2019). Hotel booking demand datasets. Data in Brief 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126

The project-specific row ID is an alignment key, not a true booking identifier. Dataset and third-party style files retain their respective rights and citations. Method references appear in the report bibliography.
