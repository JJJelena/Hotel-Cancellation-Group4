# Hotel Booking Cancellation Prediction and Booking Segmentation

Group 5 · K6312 Information Mining and Analysis

HE CAI · LI YUECHAO · LIANG SIMAN · NIU JINGQI · YOU YIRAN

This project integrates data preparation, exploratory analysis, training-only K-means segmentation, Dummy / logistic regression / random forest / XGBoost comparison, forest SHAP explanations, and OOF-based threshold and probability assessment.

## Results and scope

The cleaned data contain 118,564 bookings and 24 supervised inputs. Random-test AP is 0.373 (Dummy), 0.801 (logistic), 0.911 (forest), and 0.883 (XGBoost), with a common decision threshold of 0.50. The forest is the principal benchmark selected by training-fold AP among the tested configurations. Its F1 is 0.795 at 0.50 and 0.803 at the training-selected threshold 0.45.

Random-test performance is limited by repeated profiles. The unseen-profile forest F1 is 0.580. The shared later-arrival refit reuses randomly selected settings and must not be called untouched future validation. A separate forest selected only on early-period expanding windows has later-arrival F1 0.539 and AP 0.764. Field availability at initial booking remains unverified. Segmentation and SHAP are descriptive, not causal.

## Files

- `data/`: original and cleaned data, data dictionary, fixed split IDs and five training folds.
- `src/`: shared metric API, source-built 24-input preprocessing, training and verification commands.
- `notebooks/00_Integrated_Project.ipynb`: common reading and verification entry point.
- `notebooks/A_*.ipynb`, `B_*.ipynb`, `C_*.ipynb`, `D_*.ipynb`, `E_*.ipynb`: stage-specific analytical records.
- `results/metrics`, `predictions`, `tuning`, `tables`, `figures`, `shap`, `models`, `audit`: retained reference evidence and completed integration outputs.
- `report/`: official ICML 2026 style files, complete paper source, figures, references and compiled report.

## Environment and verification

Use Python 3.11+ in a virtual environment and install `python -m pip install -r requirements.txt`. The integration checks and E fitting were run with Python 3.11.2 / scikit-learn 1.9.1. Historical D forests and A template were created with scikit-learn 1.9.0; C metadata records 1.9.1. The original environment records are retained. A single requirements file does not retroactively make the historical creating environments identical. Original pickle artifacts were loaded and predictions matched in the recorded validation environment; CSV evidence remains portable across environments.

From this directory:

```bash
python src/run_project.py --stage verify
python src/run_project.py --stage verify --check-models
python src/verify_e.py --project-root .
```

Verification checks row IDs, labels, folds, metrics, saved model predictions, strict-time boundaries and explanation reconstruction. It does not rerun all historical hyperparameter searches. Complete model artifacts are needed for `--check-models`.

## Reproduce separately

The unified entry point writes retraining into a separate run directory to preserve reference evidence. Training may take substantial time.

```bash
python src/run_project.py --stage ab --output-root runs/reproduction
python src/run_project.py --stage dummy --output-root runs/reproduction
python src/run_project.py --stage logistic --output-root runs/reproduction
python src/run_project.py --stage forest --output-root runs/reproduction
python src/run_project.py --stage explain --output-root runs/reproduction
python src/run_project.py --stage xgboost --output-root runs/reproduction
python src/run_project.py --stage probability --output-root runs/reproduction
```

`--stage all` runs these stages in sequence. The probability stage expects LR, RF and XGBoost predictions in the same output run. To repeat only probability assessment from submitted predictions, use `python src/threshold_calibration.py --project-root . --output-root runs/probability_check`.

The source preprocessing template exactly matched the original 221-feature transformation in the integration check. All classifier preprocessing is fitted inside training folds. The submitted 24-feature contract excludes parking, ADR, deposit type, customer type, special requests and season. Re-running historical A/B/C notebooks can regenerate their local analysis outputs; keep an archival copy and use the unified CLI for isolated full-model reproduction. Notebook working directories are the project root or `notebooks/`.

## Report and submission metadata

Compile `report/main.tex` with pdfLaTeX / BibTeX (or upload the complete report source to Overleaf). Official ICML style files are unmodified. The course notice identifies this as a course project, not an ICML conference acceptance. The report uses 10pt Times-family text and Letter paper.

Set the real repository URL using `python src/set_repository.py --url https://github.com/OWNER/REPOSITORY`, then recompile and verify the PDF. The URL is not fabricated. Zoom recording information belongs in the course submission form rather than a public repository.

## Data attribution

António, N., de Almeida, A., and Nunes, L. (2019). Hotel booking demand datasets. Data in Brief 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126

The project-specific row ID is not a true booking identifier. Dataset and third-party style files retain their respective rights and citations. See the paper bibliography for algorithm and method references.
