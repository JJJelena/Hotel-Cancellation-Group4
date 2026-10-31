# Execution profiles and historical environments

The dependency files describe current execution profiles. The historical metadata describe the environments that created or verified specific retained artifacts. They serve different purposes and must not be rewritten to look identical.

## Current installation profiles

Use a clean virtual environment. Python 3.12 is the recommended interpreter for the complete project. The base profile also supports Python 3.11. The direct pins include releases with wheels for Python 3.11 and 3.12; this is not a promise of compatibility with every future Python release.

| Purpose | Interpreter | Install |
| --- | --- | --- |
| Saved-result checks, clustering and classifier training | Python 3.11 or 3.12 | `python -m pip install -r requirements.txt` |
| Complete CLI workflow including RF Tree SHAP | Python 3.12 | `python -m pip install -r requirements-full.txt` |
| Notebook interface and EDA plotting | Base interpreter, or Python 3.12 when also using SHAP | `python -m pip install -r requirements-notebooks.txt` |

`requirements-full.txt` adds SHAP 0.54.1, whose published metadata require Python >=3.12. It therefore cannot be installed as part of a Python 3.11 environment. The optional notebook file does not silently install SHAP. For the complete notebook environment, install both `requirements-full.txt` and `requirements-notebooks.txt` under Python 3.12.

Check the profile after installation:

```bash
python src/check_environment.py
python src/check_environment.py --full
python src/check_environment.py --full --notebooks
python -m pip check
```

Use only the command for the profile you installed. The check compares installed direct versions with the chosen requirement files, evaluates installed Requires-Python metadata where available, and imports the packages to check native loading. It fits no models. `--metadata-only` cannot test DLL/OpenMP loading.

### macOS and Windows native libraries

On macOS, install the LLVM OpenMP runtime before loading the pinned XGBoost wheel:

```bash
brew install libomp
```

Run Python, Homebrew and native libraries for the same architecture, especially on Apple Silicon. An import error mentioning `libomp.dylib` usually indicates a missing or incompatible OpenMP runtime. The environment check reports that failure and the relevant installation step. The Windows integration check does not certify macOS execution.

Windows XGBoost also requires the Microsoft Visual C++ runtime used by its wheel. If import fails with a missing DLL, follow the official XGBoost installation guide rather than changing model settings to hide an import problem.

## Retained artifact provenance

`environment_manifest.json` indexes the supplied historical and integration records. In particular:

- `results/audit/software_versions.json` records the historical D forest/SHAP environment, including Python 3.12.14 and scikit-learn 1.9.0.
- `results/models/logistic_regression_main.json` records the logistic model's creating environment, including Python 3.13.17 and scikit-learn 1.9.1.
- `results/audit/integration_verification.json` records a later check of saved prediction files and models. A checker environment is not the model's creating environment.
- `results/audit/xgboost_protocol.json` records the completed E experiment and its own software information.

These records intentionally differ. The original JSON files remain unchanged as provenance evidence. The current full profile does not claim to reproduce all those creating environments at once. Direct-version pins also leave platform-specific wheels and some transitive dependencies to the installer; they are not a complete cross-platform binary lock.

## Reproducibility scope

Saved prediction tables and fitted models are the reference evidence for the report. The independent checks test their relationship to the common split, labels, fold assignments and published metrics. New training can reproduce the analytical protocol without producing identical bytes or identical predictions on every platform. This particularly affects a model selected after several nearly tied candidates and chronological evaluation under a changed software/platform stack.

Keep retraining outputs in a fresh run directory. Do not overwrite the report's reference tables merely to make a new environment appear to match them. Compare a rerun with the retained evidence, record its complete environment and report any material difference. The scientific limitations of the random and time evaluation protocols remain unchanged by an installation fix.

## Official metadata and installation sources

Version-specific Python constraints were checked on 31 October 2026 against these release pages:

- [NumPy 2.2.3](https://pypi.org/project/numpy/2.2.3/)
- [pandas 2.2.3](https://pypi.org/project/pandas/2.2.3/)
- [scikit-learn 1.9.1](https://pypi.org/project/scikit-learn/1.9.1/)
- [SciPy 1.17.1](https://pypi.org/project/scipy/1.17.1/)
- [Matplotlib 3.10.6](https://pypi.org/project/matplotlib/3.10.6/)
- [joblib 1.6.0](https://pypi.org/project/joblib/1.6.0/)
- [XGBoost 3.2.0](https://pypi.org/project/xgboost/3.2.0/)
- [nbformat 5.10.4](https://pypi.org/project/nbformat/5.10.4/)
- [SHAP 0.54.1](https://pypi.org/project/shap/0.54.1/)
- [seaborn 0.13.2](https://pypi.org/project/seaborn/0.13.2/)
- [JupyterLab 4.4.6](https://pypi.org/project/jupyterlab/4.4.6/)
- [XGBoost installation guide](https://xgboost.readthedocs.io/en/release_3.2.0/install.html)
