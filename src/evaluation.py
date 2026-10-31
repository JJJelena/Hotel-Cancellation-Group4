"""
Shared evaluation template for the hotel cancellation project (owner: C).

Every supervised model (Dummy, Logistic Regression, Random Forest, XGBoost)
uses this module so that metrics, prediction files and model packages have
exactly the same definitions and formats.

Typical use (inside a notebook in /notebooks):

    import sys; from pathlib import Path
    project_root = Path.cwd().parent
    sys.path.append(str(project_root / "src"))
    import evaluation as ev

    data = ev.load_shared_data(project_root)
    pipe = ev.make_pipeline(data, YourClassifier(**best_params))
    metrics, preds = ev.run_standard_experiment(
        pipe, model="random_forest", data=data, params=best_params
    )

Fixed conventions
-----------------
model       : one of MODEL_NAMES
variant     : "main" for the primary model; e.g. "no_agent" for the ablation
experiment  : "random" (shared 80/20 split) or "chronological"
subset      : "validation" (5-fold out-of-fold predictions on the random
              training set, using A's fixed folds in data/splits/cv_folds.csv)
              or "test"
threshold   : 0.5 for the main comparison (E reports other thresholds
              separately)
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Fixed project settings (do not change without team agreement)
# ---------------------------------------------------------------------------
SEED = 42
N_FOLDS = 5
DEFAULT_THRESHOLD = 0.5
TARGET = "is_canceled"
ID_COL = "row_id"

MODEL_NAMES = ["dummy", "logistic_regression", "random_forest", "xgboost"]
EXPERIMENTS = ["random", "chronological"]
SUBSETS = ["validation", "test"]

METRIC_COLUMNS = [
    "model", "variant", "experiment", "subset", "threshold",
    "n", "n_positive",
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier",
    "tn", "fp", "fn", "tp",
]

PREDICTION_COLUMNS = [
    "row_id", "model", "variant", "experiment", "subset", "fold",
    "hotel", "y_true", "y_prob",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_shared_data(project_root: str | Path) -> dict:
    """Load A's shared data package and return all splits in one dict."""
    root = Path(project_root)
    df = pd.read_csv(root / "data" / "hotel_bookings_clean.csv")
    train_ids = pd.read_csv(root / "data" / "splits" / "train_ids.csv")[ID_COL]
    test_ids = pd.read_csv(root / "data" / "splits" / "test_ids.csv")[ID_COL]
    chrono = pd.read_csv(root / "data" / "splits" / "chronological_split.csv")
    cv_folds = pd.read_csv(root / "data" / "splits" / "cv_folds.csv")
    # Rebuild the unfitted A-v4 contract without a version-dependent pickle.
    from preprocessing_source import build_preprocessor_template
    preprocessor = build_preprocessor_template()

    features = []
    for _, _, cols in preprocessor.transformers:
        features.extend(list(cols))

    chrono_train_ids = chrono.loc[chrono["split"] == "train", ID_COL]
    chrono_test_ids = chrono.loc[chrono["split"] == "test", ID_COL]

    def subset(ids):
        return df[df[ID_COL].isin(ids)].reset_index(drop=True)

    data = {
        "project_root": root,
        "df": df,
        "features": features,
        "preprocessor": preprocessor,
        "random_train": subset(train_ids),
        "random_test": subset(test_ids),
        "chrono_train": subset(chrono_train_ids),
        "chrono_test": subset(chrono_test_ids),
        "cv_folds": cv_folds,
    }

    # Structural checks only: feature list and split files come from A's
    # package, so the template keeps working if the team changes them.
    assert features and len(features) == len(set(features))
    assert ID_COL not in features and TARGET not in features
    all_ids = set(df[ID_COL])
    for a, b in [("random_train", "random_test"), ("chrono_train", "chrono_test")]:
        ids_a, ids_b = set(data[a][ID_COL]), set(data[b][ID_COL])
        assert ids_a.isdisjoint(ids_b), f"{a} and {b} overlap."
        assert ids_a | ids_b == all_ids, f"{a} + {b} do not cover all records."

    # get_cv() must reproduce A's fixed folds exactly (cv_folds.csv, numbered 1-5)
    check = data["random_train"][[ID_COL]].copy()
    check["fold"] = _fold_numbers(data["random_train"])
    merged = check.merge(cv_folds, on=ID_COL, how="left", validate="one_to_one")
    assert merged["cv_fold"].notna().all(), "cv_folds.csv does not cover the training set."
    assert (merged["fold"] == merged["cv_fold"]).all(), (
        "ev.get_cv() does not match A's cv_folds.csv - check the data files."
    )
    return data


def get_cv() -> StratifiedKFold:
    """The single cross-validation scheme used by every team member.

    Identical to A's fixed folds in data/splits/cv_folds.csv when applied to
    data["random_train"] (verified automatically in load_shared_data)."""
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def _fold_numbers(train: pd.DataFrame) -> np.ndarray:
    """Validation fold (1-5, A's numbering) of each row in the random training set."""
    folds = np.zeros(len(train), dtype=int)
    for k, (_, val_idx) in enumerate(get_cv().split(train, train[TARGET]), start=1):
        folds[val_idx] = k
    return folds


def get_cv_folds(data: dict) -> pd.DataFrame:
    """A's fixed fold (1-5) for every row_id in the random training set."""
    return data["cv_folds"].copy()


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
def build_preprocessor(data: dict, drop_features: list[str] | None = None):
    """Clone A's template; optionally remove features (e.g. ['agent'])."""
    template = data["preprocessor"]
    if not drop_features:
        return clone(template)
    drop = set(drop_features)
    transformers = [
        (name, clone(trans), [c for c in cols if c not in drop])
        for name, trans, cols in template.transformers
    ]
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_pipeline(data: dict, classifier, drop_features=None) -> Pipeline:
    """Preprocessing template + classifier in one sklearn Pipeline."""
    return Pipeline([
        ("preprocessing", build_preprocessor(data, drop_features)),
        ("model", classifier),
    ])


def _features_used(data: dict, drop_features=None) -> list[str]:
    drop = set(drop_features or [])
    return [f for f in data["features"] if f not in drop]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_prob, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """All agreed metrics for one set of predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "n": len(y_true),
        "n_positive": int(y_true.sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def metrics_from_predictions(preds: pd.DataFrame,
                             threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """Recompute the metrics table from a prediction table (any threshold)."""
    rows = []
    keys = ["model", "variant", "experiment", "subset"]
    for key, g in preds.groupby(keys, sort=False):
        row = dict(zip(keys, key))
        row.update(compute_metrics(g["y_true"], g["y_prob"], threshold))
        rows.append(row)
    return pd.DataFrame(rows)[METRIC_COLUMNS]


def _prediction_table(part: pd.DataFrame, y_prob, model, variant,
                      experiment, subset, fold=None) -> pd.DataFrame:
    return pd.DataFrame({
        "row_id": part[ID_COL].to_numpy(),
        "model": model,
        "variant": variant,
        "experiment": experiment,
        "subset": subset,
        "fold": fold if fold is not None else -1,
        "hotel": part["hotel"].to_numpy(),
        "y_true": part[TARGET].to_numpy().astype(int),
        "y_prob": np.round(np.asarray(y_prob, dtype=float), 6),
    })[PREDICTION_COLUMNS]


# ---------------------------------------------------------------------------
# Standard experiment
# ---------------------------------------------------------------------------
def run_standard_experiment(pipeline: Pipeline, model: str, data: dict,
                            variant: str = "main", drop_features=None,
                            params: dict | None = None,
                            threshold: float = DEFAULT_THRESHOLD,
                            save: bool = True, verbose: bool = True):
    """
    Run the agreed protocol for an already-tuned pipeline.

    1. Validation: 5-fold out-of-fold probabilities on the random training set
       (A's fixed folds in cv_folds.csv, reproduced by get_cv()).
    2. Random test: fit on the full random training set, predict the test set.
    3. Chronological test: refit the SAME settings on the chronological
       training set and predict later bookings. No tuning on this split.

    Saves results/metrics/<model>_<variant>_metrics.csv,
          results/predictions/<model>_<variant>_predictions.csv,
          results/models/<model>_<variant>.joblib (fitted on random train).
    Returns (metrics_df, predictions_df).
    """
    assert model in MODEL_NAMES, f"model must be one of {MODEL_NAMES}"
    feats = _features_used(data, drop_features)
    tr, te = data["random_train"], data["random_test"]
    ctr, cte = data["chrono_train"], data["chrono_test"]
    pred_parts = []

    # 1. out-of-fold validation predictions
    oof = np.zeros(len(tr))
    folds = np.full(len(tr), -1)
    for k, (fit_idx, val_idx) in enumerate(get_cv().split(tr, tr[TARGET]), start=1):
        m = clone(pipeline).fit(tr.iloc[fit_idx][feats], tr.iloc[fit_idx][TARGET])
        oof[val_idx] = m.predict_proba(tr.iloc[val_idx][feats])[:, 1]
        folds[val_idx] = k
        if verbose:
            print(f"  fold {k}/{N_FOLDS} done")
    part = _prediction_table(tr, oof, model, variant, "random", "validation")
    part["fold"] = folds
    pred_parts.append(part)

    # 2. random test
    fitted = clone(pipeline).fit(tr[feats], tr[TARGET])
    pred_parts.append(_prediction_table(
        te, fitted.predict_proba(te[feats])[:, 1],
        model, variant, "random", "test"))

    # 3. chronological test (same settings, refit on earlier bookings)
    chrono_model = clone(pipeline).fit(ctr[feats], ctr[TARGET])
    pred_parts.append(_prediction_table(
        cte, chrono_model.predict_proba(cte[feats])[:, 1],
        model, variant, "chronological", "test"))

    preds = pd.concat(pred_parts, ignore_index=True)
    metrics = metrics_from_predictions(preds, threshold)
    validate_outputs(metrics, preds, data)

    if save:
        root = data["project_root"] / "results"
        tag = f"{model}_{variant}"
        (root / "metrics").mkdir(parents=True, exist_ok=True)
        (root / "predictions").mkdir(parents=True, exist_ok=True)
        metrics.to_csv(root / "metrics" / f"{tag}_metrics.csv", index=False)
        preds.to_csv(root / "predictions" / f"{tag}_predictions.csv", index=False)
        save_model_package(fitted, model, variant, feats, params, data)
        if verbose:
            print(f"Saved metrics, predictions and model package for '{tag}'.")
    return metrics, preds


# ---------------------------------------------------------------------------
# Model package
# ---------------------------------------------------------------------------
def save_model_package(fitted_pipeline, model, variant, features, params, data):
    """Fitted pipeline (random training set) + everything needed to reuse it."""
    try:
        import xgboost
        xgb_version = xgboost.__version__
    except ImportError:
        xgb_version = None
    package = {
        "pipeline": fitted_pipeline,
        "model": model,
        "variant": variant,
        "input_features": list(features),
        "processed_feature_names": list(
            fitted_pipeline.named_steps["preprocessing"].get_feature_names_out()
        ),
        "params": params or {},
        "trained_on": "random_train (train_ids.csv)",
        "seed": SEED,
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "xgboost": xgb_version,
        },
    }
    path = data["project_root"] / "results" / "models" / f"{model}_{variant}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, path)
    with open(path.with_suffix(".json"), "w") as f:
        json.dump({k: v for k, v in package.items() if k != "pipeline"},
                  f, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def validate_outputs(metrics: pd.DataFrame, preds: pd.DataFrame, data: dict):
    """Run before handing results to C. Raises AssertionError if anything is off."""
    assert list(metrics.columns) == METRIC_COLUMNS, "Metric columns differ from template."
    assert list(preds.columns) == PREDICTION_COLUMNS, "Prediction columns differ from template."
    assert preds["model"].isin(MODEL_NAMES).all()
    assert preds["experiment"].isin(EXPERIMENTS).all()
    assert preds["subset"].isin(SUBSETS).all()
    assert preds["y_prob"].between(0, 1).all(), "Probabilities must be in [0, 1]."
    assert not preds["y_prob"].isna().any()

    expected = {
        ("random", "validation"): set(data["random_train"][ID_COL]),
        ("random", "test"): set(data["random_test"][ID_COL]),
        ("chronological", "test"): set(data["chrono_test"][ID_COL]),
    }
    for (exp, sub), ids in expected.items():
        for (mdl, var), g in preds.groupby(["model", "variant"]):
            part = g[(g["experiment"] == exp) & (g["subset"] == sub)]
            assert set(part[ID_COL]) == ids and part[ID_COL].is_unique, (
                f"{mdl}/{var} {exp}-{sub}: row_ids do not match the shared split."
            )
    labels = data["df"].set_index(ID_COL)[TARGET]
    assert (preds["y_true"].to_numpy() == labels.loc[preds[ID_COL]].to_numpy()).all(), \
        "y_true does not match is_canceled."
    return True


# ---------------------------------------------------------------------------
# Collecting results (used by C for the final comparison)
# ---------------------------------------------------------------------------
def load_all_metrics(project_root) -> pd.DataFrame:
    files = sorted((Path(project_root) / "results" / "metrics").glob("*_metrics.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_all_predictions(project_root) -> pd.DataFrame:
    files = sorted((Path(project_root) / "results" / "predictions").glob("*_predictions.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def duplicate_overlap_ids(data: dict) -> set:
    """row_ids in the random test set whose model features also appear in training.
    Used for the de-duplicated sensitivity check required by A's README."""
    feats = data["features"]
    train_fp = set(pd.util.hash_pandas_object(
        data["random_train"][feats], index=False).astype("uint64"))
    test_fp = pd.util.hash_pandas_object(
        data["random_test"][feats], index=False).astype("uint64")
    return set(data["random_test"].loc[test_fp.isin(train_fp).to_numpy(), ID_COL])
