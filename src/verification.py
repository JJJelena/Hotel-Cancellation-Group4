"""Verify supplied prediction tables against shared split IDs and recomputed metrics.

Usage: python src/verification.py --project-root . --check-models
No training is performed. By default JSON evidence is written to
results/audit/integration_verification.json.
"""
from __future__ import annotations

import argparse
import ast
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, average_precision_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score)


def metrics(y, p, threshold):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    hard = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, hard, labels=[0, 1]).ravel()
    return dict(threshold=threshold, n=len(y), n_positive=int(y.sum()),
        accuracy=accuracy_score(y, hard),
        precision=precision_score(y, hard, zero_division=0),
        recall=recall_score(y, hard, zero_division=0),
        f1=f1_score(y, hard, zero_division=0),
        roc_auc=roc_auc_score(y, p),
        pr_auc=average_precision_score(y, p),
        brier=brier_score_loss(y, p), tn=int(tn), fp=int(fp),
        fn=int(fn), tp=int(tp))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def verify(root: Path, check_models=False):
    df = pd.read_csv(root / 'data/hotel_bookings_clean.csv')
    require(df.row_id.is_unique, 'Clean data row_id is not unique')
    require(df.is_canceled.isin([0, 1]).all(), 'Target is not binary')
    indexed = df.set_index('row_id')
    train = pd.read_csv(root / 'data/splits/train_ids.csv').row_id
    test = pd.read_csv(root / 'data/splits/test_ids.csv').row_id
    chrono = pd.read_csv(root / 'data/splits/chronological_split.csv')
    fold = pd.read_csv(root / 'data/splits/cv_folds.csv').set_index('row_id').cv_fold
    expected = {
        ('random', 'validation'): set(train),
        ('random', 'test'): set(test),
        ('chronological', 'test'): set(chrono.loc[chrono.split.eq('test'), 'row_id']),
    }
    require(set(train).isdisjoint(set(test)), 'Random train/test IDs overlap')
    require(set(train) | set(test) == set(df.row_id), 'Random split does not cover data')
    require(set(fold.index) == set(train), 'CV folds do not exactly cover random train')
    require(fold.isin([1, 2, 3, 4, 5]).all(), 'CV fold numbers must be 1-5')
    files = sorted(p for p in (root / 'results/predictions').glob('*_predictions.csv')
        if p.stem.endswith(('_main_predictions', '_no_agent_predictions')))
    require(bool(files), 'No standard prediction tables found')
    checks, failures = [], []
    for path in files:
        try:
            pred = pd.read_csv(path)
            required = {'row_id', 'model', 'variant', 'experiment', 'subset',
                'fold', 'hotel', 'y_true', 'y_prob'}
            require(required <= set(pred.columns), 'Standard prediction columns missing')
            require(np.isfinite(pred.y_prob).all() and pred.y_prob.between(0, 1).all(),
                'Predicted probabilities must be finite and in [0,1]')
            require(set(pred.row_id) <= set(df.row_id), 'Unknown row_id in predictions')
            truth = indexed.loc[pred.row_id]
            require(np.array_equal(pred.y_true, truth.is_canceled), 'Labels do not match data')
            require(np.array_equal(pred.hotel, truth.hotel), 'Hotel labels do not match data')
            actual = pd.read_csv(root / 'results/metrics' /
                path.name.replace('_predictions.csv', '_metrics.csv'))
            keys = ['model', 'variant', 'experiment', 'subset']
            require(not actual.duplicated(keys).any(), 'Duplicate metric key')
            recalculated, max_error = [], 0.0
            for (model, variant), one in pred.groupby(['model', 'variant'], sort=False):
                require(set(zip(one.experiment, one.subset)) == set(expected),
                    'Expected random validation, random test, and chronological test exactly')
                for (experiment, subset), part in one.groupby(['experiment', 'subset']):
                    require(part.row_id.is_unique, 'Duplicate row_id within prediction subset')
                    require(set(part.row_id) == expected[(experiment, subset)], 'Split IDs differ')
                    if subset == 'validation':
                        require(np.array_equal(part.fold, fold.loc[part.row_id]), 'CV fold labels differ')
                    else:
                        require(part.fold.eq(-1).all(), 'Test fold must be -1')
                    row = actual[(actual.model == model) & (actual.variant == variant)
                        & (actual.experiment == experiment) & (actual.subset == subset)]
                    require(len(row) == 1, 'Exactly one matching metric row is required')
                    calculated = metrics(part.y_true, part.y_prob, float(row.iloc[0].threshold))
                    delta = np.abs(row.iloc[0][list(calculated)].astype(float).to_numpy()
                        - np.asarray(list(calculated.values()), dtype=float))
                    require(np.isfinite(delta).all() and np.max(delta) <= 1e-12,
                        'Saved metrics disagree with predictions')
                    max_error = max(max_error, float(np.max(delta)))
                    recalculated.append(dict(model=model, variant=variant,
                        experiment=experiment, subset=subset, **calculated))
            require(len(actual) == len(recalculated), 'Unexpected extra metric rows')
            entry = dict(file=str(path.relative_to(root)), status='PASS', n=len(pred),
                exact_split_ids=True, exact_labels=True, exact_cv_folds=True,
                metrics_max_absolute_error=max_error)
            if check_models:
                import joblib
                import warnings
                package_path = root / 'results/models' / path.name.replace('_predictions.csv', '.joblib')
                require(package_path.exists(), 'Model package is missing')
                with warnings.catch_warnings(record=True) as caught:
                    package = joblib.load(package_path)
                feature_names = package['input_features']
                require('required_car_parking_spaces' not in feature_names, 'Parking appears in model inputs')
                require('is_canceled' not in feature_names and 'row_id' not in feature_names,
                    'Target or identifier appears in model inputs')
                part = pred[pred.experiment.eq('random') & pred.subset.eq('test')]
                probabilities = package['pipeline'].predict_proba(indexed.loc[part.row_id, feature_names])[:, 1]
                difference = float(np.max(np.abs(probabilities - part.y_prob.to_numpy())))
                require(difference <= 5.01e-7, 'Saved model does not reproduce its prediction export')
                entry.update(model_prediction_max_absolute_error=difference,
                    input_feature_count=len(feature_names),
                    model_load_warnings=[str(x.message) for x in caught])
            checks.append(entry)
        except Exception as error:
            failures.append(dict(file=str(path.relative_to(root)), error=repr(error)))
    strict_path = root / 'results/supplementary/strict_chronological_test_predictions.csv'
    if strict_path.exists():
        try:
            strict = pd.read_csv(strict_path)
            require(strict.row_id.is_unique and set(strict.row_id) == expected[('chronological', 'test')],
                'Strict time test IDs differ')
            require(np.array_equal(strict.y_true, indexed.loc[strict.row_id, 'is_canceled']),
                'Strict time labels differ')
            fold_path = root / 'results/audit/strict_chronological_validation_folds.csv'
            strict_folds = pd.read_csv(fold_path, parse_dates=['scheduled_arrival_date', 'training_end'])
            require((strict_folds.scheduled_arrival_date > strict_folds.training_end).all(),
                'Strict time validation is not after training')
            require((strict_folds.scheduled_arrival_date < pd.Timestamp('2017-05-01')).all(),
                'Strict time validation overlaps test period')
            require(not set(strict_folds.row_id) & set(strict.row_id), 'Strict validation/test IDs overlap')
            checks.append(dict(file=str(strict_path.relative_to(root)), status='PASS',
                exact_split_ids=True, exact_labels=True, strict_time_folds=True,
                metrics=metrics(strict.y_true, strict.cancel_probability, .5)))
        except Exception as error:
            failures.append(dict(file=str(strict_path.relative_to(root)), error=repr(error)))
    return dict(status='PASS' if not failures else 'FAIL', python=platform.python_version(),
        sklearn=sklearn.__version__, n_clean_rows=len(df),
        standard_prediction_files=len(files), checks=checks, failures=failures,
        scope='Saved prediction consistency and optional saved-model inference; no model retraining')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-models', action='store_true')
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output or root / 'results/audit/integration_verification.json'
    try:
        report = verify(root, args.check_models)
    except Exception as error:
        report = dict(status='FAIL', failures=[dict(error=repr(error))])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], report=str(output),
        standard_prediction_files=report.get('standard_prediction_files'),
        failures=report.get('failures')), ensure_ascii=False))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
