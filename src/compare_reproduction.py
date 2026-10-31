"""Compare a new training run with saved reference predictions by row ID.

This reports numeric differences. It does not impose an unvalidated tolerance
or turn successful reference verification into a claim of identical retraining.
Example: python src/compare_reproduction.py --reference-root . --run-root runs/reproduction
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)


def metrics(y, probability, threshold=.5):
    decision = np.asarray(probability) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, decision, labels=[0, 1]).ravel()
    return dict(n=len(y), threshold=float(threshold),
        precision=float(precision_score(y, decision, zero_division=0)),
        recall=float(recall_score(y, decision, zero_division=0)),
        f1=float(f1_score(y, decision, zero_division=0)),
        average_precision=float(average_precision_score(y, probability)),
        roc_auc=float(roc_auc_score(y, probability)),
        brier=float(brier_score_loss(y, probability)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def normalise(frame):
    frame = frame.copy()
    if 'y_prob' not in frame and 'cancel_probability' in frame:
        frame = frame.rename(columns={'cancel_probability': 'y_prob'})
    required = {'row_id', 'y_true', 'y_prob'}
    if not required.issubset(frame):
        raise ValueError(f'Missing probability fields: {required-set(frame)}')
    if frame.row_id.duplicated().any():
        raise ValueError('Duplicate row_id within an evaluation subset')
    if not np.isfinite(frame.y_prob).all() or not frame.y_prob.between(0, 1).all():
        raise ValueError('Probabilities must be finite and in [0,1]')
    return frame.set_index('row_id').sort_index()


def compare_one(reference, rerun, key):
    reference, rerun = normalise(reference), normalise(rerun)
    if not reference.index.equals(rerun.index):
        raise ValueError(f'{key}: reference/run row-ID populations differ')
    if not np.array_equal(reference.y_true, rerun.y_true):
        raise ValueError(f'{key}: target labels differ')
    for field in ('hotel', 'fold'):
        if field in reference and field in rerun and not np.array_equal(reference[field], rerun[field]):
            raise ValueError(f'{key}: {field} alignment differs')
    first = reference.y_prob.to_numpy(float)
    second = rerun.y_prob.to_numpy(float)
    y = reference.y_true.to_numpy(int)
    threshold = float(reference.threshold.iloc[0]) if 'threshold' in reference else .5
    if 'threshold' in rerun and not rerun.threshold.eq(threshold).all():
        raise ValueError(f'{key}: decision thresholds differ')
    baseline, repeated = metrics(y, first, threshold), metrics(y, second, threshold)
    delta = np.abs(second-first)
    metric_delta = {name: repeated[name]-baseline[name] for name in baseline
                    if name not in ('n', 'threshold')}
    return dict(key=key, row_id_and_label_alignment='identical', n=len(y),
        max_absolute_probability_difference=float(delta.max()),
        mean_absolute_probability_difference=float(delta.mean()),
        rms_probability_difference=float(np.sqrt(np.mean((second-first)**2))),
        unequal_probability_count=int(np.count_nonzero(first != second)),
        changed_decision_count=int(np.count_nonzero((first >= threshold) != (second >= threshold))),
        reference_metrics=baseline, rerun_metrics=repeated, rerun_minus_reference_metrics=metric_delta,
        scope='numeric comparison of documented exports; no bitwise-retraining acceptance threshold is assumed')


def environment_evidence(root):
    evidence = {}
    directory = root/'results/audit'
    names = ['software_versions.json', 'integration_verification.json', 'xgboost_protocol.json',
        'xgboost_run_environment.json', 'rf_run_environment.json', 'run_environment.json',
        'random_selected_parameters.json', 'strict_chronological_selected_parameters.json']
    for name in names:
        path = directory/name
        if not path.is_file():
            continue
        obj = json.loads(path.read_text(encoding='utf-8'))
        evidence[name] = {key: obj[key] for key in ('python', 'platform', 'packages', 'software',
            'versions', 'seed', 'explicit_thread_settings', 'thread_environment',
            'algorithm_settings', 'input_and_source_sha256', 'metadata_scope',
            'parameters', 'selected_params', 'selected_candidate', 'criterion') if key in obj}
    return evidence


def compare(reference_root, run_root):
    reference_root, run_root = Path(reference_root).resolve(), Path(run_root).resolve()
    compared, errors, missing = [], [], []
    reference_files = sorted(p for p in (reference_root/'results/predictions').glob('*_predictions.csv')
        if p.stem.endswith(('_main_predictions', '_no_agent_predictions')))
    for path in reference_files:
        other = run_root/'results/predictions'/path.name
        if not other.is_file():
            missing.append(path.relative_to(reference_root).as_posix())
            continue
        first, second = pd.read_csv(path), pd.read_csv(other)
        group_fields = ['model', 'variant', 'experiment', 'subset']
        first_groups = dict(tuple(first.groupby(group_fields, sort=False)))
        second_groups = dict(tuple(second.groupby(group_fields, sort=False)))
        if first_groups.keys() != second_groups.keys():
            errors.append(dict(file=path.name, error='Evaluation subset keys differ'))
            continue
        for group, frame in first_groups.items():
            key = '/'.join(map(str, group))
            try:
                compared.append(compare_one(frame, second_groups[group], key))
            except (AssertionError, ValueError, KeyError) as exc:
                errors.append(dict(file=path.name, key=key, error=str(exc)))
    strict_name = 'results/supplementary/strict_chronological_test_predictions.csv'
    first, second = reference_root/strict_name, run_root/strict_name
    if first.is_file() and second.is_file():
        try:
            compared.append(compare_one(pd.read_csv(first), pd.read_csv(second), 'random_forest/main/strict_chronological/test'))
        except (AssertionError, ValueError, KeyError) as exc:
            errors.append(dict(file=strict_name, error=str(exc)))
    elif first.is_file():
        missing.append(strict_name)
    return dict(schema_version='retraining-comparison-v1',
        status='structural_mismatch' if errors else 'comparison_completed' if compared else 'no_matching_run_predictions',
        compared_subset_count=len(compared), comparisons=compared, structural_errors=errors,
        reference_outputs_not_present_in_this_run=missing,
        reference_environment_evidence=environment_evidence(reference_root),
        run_environment_evidence=environment_evidence(run_root),
        claim_boundary='Saved-artifact verification and new-training numeric comparison are different checks. This file reports differences without an unsupported pass tolerance. Missing reference outputs can be expected for a single-stage rerun. Cross-platform or cross-version bitwise reproducibility is not claimed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = compare(args.reference_root, args.run_root)
    output = args.output or args.run_root/'results/audit/retraining_comparison.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = [dict(key=row['key'], n=row['n'],
        max_probability_difference=row['max_absolute_probability_difference'],
        mean_probability_difference=row['mean_absolute_probability_difference'],
        changed_decisions=row['changed_decision_count'],
        reference_f1=row['reference_metrics']['f1'], rerun_f1=row['rerun_metrics']['f1'],
        delta_f1=row['rerun_minus_reference_metrics']['f1'],
        reference_ap=row['reference_metrics']['average_precision'], rerun_ap=row['rerun_metrics']['average_precision'],
        delta_ap=row['rerun_minus_reference_metrics']['average_precision'],
        reference_brier=row['reference_metrics']['brier'], rerun_brier=row['rerun_metrics']['brier']) for row in result['comparisons']]
    pd.DataFrame(rows).to_csv(output.with_suffix('.csv'), index=False)
    print(json.dumps({'status': result['status'], 'compared_subsets': result['compared_subset_count'],
        'structural_errors': result['structural_errors'], 'report': str(output)}, ensure_ascii=True))
    if result['status'] != 'comparison_completed':
        sys.exit(1)


if __name__ == '__main__':
    main()
