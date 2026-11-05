"""Reproduce C's logistic regression selection and same-parameter agent ablation.

Usage: python src/train_logistic.py --project-root . --output-root rerun
The default performs all 12 training-only CV candidates. --plan-only checks
the CSV contract and prints the plan without training or writing results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def choose_one_standard_error(table, n_folds=5):
    table = table.sort_values('cv_pr_auc', ascending=False).reset_index(drop=True)
    best = table.iloc[0]
    standard_error = float(best.cv_pr_auc_sd / np.sqrt(n_folds))
    minimum = float(best.cv_pr_auc - standard_error)
    eligible = table[table.cv_pr_auc >= minimum]
    chosen = eligible.sort_values('C', kind='stable').iloc[0]
    weight = chosen.class_weight
    params = dict(C=float(chosen.C), class_weight=None if pd.isna(weight) or weight == 'None' else weight)
    return params, dict(rule='smallest C within one standard error of highest mean CV AP',
        highest_cv_mean_ap=float(best.cv_pr_auc), best_cv_standard_error=standard_error,
        eligibility_minimum_ap=minimum, selected_cv_mean_ap=float(chosen.cv_pr_auc),
        selected_parameters=params, n_folds=n_folds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--skip-ablation', action='store_true')
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root / 'src'))
    import evaluation as ev
    from preprocessing_source import load_shared_data
    from reproduction_metadata import write_run_environment
    data = load_shared_data(project_root)
    plan = dict(n_train=len(data['random_train']), input_features=len(data['features']),
        C=[0.001, .01, .1, 1, 10, 100], class_weight=[None, 'balanced'],
        cv='fixed A-v4 five stratified folds', selection='one-standard-error on mean fold average precision',
        threshold=.5, preprocessing='source-built A-v4 template, fitted inside each pipeline/fold')
    print(json.dumps(plan, indent=2))
    if args.plan_only:
        return
    output_root = (args.output_root or project_root/'runs'/'logistic_reproduction').resolve()
    if output_root == project_root:
        parser.error('Choose a separate output root to preserve reference results.')
    data['project_root'] = output_root
    tuning = output_root / 'results/tuning'
    tuning.mkdir(parents=True, exist_ok=True)
    write_run_environment(output_root/'results/audit/logistic_run_environment.json',
        project_root=project_root,stage='logistic_training',seed=ev.SEED,
        thread_settings={'grid_search_n_jobs':args.n_jobs},
        algorithm_settings={'C':plan['C'],'class_weight':plan['class_weight'],
            'max_iter':5000,'selection':plan['selection'],
            'decision_threshold':ev.DEFAULT_THRESHOLD,
            'agent_ablation':not args.skip_ablation})
    pipe = ev.make_pipeline(data, LogisticRegression(max_iter=5000, random_state=ev.SEED))
    grid = {'model__C': plan['C'], 'model__class_weight': plan['class_weight']}
    scoring = dict(pr_auc='average_precision', roc_auc='roc_auc', f1='f1',
        recall='recall', precision='precision')
    search = GridSearchCV(pipe, grid, cv=ev.get_cv(), scoring=scoring,
        refit='pr_auc', n_jobs=args.n_jobs)
    train = data['random_train']
    search.fit(train[data['features']], train.is_canceled)
    raw = pd.DataFrame(search.cv_results_)
    table = pd.DataFrame(dict(C=raw['param_model__C'],
        class_weight=raw['param_model__class_weight'].fillna('None'),
        cv_pr_auc=raw.mean_test_pr_auc, cv_pr_auc_sd=raw.std_test_pr_auc,
        cv_roc_auc=raw.mean_test_roc_auc, cv_f1=raw.mean_test_f1,
        cv_recall=raw.mean_test_recall, cv_precision=raw.mean_test_precision))
    table = table.sort_values('cv_pr_auc', ascending=False).reset_index(drop=True)
    table.to_csv(tuning / 'logistic_regression_cv_results.csv', index=False)
    params, evidence = choose_one_standard_error(table)
    (tuning / 'logistic_regression_selection.json').write_text(
        json.dumps(evidence, indent=2), encoding='utf-8')
    selected = ev.make_pipeline(data,
        LogisticRegression(max_iter=5000, random_state=ev.SEED, **params))
    main_metrics, _ = ev.run_standard_experiment(selected,
        model='logistic_regression', data=data, params=params)
    import joblib
    package = joblib.load(output_root / 'results/models/logistic_regression_main.joblib')
    coefficients = package['pipeline'].named_steps['model'].coef_.ravel()
    coefficient_table = pd.DataFrame(dict(
        feature=[name.split('__', 1)[1] for name in package['processed_feature_names']],
        coef=coefficients))
    # Retained for compatibility with C's table schema. With full one-hot
    # encoding, categorical comparisons use exp(beta_a-beta_b), not a raw
    # exponentiated coefficient against a specified reference category.
    coefficient_table['odds_ratio'] = np.exp(coefficient_table.coef)
    coefficient_table.sort_values('coef', ascending=False).to_csv(
        output_root / 'results/metrics/logistic_regression_coefficients.csv', index=False)
    if not args.skip_ablation:
        reduced = ev.make_pipeline(data,
            LogisticRegression(max_iter=5000, random_state=ev.SEED, **params),
            drop_features=['agent'])
        reduced_metrics, _ = ev.run_standard_experiment(reduced,
            model='logistic_regression', variant='no_agent', data=data,
            drop_features=['agent'], params={**params, 'dropped': ['agent'],
                'selection': 'main-model settings reused, no retuning'})
        key = ['experiment', 'subset']
        columns = ['precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        ablation = main_metrics[key + columns].merge(reduced_metrics[key + columns],
            on=key, suffixes=('_with', '_without'))
        for column in columns:
            ablation[column + '_change'] = ablation[column + '_without'] - ablation[column + '_with']
        ablation.to_csv(output_root / 'results/metrics/logistic_regression_agent_ablation.csv', index=False)
    print('Logistic regression selection and evaluation completed:', output_root)


if __name__ == '__main__':
    main()
