"""Train XGBoost under the shared A-v4 five-fold evaluation contract.

Run: python src/train_xgboost.py --project-root .
Only training folds select parameters. Chronological transfer uses these same
parameters and is not a strictly independent temporal-selection experiment.
"""
from pathlib import Path
import argparse, json, sys, time, warnings, platform
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
LOCAL_DEPENDENCIES = HERE.parent / 'ml_packages'
if LOCAL_DEPENDENCIES.exists():
    sys.path.insert(0, str(LOCAL_DEPENDENCIES))
from xgboost import XGBClassifier
import xgboost, sklearn

CANDIDATES = [
    dict(max_depth=3, n_estimators=200, learning_rate=0.05),
    dict(max_depth=5, n_estimators=300, learning_rate=0.05),
    dict(max_depth=7, n_estimators=300, learning_rate=0.05),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=HERE.parent)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--support-dir', type=Path)
    args = parser.parse_args()
    source = args.project_root.resolve()
    output = (args.output_root or source/'runs'/'xgboost_reproduction').resolve()
    if output == source:
        parser.error('Choose a separate output root to preserve reference results.')
    for sub in ['metrics', 'predictions', 'models', 'tuning', 'audit']:
        (output/'results'/sub).mkdir(parents=True, exist_ok=True)
    if args.support_dir:
        sys.path.insert(0, str(args.support_dir.resolve()))
    sys.path.insert(0, str(source/'src'))
    import evaluation as ev
    from preprocessing_source import load_shared_data
    data = load_shared_data(source)
    tr, te = data['random_train'], data['random_test']
    features = data['features']
    y = tr.is_canceled.to_numpy()
    folds = data['cv_folds'].set_index('row_id').loc[tr.row_id, 'cv_fold'].to_numpy()
    oof = np.zeros((len(tr), len(CANDIDATES)))
    score_rows=[]
    started=time.time()
    for k in range(1, 6):
        fit, val = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        prep = clone(data['preprocessor'])
        X_fit=prep.fit_transform(tr.iloc[fit][features],y[fit])
        X_val=prep.transform(tr.iloc[val][features])
        for j, params in enumerate(CANDIDATES):
            model=classifier(params)
            model.fit(X_fit,y[fit])
            p=model.predict_proba(X_val)[:,1]
            oof[val,j]=p
            ap=average_precision_score(y[val],p)
            score_rows.append(dict(candidate=j+1, fold=k, average_precision=ap, **params))
            print(f'fold {k}/5 candidate {j+1}/3 AP={ap:.6f} elapsed={time.time()-started:.1f}s',flush=True)
    scores=pd.DataFrame(score_rows)
    scores.to_csv(output/'results/tuning/xgboost_cv_fold_metrics.csv',index=False)
    summary=scores.groupby('candidate',as_index=False).agg(mean_ap=('average_precision','mean'),std_ap=('average_precision','std'))
    for key in CANDIDATES[0]:
        summary[key]=[p[key] for p in CANDIDATES]
    summary.to_csv(output/'results/tuning/xgboost_cv_results.csv',index=False)
    winner=int(summary.sort_values(['mean_ap','candidate'],ascending=[False,True]).iloc[0].candidate)-1
    chosen=CANDIDATES[winner]
    print('Selected',winner+1,chosen,flush=True)
    parts=[prediction_table(tr,oof[:,winner],'random','validation',folds,ev)]
    pipe=ev.make_pipeline(data,classifier(chosen))
    pipe.fit(tr[features],tr.is_canceled)
    parts.append(prediction_table(te,pipe.predict_proba(te[features])[:,1],'random','test',None,ev))
    export_data=dict(data,project_root=output)
    ev.save_model_package(pipe,'xgboost','main',features,chosen,export_data)
    ctr,cte=data['chrono_train'],data['chrono_test']
    chrono=clone(pipe).fit(ctr[features],ctr.is_canceled)
    parts.append(prediction_table(cte,chrono.predict_proba(cte[features])[:,1],'chronological','test',None,ev))
    joblib.dump({'pipeline':chrono,'params':chosen,'selection_scope':'random training five-fold CV; temporal transfer only'},output/'results/models/xgboost_chronological.joblib',compress=3)
    preds=pd.concat(parts,ignore_index=True)
    metrics=ev.metrics_from_predictions(preds)
    ev.validate_outputs(metrics,preds,data)
    preds.to_csv(output/'results/predictions/xgboost_main_predictions.csv',index=False,float_format='%.17g')
    metrics.to_csv(output/'results/metrics/xgboost_main_metrics.csv',index=False,float_format='%.17g')
    note=dict(candidate_grid=CANDIDATES,selected_candidate=winner+1,selected_params=chosen,
        criterion='mean five-fold training average precision; tie: first candidate',
        cv='A-v4 fixed cv_folds.csv, five folds, seed 42',
        features=features,preprocessing='A-v4 source template: median + StandardScaler; most_frequent + OneHotEncoder(min_frequency=50,handle_unknown=ignore)',
        probability_export='full precision',runtime_seconds=time.time()-started,
        software=dict(python=platform.python_version(),sklearn=sklearn.__version__,xgboost=xgboost.__version__,pandas=pd.__version__,numpy=np.__version__),
        chronological_limitation='Hyperparameters selected on random training rows overlap the later temporal test period. This is transfer evaluation, not strict temporal tuning.',
        oof_limitation='Selected-model OOF probabilities reuse folds used for hyperparameter selection; OOF metrics are not nested-CV estimates.')
    (output/'results/audit/xgboost_protocol.json').write_text(json.dumps(note,indent=2),encoding='utf-8')
    print(metrics.to_string(index=False),flush=True)

def classifier(params):
    return XGBClassifier(**params,subsample=.85,colsample_bytree=.85,min_child_weight=1,
        reg_lambda=1,objective='binary:logistic',eval_metric='logloss',tree_method='hist',
        random_state=42,n_jobs=6)

def prediction_table(part,p,experiment,subset,folds,ev):
    table=pd.DataFrame(dict(row_id=part.row_id.to_numpy(),model='xgboost',variant='main',
        experiment=experiment,subset=subset,fold=-1 if folds is None else folds,
        hotel=part.hotel.to_numpy(),y_true=part.is_canceled.to_numpy(),y_prob=np.asarray(p,dtype=float)))
    return table[ev.PREDICTION_COLUMNS]

if __name__=='__main__':
    warnings.filterwarnings('ignore',category=UserWarning)
    main()
