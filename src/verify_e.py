"""Independently recompute the E metric, threshold and calibration exports.

Run from a complete project: python src/verify_e.py --project-root .
"""
from pathlib import Path
import argparse,json,sys
import joblib,numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

def logit(p):
    p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)

def close(a,b,message,atol=1e-10):
    assert np.allclose(np.asarray(a,dtype=float),np.asarray(b,dtype=float),atol=atol,rtol=0),message

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--e-root',type=Path)
    parser.add_argument('--prediction-root',type=Path,
        help='LR/RF prediction root; defaults to project for the original E-only workflow.')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    project=args.project_root.resolve(); eroot=(args.e_root or project).resolve()
    prediction_root=(args.prediction_root or project).resolve()
    sys.path.insert(0,str(project/'src'))
    import evaluation as ev
    results=eroot/'results'; checks={}
    pred=pd.read_csv(results/'predictions/xgboost_main_predictions.csv')
    recorded=pd.read_csv(results/'metrics/xgboost_main_metrics.csv')
    recomputed=ev.metrics_from_predictions(pred)
    assert recomputed.shape==recorded.shape
    for col in recorded:
        if pd.api.types.is_numeric_dtype(recorded[col]):
            close(recomputed[col],recorded[col],f'XGB metrics {col}')
        else:
            assert recomputed[col].equals(recorded[col]),col
    checks['xgboost_metric_export']='PASS: all three subsets recomputed from probabilities'
    summary=pd.read_csv(results/'tuning/xgboost_cv_results.csv')
    foldscores=pd.read_csv(results/'tuning/xgboost_cv_fold_metrics.csv')
    means=foldscores.groupby('candidate').average_precision.mean()
    close(summary.set_index('candidate').mean_ap,means,'mean AP')
    winner=int(summary.sort_values(['mean_ap','candidate'],ascending=[False,True]).iloc[0].candidate)
    protocol=json.loads((results/'audit/xgboost_protocol.json').read_text(encoding='utf-8'))
    assert winner==protocol['selected_candidate']
    checks['xgboost_training_selection']='PASS: mean AP selection matches fixed 5-fold scores'
    exported_test=pd.read_csv(results/'predictions/calibration_random_test_predictions.csv')
    exported_oof=pd.read_csv(results/'predictions/calibration_oof_predictions.csv')
    table=pd.read_csv(results/'tables/threshold_calibration_test_comparison.csv')
    grid=pd.read_csv(results/'tables/threshold_validation_grid.csv')
    dataset=pd.read_csv(project/'data/hotel_bookings_clean.csv').set_index('row_id')
    test_ids=set(pd.read_csv(project/'data/splits/test_ids.csv').row_id)
    train_ids=set(pd.read_csv(project/'data/splits/train_ids.csv').row_id)
    fixed_folds=pd.read_csv(project/'data/splits/cv_folds.csv').set_index('row_id').cv_fold
    for name in ['logistic_regression','random_forest','xgboost']:
        source=eroot if name=='xgboost' else prediction_root
        base=pd.read_csv(source/'results/predictions'/f'{name}_main_predictions.csv')
        raw_test=base.query("experiment == 'random' and subset == 'test'").sort_values('row_id').reset_index(drop=True)
        raw_oof=base.query("experiment == 'random' and subset == 'validation'").sort_values('row_id').reset_index(drop=True)
        saved_oof=exported_oof[exported_oof.model.eq(name)].sort_values('row_id').reset_index(drop=True)
        assert set(saved_oof.row_id)==train_ids and saved_oof.row_id.is_unique
        assert saved_oof.row_id.equals(raw_oof.row_id)
        close(saved_oof.y_true,dataset.loc[saved_oof.row_id,'is_canceled'],'OOF labels',0)
        close(saved_oof.fold,fixed_folds.loc[saved_oof.row_id],'OOF folds',0)
        close(saved_oof.raw_probability,raw_oof.y_prob,'Raw OOF scores')
        # Refit each univariate cross-fitted calibrator independently.
        calculated=np.zeros(len(saved_oof))
        for fold in range(1,6):
            fit=saved_oof.fold.to_numpy()!=fold; held=~fit
            m=LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000).fit(logit(raw_oof.y_prob.to_numpy()[fit]),raw_oof.y_true.to_numpy()[fit])
            calculated[held]=m.predict_proba(logit(raw_oof.y_prob.to_numpy()[held]))[:,1]
        close(calculated,saved_oof.sigmoid_probability,'Cross-fitted sigmoid OOF')
        calibrator=joblib.load(results/'models'/f'{name}_sigmoid_calibrator.joblib')
        refit=LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000).fit(logit(raw_oof.y_prob),raw_oof.y_true)
        close(calibrator.coef_,refit.coef_,'Final sigmoid refit coefficients')
        close(calibrator.intercept_,refit.intercept_,'Final sigmoid refit intercept')
        transformed=calibrator.predict_proba(logit(raw_test.y_prob))[:,1]
        for variant,column in [('raw','raw_probability'),('sigmoid','sigmoid_probability')]:
            tp=exported_test[(exported_test.model.eq(name))&(exported_test.probability.eq(variant))].sort_values('row_id').reset_index(drop=True)
            assert set(tp.row_id)==test_ids and tp.row_id.is_unique
            assert tp.row_id.equals(raw_test.row_id)
            close(tp.y_true,dataset.loc[tp.row_id,'is_canceled'],'Test labels',0)
            close(tp.y_prob,raw_test.y_prob if variant=='raw' else transformed,'Calibrator transform matches test export')
            g=grid[(grid.model.eq(name))&(grid.probability.eq(variant))].sort_values('threshold').reset_index(drop=True)
            close(g.threshold,np.round(np.arange(.05,.951,.05),2),'Threshold grid')
            for row in g.itertuples():
                m=ev.compute_metrics(saved_oof.y_true,saved_oof[column],float(row.threshold))
                for key,value in m.items():close(getattr(row,key),value,f'Validation grid {name}/{variant}/{key}')
            best=g.sort_values(['f1','threshold'],ascending=[False,True]).iloc[0]
            selections=table[(table.model.eq(name))&(table.probability.eq(variant))]
            assert len(selections)==2
            for row in selections.itertuples():
                close(row.validation_selected_threshold,best.threshold,'Selected threshold from OOF')
                close(row.validation_f1_at_selected,best.f1,'Selected threshold validation F1')
                wanted=.5 if row.selection_rule=='default' else best.threshold
                close(row.threshold,wanted,'Applied threshold rule')
                m=ev.compute_metrics(tp.y_true,tp.y_prob,float(row.threshold))
                m['log_loss']=log_loss(tp.y_true,tp.y_prob,labels=[0,1])
                for key,value in m.items():close(getattr(row,key),value,f'Test table {name}/{variant}/{key}')
        checks[name]='PASS: row IDs/labels/folds, 38 OOF grid rows, tie rule, sigmoid OOF refits, final calibrator refit and test transforms, four final test rows'
    checks['status']='PASS'
    checks['scope']='Independent recomputation of exported results; does not convert selected-model OOF or shared temporal transfer into strict nested/temporal validation.'
    path=args.output or project/'runs/verification/e_verification.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
