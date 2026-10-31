"""Compare training-OOF threshold choices and sigmoid calibration on test.

No test labels are used for threshold or calibration fitting. Sigmoid
calibration is cross-fitted on training OOF probabilities for threshold choice;
a final sigmoid fitted to all training OOF pairs transforms random-test scores.
Selected-model OOF scores are not nested estimates of model-selection error.
"""
from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import joblib

def logit(p):
    p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--xgb-root',type=Path)
    parser.add_argument('--prediction-root',type=Path)
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args()
    source=args.project_root.resolve()
    out=(args.output_root or source/'runs'/'probability_reproduction').resolve()
    if out == source:
        parser.error('Choose a separate output root to preserve reference results.')
    base_root=(args.prediction_root or source).resolve()
    xroot=(args.xgb_root or base_root).resolve()
    sys.path.insert(0,str(source/'src'))
    import evaluation as ev
    for folder in ['tables','figures','models','predictions','audit']:
        (out/'results'/folder).mkdir(parents=True,exist_ok=True)
    tables=[]; final=[]; bins=[]; pred_tables=[]; oof_tables=[]
    fig,axes=plt.subplots(1,2,figsize=(9,3.4))
    axes[0].plot([0,1],[0,1],'--',color='gray',linewidth=1)
    colours={'logistic_regression':'#2b6f9c','random_forest':'#c67b32','xgboost':'#43805e'}
    names={'logistic_regression':'Logistic regression','random_forest':'Random forest','xgboost':'XGBoost'}
    expected_test=None; expected_val=None
    for name in names:
        root=xroot if name=='xgboost' else base_root
        pred=pd.read_csv(root/'results/predictions'/f'{name}_main_predictions.csv')
        val=pred.query("experiment == 'random' and subset == 'validation'").sort_values('row_id').reset_index(drop=True)
        test=pred.query("experiment == 'random' and subset == 'test'").sort_values('row_id').reset_index(drop=True)
        assert len(val)==94851 and len(test)==23713
        assert val.row_id.is_unique and test.row_id.is_unique
        if expected_test is None:
            expected_test=test[['row_id','y_true']]; expected_val=val[['row_id','y_true','fold']]
        else:
            pd.testing.assert_frame_equal(expected_test,test[['row_id','y_true']])
            pd.testing.assert_frame_equal(expected_val,val[['row_id','y_true','fold']])
        y=val.y_true.to_numpy(); p=val.y_prob.to_numpy(); folds=val.fold.to_numpy()
        crosscal=np.zeros(len(val))
        for fold in range(1,6):
            fit=folds!=fold; held=folds==fold
            cal=LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000).fit(logit(p[fit]),y[fit])
            crosscal[held]=cal.predict_proba(logit(p[held]))[:,1]
        calibrator=LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000).fit(logit(p),y)
        calibrated=calibrator.predict_proba(logit(test.y_prob))[:,1]
        joblib.dump(calibrator,out/'results/models'/f'{name}_sigmoid_calibrator.joblib')
        oof_tables.append(pd.DataFrame(dict(row_id=val.row_id,y_true=y,fold=folds,
            model=name,raw_probability=p,sigmoid_probability=crosscal)))
        for variant,vprob,tprob in [('raw',p,test.y_prob.to_numpy()),('sigmoid',crosscal,calibrated)]:
            rows=[]
            for threshold in np.round(np.arange(.05,.951,.05),2):
                row=dict(model=name,probability=variant,subset='validation',**ev.compute_metrics(y,vprob,float(threshold)))
                rows.append(row)
            table=pd.DataFrame(rows); tables.append(table)
            best=table.sort_values(['f1','threshold'],ascending=[False,True]).iloc[0]
            for rule,threshold in [('default',.5),('training_oof_max_f1',float(best.threshold))]:
                final.append(dict(model=name,probability=variant,selection_rule=rule,
                    validation_selected_threshold=float(best.threshold),validation_f1_at_selected=float(best.f1),
                    subset='random_test',**ev.compute_metrics(test.y_true,tprob,threshold),
                    log_loss=log_loss(test.y_true,tprob,labels=[0,1])))
            ptab=pd.DataFrame(dict(row_id=test.row_id,y_true=test.y_true,model=name,
                probability=variant,y_prob=tprob))
            pred_tables.append(ptab)
            # Equal-frequency bins evaluated only after the calibrator is fixed.
            group=pd.DataFrame(dict(y=test.y_true.to_numpy(),p=tprob))
            group['bin']=pd.qcut(group.p,q=10,labels=False,duplicates='drop')
            b=group.groupby('bin',as_index=False).agg(n=('y','size'),mean_probability=('p','mean'),observed_cancel_rate=('y','mean'))
            b['model']=name; b['probability']=variant; bins.append(b)
            if variant=='raw':
                axes[0].plot(b.mean_probability,b.observed_cancel_rate,'o-',markersize=3,label=names[name],color=colours[name])
                axes[1].plot(table.threshold,table.f1,label=names[name],color=colours[name])
    pd.concat(tables,ignore_index=True).to_csv(out/'results/tables/threshold_validation_grid.csv',index=False)
    pd.DataFrame(final).to_csv(out/'results/tables/threshold_calibration_test_comparison.csv',index=False)
    pd.concat(bins,ignore_index=True).to_csv(out/'results/tables/calibration_test_bins.csv',index=False)
    pd.concat(pred_tables,ignore_index=True).to_csv(out/'results/predictions/calibration_random_test_predictions.csv',index=False,float_format='%.17g')
    pd.concat(oof_tables,ignore_index=True).to_csv(out/'results/predictions/calibration_oof_predictions.csv',index=False,float_format='%.17g')
    axes[0].set(xlabel='Mean predicted probability',ylabel='Observed cancellation rate',title='Raw probabilities on random test')
    axes[1].set(xlabel='Decision threshold',ylabel='Training OOF F1',title='Threshold choice using training OOF')
    for ax in axes:
        ax.legend(fontsize=7,frameon=False); ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out/'results/figures/calibration_and_thresholds.png',dpi=250)
    fig.savefig(out/'results/figures/calibration_and_thresholds.pdf')
    info=dict(n_validation=94851,n_test=23713,folds=5,threshold_grid='0.05 to 0.95 in 0.05 steps',
        threshold_criterion='Maximum selected-model training OOF F1; ties use smaller threshold',
        sigmoid='Univariate logistic regression on clipped logit probabilities, C=1e6; cross-fitted across existing OOF folds for validation scores, fitted to all OOF pairs for test transformation',
        test_use='Final evaluation only; test labels are never used for fitting or threshold selection',
        limitations=['OOF probabilities come from hyperparameter-selected models and are not nested CV.',
            'Base OOF models share training rows, so cross-fitted calibration is a practical training diagnostic, not fully nested calibration validation.',
            'Sigmoid mapping is estimated on OOF score distributions and applied to the full-training model scores.',
            'Reported test results compare prespecified methods; no test-optimal threshold is selected.'])
    (out/'results/audit/threshold_calibration_protocol.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(pd.DataFrame(final)[['model','probability','selection_rule','threshold','f1','precision','recall','brier','log_loss']].to_string(index=False))

if __name__=='__main__':main()
