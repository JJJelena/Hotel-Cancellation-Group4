"""A-v4 feature contract, C's unchanged evaluation API and RF utilities."""
from pathlib import Path
import os
import json, hashlib, platform
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve, auc,
    confusion_matrix, brier_score_loss, log_loss)
import evaluation as ev

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('HOTEL_OUTPUT_ROOT', str(ROOT))) / 'results'
for sub in ['audit','tables','metrics','predictions','models','figures','shap','tuning','cv_cache','supplementary']:
    (OUT/sub).mkdir(parents=True, exist_ok=True)
SEED, THRESHOLD = ev.SEED, ev.DEFAULT_THRESHOLD
GRID = [
    dict(max_depth=16, min_samples_leaf=5, max_features='sqrt'),
    dict(max_depth=24, min_samples_leaf=2, max_features='sqrt'),
    dict(max_depth=None, min_samples_leaf=2, max_features='sqrt'),
    dict(max_depth=16, min_samples_leaf=5, max_features=0.35),
    dict(max_depth=24, min_samples_leaf=2, max_features=0.35),
    dict(max_depth=None, min_samples_leaf=8, max_features='sqrt'),
]

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,
        default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def shared_data():
    data=ev.load_shared_data(ROOT)
    data['project_root'] = OUT.parent
    df=data['df']; features=data['features']
    assert len(df)==118564 and df.row_id.is_unique
    assert len(features)==24 and 'required_car_parking_spaces' not in features
    forbidden={'row_id','is_canceled','reservation_status','reservation_status_date',
        'booking_changes','assigned_room_type','days_in_waiting_list','company',
        'adr','deposit_type','customer_type','total_of_special_requests','arrival_season'}
    assert not forbidden.intersection(features)
    assert not df[features].isna().any().any()
    dates=pd.to_datetime(df.arrival_date_year.astype(str)+'-'+df.arrival_date_month+'-'+
        df.arrival_date_day_of_month.astype(str),format='%Y-%B-%d')
    date_map=pd.Series(dates.to_numpy(),index=df.row_id)
    for key in ['df','random_train','random_test','chrono_train','chrono_test']:
        data[key]['scheduled_arrival_date']=data[key].row_id.map(date_map)
    assert (len(data['random_train']),len(data['random_test']))==(94851,23713)
    assert (len(data['chrono_train']),len(data['chrono_test']))==(96428,22136)
    assert data['chrono_train'].scheduled_arrival_date.max()==pd.Timestamp('2017-04-30')
    assert data['chrono_test'].scheduled_arrival_date.min()==pd.Timestamp('2017-05-01')
    assert not hasattr(data['preprocessor'],'transformers_')
    return data

def load_inputs():
    d=shared_data()
    return d['df'],d['preprocessor'],d['features'],{
        'random':(d['random_train'],d['random_test']),
        'chronological':(d['chrono_train'],d['chrono_test']),
        'strict_chronological':(d['chrono_train'],d['chrono_test'])}

def classifier(params):
    return RandomForestClassifier(n_estimators=160,random_state=SEED,n_jobs=10,
        class_weight=None,**params)

def pipeline(template,params,drop=()):
    pre=clone(template)
    if drop:
        pre.transformers=[(name,clone(trans),[x for x in cols if x not in drop])
            for name,trans,cols in pre.transformers]
    return Pipeline([('preprocessing',pre),('model',classifier(params))])

def evaluate(y,p,threshold=THRESHOLD,weights=None):
    y,p=np.asarray(y),np.asarray(p)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    pred=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1],sample_weight=weights).ravel()
    prec,rec,_=precision_recall_curve(y,p,sample_weight=weights)
    return dict(n=len(y),positive_rate=float(np.average(y,weights=weights)),threshold=threshold,
        accuracy=accuracy_score(y,pred,sample_weight=weights),
        precision=precision_score(y,pred,zero_division=0,sample_weight=weights),
        recall=recall_score(y,pred,zero_division=0,sample_weight=weights),
        f1=f1_score(y,pred,zero_division=0,sample_weight=weights),
        roc_auc=roc_auc_score(y,p,sample_weight=weights) if len(np.unique(y))>1 else np.nan,
        pr_auc_ap=average_precision_score(y,p,sample_weight=weights),
        pr_auc_trapezoid=auc(rec,prec),brier=brier_score_loss(y,p,sample_weight=weights),
        log_loss=log_loss(y,p,labels=[0,1],sample_weight=weights),tn=tn,fp=fp,fn=fn,tp=tp)

def predictions(data,p,experiment,phase,model_name='RandomForest',fold=None):
    out=data[['row_id','hotel','scheduled_arrival_date']].copy()
    out['y_true']=data.is_canceled.to_numpy();out['cancel_probability']=p
    out['y_pred']=(np.asarray(p)>=THRESHOLD).astype(int)
    out['model_name'],out['experiment'],out['phase'],out['threshold']=model_name,experiment,phase,THRESHOLD
    if fold is not None:out['fold']=fold
    return out

def profile_hash(data,features):
    return pd.util.hash_pandas_object(data[features],index=False).astype(str)

def model_file(experiment):
    return OUT/'models'/({'random':'random_forest_main.joblib',
        'chronological':'rf_chronological.joblib','strict_chronological':'rf_strict_chronological.joblib'}[experiment])

def load_pipeline(experiment):
    obj=joblib.load(model_file(experiment))
    return obj['pipeline'] if isinstance(obj,dict) else obj
