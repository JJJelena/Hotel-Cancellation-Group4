"""Training-only RF selection, C-standard exports and a separate strict time check."""
from common import *
import argparse,time,importlib.metadata
from sklearn.model_selection import TimeSeriesSplit
from reproduction_metadata import write_run_environment

def prepare(stage='forest'):
    d=shared_data();df=d['df'];features=d['features']
    files=[ROOT/'data/hotel_bookings_clean.csv',ROOT/'data/data_dictionary.csv',
        ROOT/'src/preprocessing_source.py',ROOT/'src/evaluation.py']
    files+=sorted((ROOT/'data/splits').glob('*.csv'))
    overlap=set(d['random_train'].row_id)&set(d['chrono_test'].row_id)
    protocol=dict(data_version='A-v4',feature_count=24,features=features,seed=SEED,trees=160,
        candidates=GRID,selection='highest mean training-only CV average precision; ties use candidate order',
        threshold=THRESHOLD,class_weight=None,
        random_cv='A-v4 cv_folds.csv, five stratified folds, shuffle=True, random_state=42',
        team_chronological='C protocol: same random-CV-selected settings, refit chronological train; not a fully independent future holdout',
        strict_chronological='independent parameter selection using five expanding windows of unique scheduled arrival dates within chronological TRAIN only',
        random_train_chronological_test_overlap=len(overlap),
        standard_export='shared C metric API; source-built A-v4 preprocessing; metrics recomputed from six-decimal probability export',
        tuning_scores='full precision mean AP; selected-parameter OOF is not nested CV',
        shap='RF candidate; 400 random test rows seed=2026; 100 random train background rows seed=42; exact class-1 interventional probability SHAP',
        cases='lowest and highest scores in explanation sample; outcomes not used in selection',
        agent_sensitivity='drop agent with the same selected settings; diagnostic only; not retuned',
        duplicate_sensitivity='unseen 24-feature profiles, plus equal total weight per unseen profile',
        parking_exclusion='team precaution about initial-booking availability, not proof of leakage or causation',
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in files})
    p=OUT/'audit/experiment_protocol.json'
    if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==protocol,'Frozen protocol changed'
    else:write_json(p,protocol)
    versions={}
    for name in ['numpy','pandas','scikit-learn','shap','matplotlib','scipy','joblib','nbformat']:
        try:
            versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            if name!='shap':
                raise
            versions[name]=None
    software=dict(python=platform.python_version(),packages=versions)
    version_path=OUT/'audit/software_versions.json'
    if version_path.exists():
        assert json.loads(version_path.read_text())==software,'Software changed; use a fresh results directory instead of cached fits'
    else:write_json(version_path,software)
    write_run_environment(OUT/'audit/rf_run_environment.json',project_root=ROOT,
        stage='random_forest_'+stage,seed=SEED,
        thread_settings={'random_forest_n_jobs':10},
        algorithm_settings={'n_estimators':160,'candidate_grid':GRID,
            'class_weight':None,'decision_threshold':THRESHOLD,
            'random_cv':'A-v4 fixed five stratified folds',
            'strict_time_cv':'five expanding windows within earlier-arrival training'})
    rows=[]
    for exp,tkey,vkey in [('random','random_train','random_test'),('chronological','chrono_train','chrono_test'),('strict_chronological','chrono_train','chrono_test')]:
        tr,te=d[tkey],d[vkey];htr,hte=profile_hash(tr,features),profile_hash(te,features);seen=hte.isin(set(htr))
        rows.append(dict(experiment=exp,n_train=len(tr),n_test=len(te),train_positive_rate=tr.is_canceled.mean(),
            test_positive_rate=te.is_canceled.mean(),overlapping_test_rows=int(seen.sum()),
            overlapping_test_fraction=float(seen.mean()),unique_train_profiles=htr.nunique(),unique_test_profiles=hte.nunique()))
    pd.DataFrame(rows).to_csv(OUT/'audit/data_audit.csv',index=False)
    raw=pd.read_csv(ROOT/'data/hotel_bookings.csv');parking=[]
    for name,frame in [('raw',raw),('clean',df)]:
        for condition,mask in [('equals_1',frame.required_car_parking_spaces.eq(1)),('greater_than_0',frame.required_car_parking_spaces.gt(0))]:
            parking.append(dict(dataset=name,condition=condition,n=int(mask.sum()),cancellations=int(frame.loc[mask,'is_canceled'].sum())))
    pd.DataFrame(parking).to_csv(OUT/'audit/parking_diagnostic.csv',index=False)
    fold=d['random_train'][['row_id']].merge(d['cv_folds'],on='row_id',validate='one_to_one')
    fold['profile']=profile_hash(d['random_train'],features).to_numpy()
    cross=fold.groupby('profile').cv_fold.nunique().gt(1)
    write_json(OUT/'audit/feature_contract.json',dict(input_features=features,parking_used=False,
        duplicate_profiles_across_cv=int(cross.sum()),rows_in_cross_fold_profiles=int(fold.profile.isin(cross[cross].index).sum()),
        template_fitted=False,initial_booking_availability='not established by the retrospective source dataset'))
    write_json(OUT/'audit/time_protocol_overlap.json',dict(random_training_rows_in_chronological_test=len(overlap),
        chronological_test_rows=len(d['chrono_test']),team_result_scope=protocol['team_chronological'],
        strict_result_scope=protocol['strict_chronological']))
    print('AUDIT PASS: A-v4 24 features, fixed folds verified, time overlap',len(overlap),flush=True)
    return d

def folds_for(data,experiment):
    if experiment=='random':return list(ev.get_cv().split(data,data.is_canceled))
    dates=np.sort(data.scheduled_arrival_date.unique());folds=[]
    for trd,vad in TimeSeriesSplit(n_splits=5).split(dates):
        tr=np.flatnonzero(data.scheduled_arrival_date.isin(dates[trd]));va=np.flatnonzero(data.scheduled_arrival_date.isin(dates[vad]))
        assert data.iloc[tr].scheduled_arrival_date.max()<data.iloc[va].scheduled_arrival_date.min()
        folds.append((tr,va))
    return folds

def tune(d,experiment):
    train=d['random_train' if experiment=='random' else 'chrono_train'];features=d['features']
    folds=folds_for(train,experiment);assignments=[]
    for f,(tr,va) in enumerate(folds,start=1):
        part=train.iloc[va][['row_id','scheduled_arrival_date']].copy();part['fold']=f
        part['training_end']=str(train.iloc[tr].scheduled_arrival_date.max().date());part['training_n']=len(tr)
        assignments.append(part)
    pd.concat(assignments).to_csv(OUT/f'audit/{experiment}_validation_folds.csv',index=False)
    stats=[];fingerprint=sha(OUT/'audit/experiment_protocol.json')
    for cid,params in enumerate(GRID):
        for fid,(tr,va) in enumerate(folds,start=1):
            cache=OUT/f'cv_cache/{experiment}_c{cid}_f{fid}.npz';started=time.perf_counter()
            if cache.exists():
                stored=np.load(cache);assert str(stored['protocol_sha256'])==fingerprint
                assert np.array_equal(stored['row_id'],train.iloc[va].row_id)
                p=stored['p'];seconds=float(stored['seconds'])
            else:
                est=ev.make_pipeline(d,classifier(params));est.fit(train.iloc[tr][features],train.iloc[tr].is_canceled)
                p=est.predict_proba(train.iloc[va][features])[:,1];seconds=time.perf_counter()-started
                np.savez_compressed(cache,p=p,row_id=train.iloc[va].row_id.to_numpy(),seconds=seconds,protocol_sha256=fingerprint)
                del est
            stat=dict(candidate=cid,fold=fid,fit_predict_seconds=seconds,**params,**evaluate(train.iloc[va].is_canceled,p))
            stats.append(stat)
            print(f'{experiment} candidate {cid+1}/6 fold {fid}/5 AP={stat["pr_auc_ap"]:.5f} {seconds:.1f}s',flush=True)
            pd.DataFrame(stats).to_csv(OUT/f'tuning/{experiment}_rf_cv_fold_metrics.csv',index=False)
    frame=pd.DataFrame(stats);summary=frame.groupby('candidate').agg({m:['mean','std'] for m in ['pr_auc_ap','roc_auc','f1','precision','recall','brier']})
    summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index()
    summary['parameters']=[json.dumps(GRID[int(i)]) for i in summary.candidate]
    summary.to_csv(OUT/f'tuning/{experiment}_rf_cv_summary.csv',index=False)
    if experiment=='random':summary.to_csv(OUT/'tuning/random_forest_cv_results.csv',index=False)
    best=int(summary.sort_values(['pr_auc_ap_mean','candidate'],ascending=[False,True]).iloc[0].candidate)
    selected=dict(candidate=best,parameters=GRID[best],criterion='highest training CV mean average precision',
        cv_ap=float(summary.loc[summary.candidate.eq(best),'pr_auc_ap_mean'].iloc[0]),
        scope='random train with A fixed folds' if experiment=='random' else 'chronological train only, expanding date windows')
    write_json(OUT/f'audit/{experiment}_selected_parameters.json',selected)
    print('SELECTED',experiment,selected,flush=True)
    return selected

def standard(d):
    selected=json.loads((OUT/'audit/random_selected_parameters.json').read_text())
    params=selected['parameters'];pipe=ev.make_pipeline(d,classifier(params))
    saved_paths=[OUT/'metrics/random_forest_main_metrics.csv',OUT/'predictions/random_forest_main_predictions.csv',OUT/'models/random_forest_main.joblib']
    reuse_standard=all(p.exists() for p in saved_paths)
    if reuse_standard:
        metrics,preds=pd.read_csv(saved_paths[0]),pd.read_csv(saved_paths[1])
        ev.validate_outputs(metrics,preds,d)
        saved_model=joblib.load(saved_paths[2])
        assert saved_model['input_features']==d['features']
        assert saved_model['pipeline'].named_steps['model'].get_params()==classifier(params).get_params()
        recomputed=ev.metrics_from_predictions(preds).set_index(['model','variant','experiment','subset']).sort_index()
        actual=metrics.set_index(['model','variant','experiment','subset']).sort_index()
        assert np.allclose(actual.to_numpy(float),recomputed.to_numpy(float),atol=1e-12,rtol=1e-12)
        print('Reusing validated completed C-standard exports.',flush=True)
        del saved_model
    else:
        metrics,preds=ev.run_standard_experiment(pipe,model='random_forest',data=d,params={
            **classifier(params).get_params(), 'selection_cv_mean_ap':selected['cv_ap'],'data_version':'A-v4'})
    # Keep an additional full-precision export for numeric checks and SHAP.
    pkg=joblib.load(OUT/'models/random_forest_main.joblib')
    if not reuse_standard or saved_paths[2].stat().st_size>100*1024**2:
        joblib.dump(pkg,OUT/'models/random_forest_main.joblib',compress=3)
    fitted=pkg['pipeline'];features=d['features']
    tr,te=d['random_train'],d['random_test'];oof=np.zeros(len(tr));best=selected['candidate']
    for fid,(_,va) in enumerate(folds_for(tr,'random'),start=1):oof[va]=np.load(OUT/f'cv_cache/random_c{best}_f{fid}.npz')['p']
    exported=preds[(preds.experiment=='random')&(preds.subset=='validation')]
    delta=np.abs(oof-exported.y_prob.to_numpy())
    assert delta.max()<=5.01e-7,'OOF export differs beyond six-decimal rounding precision'
    write_json(OUT/'audit/probability_export_precision.json',dict(decimals=6,maximum_oof_raw_to_export_error=float(delta.max()),
        tolerance=5.01e-7,note='Parallel tree summation can differ at a decimal half-unit boundary. Compare raw scores with the six-decimal export at its half-unit precision.'))
    predictions(tr,oof,'random','validation',fold=d['cv_folds'].set_index('row_id').loc[tr.row_id,'cv_fold'].to_numpy()).to_csv(OUT/'supplementary/random_validation_full_precision.csv',index=False)
    predictions(te,fitted.predict_proba(te[features])[:,1],'random','test').to_csv(OUT/'supplementary/random_test_full_precision.csv',index=False)
    chrono=ev.make_pipeline(d,classifier(params)).fit(d['chrono_train'][features],d['chrono_train'].is_canceled)
    joblib.dump(chrono,OUT/'models/rf_chronological.joblib',compress=3)
    p=chrono.predict_proba(d['chrono_test'][features])[:,1]
    assert np.max(np.abs(p-preds.loc[preds.experiment.eq('chronological'),'y_prob'].to_numpy()))<=5.01e-7
    predictions(d['chrono_test'],p,'chronological','test').to_csv(OUT/'supplementary/chronological_test_full_precision.csv',index=False)
    write_json(OUT/'models/rf_chronological_metadata.json',dict(input_features=features,parameters=classifier(params).get_params(),
        trained_on='chrono_train through 2017-04-30',selection_source='random training CV, shared C protocol',
        independent_time_selection=False,model_sha256=sha(OUT/'models/rf_chronological.joblib')))
    print('STANDARD RESULTS',metrics.to_dict('records'),flush=True)

def strict(d):
    selected=json.loads((OUT/'audit/strict_chronological_selected_parameters.json').read_text())
    tr,te=d['chrono_train'],d['chrono_test'];features=d['features'];parts=[]
    for fid,(_,va) in enumerate(folds_for(tr,'strict_chronological'),start=1):
        p=np.load(OUT/f'cv_cache/strict_chronological_c{selected["candidate"]}_f{fid}.npz')['p']
        parts.append(predictions(tr.iloc[va],p,'strict_chronological','validation',fold=fid))
    val=pd.concat(parts,ignore_index=True)
    est=ev.make_pipeline(d,classifier(selected['parameters'])).fit(tr[features],tr.is_canceled)
    p=est.predict_proba(te[features])[:,1];test=predictions(te,p,'strict_chronological','test')
    joblib.dump(est,OUT/'models/rf_strict_chronological.joblib',compress=3)
    val.to_csv(OUT/'supplementary/strict_chronological_validation_predictions.csv',index=False)
    test.to_csv(OUT/'supplementary/strict_chronological_test_predictions.csv',index=False)
    rows=[dict(model_name='RandomForest',experiment='strict_chronological',phase=phase,**evaluate(f.y_true,f.cancel_probability)) for phase,f in [('validation',val),('test',test)]]
    pd.DataFrame(rows).to_csv(OUT/'supplementary/strict_chronological_metrics.csv',index=False)
    write_json(OUT/'models/rf_strict_chronological_metadata.json',dict(input_features=features,
        processed_features=est.named_steps['preprocessing'].get_feature_names_out().tolist(),
        parameters=classifier(selected['parameters']).get_params(),selection=selected,
        trained_on='chronological train only, scheduled arrivals through 2017-04-30',
        model_sha256=sha(OUT/'models/rf_strict_chronological.joblib'),independent_time_selection=True))
    print('STRICT RESULTS',rows,flush=True)

def agent_sensitivity(d):
    selected=json.loads((OUT/'audit/random_selected_parameters.json').read_text())
    pipe=ev.make_pipeline(d,classifier(selected['parameters']),drop_features=['agent'])
    metrics,preds=ev.run_standard_experiment(pipe,model='random_forest',variant='no_agent',data=d,
        drop_features=['agent'],params={**classifier(selected['parameters']).get_params(),
            'selection':'main-model settings reused, no retuning','data_version':'A-v4'})
    package_path=OUT/'models/random_forest_no_agent.joblib'
    joblib.dump(joblib.load(package_path),package_path,compress=3)
    pd.DataFrame([dict(experiment=r['experiment'],phase=r['subset'],removed_feature='agent',
        **evaluate(g.y_true,g.y_prob)) for r in metrics.to_dict('records')
        for g in [preds[(preds.experiment==r['experiment'])&(preds.subset==r['subset'])]]]).to_csv(OUT/'tables/feature_sensitivity.csv',index=False)

def summaries(d,export_thresholds=True):
    path=OUT/'predictions/random_forest_main_predictions.csv'
    if not path.exists():return
    frame=pd.read_csv(path);legacy=[]
    for (exp,phase),g in frame.groupby(['experiment','subset'],sort=False):
        key='random_train' if phase=='validation' else 'random_test' if exp=='random' else 'chrono_test'
        data=d[key];aligned=g.set_index('row_id').loc[data.row_id]
        predictions(data,aligned.y_prob.to_numpy(),exp,phase,fold=aligned.fold.to_numpy()).to_csv(OUT/f'supplementary/{exp}_{phase}_predictions.csv',index=False)
        legacy.append(dict(model_name='RandomForest',experiment=exp,phase=phase,**evaluate(aligned.y_true,aligned.y_prob)))
    strictpath=OUT/'supplementary/strict_chronological_metrics.csv'
    if strictpath.exists():legacy.extend(pd.read_csv(strictpath).to_dict('records'))
    for exp,te in [('random',d['random_test']),('chronological',d['chrono_test'])]:
        legacy.append(dict(model_name='DummyMajority_check',experiment=exp,phase='test',**evaluate(te.is_canceled,np.zeros(len(te)))))
    pd.DataFrame(legacy).to_csv(OUT/'tables/metrics.csv',index=False)
    # C files are inputs to comparison, not results recomputed by D.
    files=['dummy_main_metrics.csv','logistic_regression_main_metrics.csv','logistic_regression_no_agent_metrics.csv','random_forest_main_metrics.csv']
    available=[OUT/'metrics'/name for name in files if (OUT/'metrics'/name).is_file()]
    pd.concat([pd.read_csv(path) for path in available],ignore_index=True).to_csv(OUT/'tables/team_available_model_comparison.csv',index=False)
    overlap=[];subgroups=[];thresholds=[]
    for exp,tkey,vkey in [('random','random_train','random_test'),('chronological','chrono_train','chrono_test'),('strict_chronological','chrono_train','chrono_test')]:
        tr,te=d[tkey],d[vkey]
        pp=OUT/f'supplementary/{exp}_test_predictions.csv'
        if not pp.exists():continue
        pred=pd.read_csv(pp).set_index('row_id').loc[te.row_id];p=pred.cancel_probability.to_numpy()
        htr,hte=profile_hash(tr,d['features']),profile_hash(te,d['features']);seen=hte.isin(set(htr)).to_numpy()
        flags=te[['row_id']].copy();flags['seen_in_train']=seen;flags['feature_profile_hash']=hte
        flags.to_csv(OUT/f'audit/{exp}_test_overlap_flags.csv',index=False)
        for label,mask in [('seen_profile',seen),('unseen_profile',~seen)]:
            if mask.any():overlap.append(dict(experiment=exp,sensitivity=label,evaluation_status='evaluated',**evaluate(te.loc[mask,'is_canceled'],p[mask])))
            else:overlap.append(dict(experiment=exp,sensitivity=label,n=0,evaluation_status='not_applicable_no_rows'))
        h=hte[~seen];w=(1/h.map(h.value_counts())).to_numpy()
        overlap.append(dict(experiment=exp,sensitivity='unseen_profile_equal_weight',**evaluate(te.loc[~seen,'is_canceled'],p[~seen],weights=w)))
        for hotel in sorted(te.hotel.unique()):
            mask=te.hotel.eq(hotel).to_numpy();subgroups.append(dict(experiment=exp,group=hotel,**evaluate(te.loc[mask,'is_canceled'],p[mask])))
        for month in sorted(te.scheduled_arrival_date.dt.to_period('M').astype(str).unique()):
            mask=te.scheduled_arrival_date.dt.to_period('M').astype(str).eq(month).to_numpy()
            subgroups.append(dict(experiment=exp,group=month,**evaluate(te.loc[mask,'is_canceled'],p[mask])))
    val=frame[(frame.experiment=='random')&(frame.subset=='validation')]
    for threshold in [.3,.4,.5,.6,.7]:thresholds.append(dict(experiment='random',phase='validation',**evaluate(val.y_true,val.y_prob,threshold)))
    pd.DataFrame(overlap).to_csv(OUT/'tables/duplicate_sensitivity.csv',index=False)
    pd.DataFrame(subgroups).to_csv(OUT/'tables/subgroup_metrics.csv',index=False)
    if export_thresholds:
        pd.DataFrame(thresholds).to_csv(OUT/'tables/validation_thresholds_for_E.csv',index=False)

if __name__=='__main__':
    if OUT.parent.resolve() == ROOT.resolve():
        raise SystemExit('Use python src/run_project.py --stage forest --output-root runs/reproduction to preserve reference results.')
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['audit','random_tuning','standard','strict_tuning','strict','agent','summary','all'],default='audit')
    args=parser.parse_args();d=prepare(args.stage)
    if args.stage in ['random_tuning','all']:tune(d,'random')
    if args.stage in ['standard','all']:standard(d)
    if args.stage in ['strict_tuning','all']:tune(d,'strict_chronological')
    if args.stage in ['strict','all']:strict(d)
    if args.stage in ['agent','all']:agent_sensitivity(d)
    if args.stage in ['summary','standard','strict','agent','all']:summaries(d)
