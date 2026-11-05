"""Exact class-1 probability SHAP with raw-variable aggregation and additivity gates."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from shap_compat import rf_probability_explainer, POLICY

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'savefig.dpi':180})

def transformed_groups(pre, features):
    names=pre.get_feature_names_out().tolist()
    mapping=[]
    for name in names:
        short=name.split('__',1)[1]
        # Longest prefix prevents arrival_date_year / arrival_date_month collisions.
        matches=[f for f in features if short == f or short.startswith(f+'_')]
        assert matches, name
        mapping.append(max(matches,key=len))
    return names,mapping

def run(experiment='random',model_path=None):
    _,_,features,splits=load_inputs()
    train,test=splits[experiment]
    path=Path(model_path) if model_path else model_file(experiment)
    est=joblib.load(path)
    if isinstance(est,dict):
        assert est['input_features']==features
        est=est['pipeline']
    assert list(est.named_steps)==['preprocessing','model']
    pre,model=est.named_steps['preprocessing'],est.named_steps['model']
    pos=int(np.flatnonzero(model.classes_==1)[0])
    bg=train.sample(n=100,random_state=42)
    sample=test.sample(n=min(400,len(test)),random_state=2026).copy()
    def dense(x):return x.toarray() if hasattr(x,'toarray') else np.asarray(x)
    background=dense(pre.transform(bg[features])).astype(np.float32)
    x=dense(pre.transform(sample[features])).astype(np.float32)
    names,mapping=transformed_groups(pre,features)
    assert len(names)==x.shape[1]
    cache=OUT/f'shap/{experiment}_exact_shap.npz'
    cache_valid=False
    if cache.exists():
        stored=np.load(cache)
        assert np.array_equal(stored['row_id'],sample.row_id)
        assert str(stored['model_sha256']) == sha(path)
        assert np.array_equal(stored['background_row_id'],bg.row_id)
        cache_valid='numeric_policy' in stored.files and str(stored['numeric_policy'])==POLICY
    if cache_valid:
        sv,base=stored['values'],float(stored['base'])
        numeric_meta=json.loads(str(stored['numeric_metadata']))
    else:
        explainer,numeric_meta=rf_probability_explainer(model,background)
        raw=explainer.shap_values(x,approximate=False,check_additivity=True)
        sv=raw[:,:,pos] if isinstance(raw,np.ndarray) and raw.ndim==3 else raw[pos]
        base=float(np.asarray(explainer.expected_value).reshape(-1)[pos])
    p=est.predict_proba(sample[features])[:,pos]
    err=np.abs(base+sv.sum(axis=1)-p)
    assert err.max()<1e-5, f'SHAP not additive: {err.max()}'
    if not cache_valid:
        np.savez_compressed(cache,values=sv,base=base,row_id=sample.row_id.to_numpy(),
            model_sha256=sha(path),background_row_id=bg.row_id.to_numpy(),numeric_policy=POLICY,
            numeric_metadata=json.dumps(numeric_meta))
    grouped=np.column_stack([sv[:,np.array(mapping)==f].sum(axis=1) for f in features])
    assert np.max(np.abs(grouped.sum(axis=1)-sv.sum(axis=1)))<1e-10
    grouped_frame=pd.DataFrame(grouped,columns=features)
    grouped_frame.insert(0,'row_id',sample.row_id.to_numpy())
    grouped_frame.to_csv(OUT/f'shap/{experiment}_grouped_shap_values.csv',index=False)
    sample.to_csv(OUT/f'shap/{experiment}_explained_rows.csv',index=False)
    rank=pd.DataFrame({'feature':features,'mean_abs_shap':np.abs(grouped).mean(axis=0),
        'mean_signed_shap':grouped.mean(axis=0)})
    rank=rank.sort_values('mean_abs_shap',ascending=False)
    rank.to_csv(OUT/f'tables/{experiment}_shap_importance.csv',index=False)
    pd.DataFrame({'encoded_feature':names,'original_feature':mapping}).to_csv(OUT/f'shap/{experiment}_feature_mapping.csv',index=False)
    # Bootstrap describes sensitivity to the 400 explained rows, not model uncertainty.
    rng=np.random.default_rng(42)
    boot=np.stack([np.abs(grouped[rng.integers(0,len(sample),len(sample))]).mean(0) for _ in range(300)])
    stability=pd.DataFrame({'feature':features,'sample_bootstrap_low':np.quantile(boot,.025,axis=0),
        'sample_bootstrap_high':np.quantile(boot,.975,axis=0)})
    stability.to_csv(OUT/f'tables/{experiment}_shap_sample_stability.csv',index=False)
    top=rank.head(12).iloc[::-1]
    fig,ax=plt.subplots(figsize=(8.6,5.4))
    ax.barh(top.feature,top.mean_abs_shap*100,color='#167d9a')
    ax.set_xlabel('Mean absolute grouped SHAP value (probability percentage points)')
    ax.set_title('Random forest: global predictive importance',loc='left',weight='bold')
    ax.text(0,-.19,'400 held-out bookings, 100 training background rows. One-hot contributions summed before |SHAP|.',
        transform=ax.transAxes,fontsize=8.5)
    fig.tight_layout();fig.savefig(OUT/f'figures/{experiment}_shap_global.png',bbox_inches='tight');plt.close(fig)
    shap.summary_plot(sv,x,feature_names=names,max_display=15,show=False)
    plt.gcf().set_size_inches(9,6)
    plt.title('Encoded-feature SHAP distribution (class 1 probability)')
    plt.tight_layout();plt.savefig(OUT/f'figures/{experiment}_shap_beeswarm.png',bbox_inches='tight');plt.close()
    cases=[]
    for label,i in [('low',int(np.argmin(p))),('high',int(np.argmax(p)))]:
        contributions=pd.DataFrame({'feature':features,'shap':grouped[i],
            'observed_value':[sample.iloc[i][f] for f in features]})
        contributions=contributions.reindex(contributions.shap.abs().sort_values(ascending=False).index)
        case=dict(case=label,row_id=int(sample.iloc[i].row_id),y_true=int(sample.iloc[i].is_canceled),
            probability=float(p[i]),base_probability=base,sum_shap=float(grouped[i].sum()),
            reconstruction_error=float(err[i]),hotel=sample.iloc[i].hotel,
            scheduled_arrival_date=str(sample.iloc[i].scheduled_arrival_date.date()),
            contributions=contributions.to_dict('records'))
        cases.append(case)
        explanation=shap.Explanation(values=grouped[i],base_values=base,
            data=np.array([str(sample.iloc[i][f]) for f in features],dtype=object),feature_names=features)
        shap.plots.waterfall(explanation,max_display=9,show=False)
        plt.gcf().set_size_inches(9,5.5)
        plt.title(f'{label.capitalize()} score example: row_id {case["row_id"]}, p(cancel)={p[i]:.3f}',fontsize=13)
        plt.tight_layout();plt.savefig(OUT/f'figures/{experiment}_case_{label}.png',bbox_inches='tight');plt.close()
    # A continuous feature already in the pre-registered feature set, in original units.
    f='lead_time';i=features.index(f)
    fig,ax=plt.subplots(figsize=(8.3,4.8))
    sc=ax.scatter(sample[f],grouped[:,i]*100,c=sample.total_nights,cmap='viridis',s=23,alpha=.65)
    ax.axhline(0,color='#777',linewidth=.8)
    ax.set(xlabel='Lead time (days)',ylabel='Grouped lead-time SHAP (percentage points)',
        title='Lead time and model attribution')
    fig.colorbar(sc,ax=ax,label='Total booked nights')
    fig.tight_layout();fig.savefig(OUT/f'figures/{experiment}_lead_time_dependence.png',bbox_inches='tight');plt.close(fig)
    # Derived and correlated features split attribution. Provide family sums as a supplement.
    families={'Lead time family':['lead_time','lead_time_group'],
        'Arrival calendar':['arrival_date_year','arrival_date_month','arrival_date_week_number','arrival_date_day_of_month'],
        'Stay length':['stays_in_weekend_nights','stays_in_week_nights','total_nights'],
        'Guest composition':['adults','children','babies','total_guests','family_booking']}
    family_rows=[]
    for name,fs in families.items():
        v=grouped[:,[features.index(f) for f in fs]].sum(1)
        family_rows.append({'family':name,'members':','.join(fs),'mean_abs_summed_shap':np.abs(v).mean()})
    pd.DataFrame(family_rows).to_csv(OUT/f'tables/{experiment}_shap_feature_families.csv',index=False)
    write_json(OUT/f'shap/{experiment}_explanation_summary.json',dict(model_path=os.path.relpath(path, ROOT),
        model_sha256=sha(path),model_name='RandomForest',selected_tree_status='Random-forest probability explanation under the shared feature and evaluation contract',
        output='uncalibrated cancellation probability',positive_class=1,background_n=100,sample_n=len(sample),
        feature_perturbation='interventional',approximate=False,base_probability=base,
        numeric_alignment=numeric_meta,
        max_additivity_error=float(err.max()),cases=cases,
        limitations=['Predictive attribution, not causation.','Correlated and derived features share credit.',
            'Independent perturbations can create unrealistic feature combinations.',
            '400-row global summary is a sample estimate.','Model scores are not guaranteed calibrated probabilities.']))
    print('SHAP PASS',rank.head(8).to_dict('records'), 'additivity error',err.max(),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--experiment',default='random',choices=['random','chronological','strict_chronological'])
    p.add_argument('--model-path',default=None)
    if OUT.parent.resolve()==ROOT.resolve():
        p.error('Use python src/run_project.py --stage explain --output-root runs/reproduction to preserve reference results.')
    a=p.parse_args();run(a.experiment,a.model_path)
