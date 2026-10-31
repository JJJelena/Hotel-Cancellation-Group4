"""Reproduce descriptive analysis and training-fitted booking segments.

Usage: python src/reproduce_ab.py --root .
The supplied split identifiers remain authoritative. The cancellation target
is used only for descriptive summaries after clustering has been fitted.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


SEGMENT_NAMES = {
    0: 'Mainstream', 1: 'Higher ADR or larger party', 2: 'Early planner',
    3: 'Long stay', 4: 'Prior cancellation history',
}
LEAD_BANDS = ['0-7 days', '8-30 days', '31-90 days', '91-180 days', '181+ days']


def engineer_cluster_features(frame, caps):
    features = frame[['lead_time', 'total_nights', 'total_guests', 'adr']].copy()
    for column in features:
        features[column] = features[column].clip(upper=caps[column])
    features['previous_cancellations_log'] = np.log1p(frame['previous_cancellations'])
    return features


def canonical_labels(train_frame, labels):
    """Assign semantic display IDs from input profiles, never from outcomes."""
    profiles = train_frame.assign(cluster=labels).groupby('cluster')[
        ['lead_time', 'total_nights', 'adr', 'previous_cancellations']].mean()
    remaining = set(profiles.index)
    mapping = {}
    for display_id, feature in [(4, 'previous_cancellations'), (3, 'total_nights'),
                                (1, 'adr'), (2, 'lead_time')]:
        original_id = profiles.loc[sorted(remaining), feature].idxmax()
        mapping[int(original_id)] = display_id
        remaining.remove(original_id)
    mapping[int(remaining.pop())] = 0
    return mapping


def rate_table(frame, grouping):
    result = frame.groupby(grouping, observed=True)['is_canceled'].agg(
        n='size', n_canceled='sum', cancellation_rate='mean').reset_index()
    result['cancellation_rate_pct'] = 100 * result['cancellation_rate']
    return result


def plot_results(out, hotel, lead, cluster_rates, overall):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                         'axes.titlesize': 9, 'axes.labelsize': 8,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'legend.fontsize': 8, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig_dir = out / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    blue, orange = '#426B8A', '#B4653A'
    fig, axes = plt.subplots(2, 1, figsize=(3.25, 3.4),
                             gridspec_kw={'height_ratios': [1, 1.9]})
    for ax, table, names, title in [
        (axes[0], hotel, ['City', 'Resort'], '(a) Hotel type'),
        (axes[1], lead, ['0-7 days', '8-30 days', '31-90 days', '91-180 days', '181+ days'], '(b) Lead time')]:
        positions=np.arange(len(table))
        rates=table.cancellation_rate_pct.to_numpy()
        ax.barh(positions, rates, color=blue, height=.62)
        for x, value, n in zip(positions, rates, table.n):
            ax.text(value + 1.5, x, f'{value:.1f}%', ha='left', va='center', fontsize=8)
        ax.axvline(100*overall, color='#666666', ls='--', lw=.8)
        ax.set_yticks(positions, [f'{name}\n(n={n:,})' for name,n in zip(names,table.n)])
        ax.invert_yaxis()
        ax.set_xlim(0, 73)
        ax.set_xticks([0,20,40,60])
        ax.set_title(title, loc='left')
        ax.set_axisbelow(True)
        ax.grid(axis='x', alpha=.15)
    axes[0].tick_params(axis='x',labelbottom=False)
    axes[1].set_xlabel('Cancellation rate (%)')
    fig.subplots_adjust(left=.31,right=.99,bottom=.13,top=.93,hspace=.50)
    fig.savefig(fig_dir/'eda_hotel_lead.png', dpi=320)
    fig.savefig(fig_dir/'eda_hotel_lead.pdf')
    plt.close(fig)

    fig, ax=plt.subplots(figsize=(3.25, 2.8))
    x=np.arange(5)
    width=.36
    for name, shift, color in [('train', -.5*width, blue), ('test', .5*width, orange)]:
        part=cluster_rates.loc[cluster_rates['split']==name].set_index('cluster').loc[x]
        rates=part.cancellation_rate_pct.to_numpy()
        ax.scatter(rates,x+shift,s=15,color=color,label=name.capitalize(),zorder=3)
        for pos, rate, n in zip(x+shift, rates, part.n):
            ax.text(rate+3, pos, f'{rate:.1f}', ha='left', va='center', fontsize=8)
    counts=cluster_rates.pivot(index='cluster',columns='split',values='n').loc[x]
    names=['C0 Mainstream','C1 Higher ADR','C2 Early planner','C3 Long stay','C4 Prior cancels']
    ax.set_yticks(x,[f'{name}\nn={tr:,} / {te:,}' for name,tr,te in zip(names,counts['train'],counts['test'])])
    ax.set_xlabel('Cancellation rate (%)')
    ax.set_xlim(0,116)
    ax.set_xticks([0,50,100])
    ax.set_ylim(4.45,-.55)
    ax.grid(axis='x', alpha=.15)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='upper left',bbox_to_anchor=(-.85,1.13),ncol=2,
              columnspacing=1,handletextpad=.3)
    fig.subplots_adjust(left=.445,right=.99,bottom=.18,top=.85)
    fig.savefig(fig_dir/'cluster_train_test.png', dpi=320)
    fig.savefig(fig_dir/'cluster_train_test.pdf')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path)
    args=parser.parse_args()
    root=args.root.resolve()
    out=args.out.resolve() if args.out else root/'results'
    tables=out/'tables'
    tables.mkdir(parents=True, exist_ok=True)
    frame=pd.read_csv(root/'data/hotel_bookings_clean.csv')
    train_ids=pd.read_csv(root/'data/splits/train_ids.csv').row_id
    test_ids=pd.read_csv(root/'data/splits/test_ids.csv').row_id
    assert frame.row_id.is_unique
    assert set(train_ids).isdisjoint(test_ids)
    assert set(train_ids)|set(test_ids)==set(frame.row_id)
    train=frame.loc[frame.row_id.isin(train_ids)].copy()
    assert len(train)==94851 and len(frame)==118564

    caps={feature:float(train[feature].quantile(q)) for feature,q in
          {'adr':.99,'total_guests':.99,'lead_time':.995,'total_nights':.995}.items()}
    scaler=StandardScaler()
    train_scaled=scaler.fit_transform(engineer_cluster_features(train, caps))
    model=KMeans(n_clusters=5, n_init=10, random_state=42)
    labels=model.fit_predict(train_scaled)
    mapping=canonical_labels(train, labels)
    full_raw_labels=model.predict(scaler.transform(engineer_cluster_features(frame,caps)))
    frame['cluster']=pd.Series(full_raw_labels,index=frame.index).map(mapping)
    frame['segment_name']=frame.cluster.map(SEGMENT_NAMES)
    frame['split']=np.where(frame.row_id.isin(train_ids),'train','test')
    profile=frame.groupby(['cluster','segment_name'],observed=True).agg(
        n=('row_id','size'),lead_time_mean=('lead_time','mean'),
        total_nights_mean=('total_nights','mean'),total_guests_mean=('total_guests','mean'),
        adr_mean=('adr','mean'),previous_cancellations_mean=('previous_cancellations','mean'),
        cancellation_rate=('is_canceled','mean')).reset_index()
    profile['pct_of_total']=100*profile.n/len(frame)
    profile['cancellation_rate_pct']=100*profile.cancellation_rate
    cluster_rates=rate_table(frame,['cluster','segment_name','split'])
    cluster_hotel=rate_table(frame,['cluster','segment_name','hotel'])
    hotel=rate_table(frame,['hotel'])
    lead=rate_table(frame,['lead_time_group']).set_index('lead_time_group').loc[LEAD_BANDS].reset_index()
    for name,table in [('cluster_profile_full',profile),('cluster_rates_by_split',cluster_rates),
                       ('cluster_rates_by_hotel',cluster_hotel),('eda_hotel_rates',hotel),
                       ('eda_lead_time_rates',lead)]:
        table.to_csv(tables/f'{name}.csv',index=False,float_format='%.10g')
    frame[['row_id','cluster','segment_name','split']].to_csv(tables/'cluster_assignments.csv',index=False)
    metadata={'fit_subset':'shared random training set only', 'fit_rows':len(train),
              'assignment_rows':len(frame),'random_state':42,'n_init':10,'k':5,
              'cluster_inputs':list(engineer_cluster_features(train,caps)),
              'outlier_caps_from_training':caps,'label_mapping':mapping,
              'outcome_used_for_fit_or_k_selection':False,
              'k_selection_note':'The supplied training diagnostics give silhouette 0.3165 at k=3, 0.3032 at k=5 and 0.3049 at k=6; k=5 retains interpretable distinctions. This script reproduces the chosen k=5 fit.',
              'profile_basis':'Uncapped original measurements; rates describe associations.'}
    (out/'ab_analysis_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    plot_results(out,hotel,lead,cluster_rates,frame.is_canceled.mean())
    print(json.dumps({'out':str(out),'cluster_sizes':profile.n.tolist()},indent=2))


if __name__=='__main__':
    main()
