"""Create a readable single-column plot from the saved grouped forest SHAP table."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args()
    root=args.project_root.resolve()
    out=(args.output_root or root/'runs'/'figure_reproduction').resolve()/'results'/'figures'
    out.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(root/'results/tables/random_shap_importance.csv').head(8).iloc[::-1]
    names={'country':'Country','agent':'Agent','lead_time':'Lead time','previous_cancellations':'Prior cancellations','market_segment':'Market segment','arrival_date_year':'Arrival year','lead_time_group':'Lead-time band','arrival_date_week_number':'Arrival week'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(3.25,2.45))
    values=data.mean_abs_shap*100
    ax.barh([names[x] for x in data.feature],values,color='#187f96',height=.7)
    ax.set(xlabel='Mean |SHAP| (probability pp)',xlim=(0,23))
    ax.set_xticks([0,5,10,15,20])
    ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
    for i,value in enumerate(values):ax.text(value+.25,i,f'{value:.2f}',va='center',fontsize=7.5)
    fig.subplots_adjust(left=.36,right=.99,bottom=.21,top=.98)
    fig.savefig(out/'forest_shap_single.pdf')
    fig.savefig(out/'forest_shap_single.png',dpi=320)

if __name__=='__main__':main()
