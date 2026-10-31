"""Unified entry point. Verification reads saved evidence; retraining writes a new run."""
from pathlib import Path
import argparse, os, subprocess, sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['verify','ab','dummy','logistic','forest','explain','xgboost','probability','all'], default='verify')
    parser.add_argument('--output-root',type=Path,default=ROOT/'runs'/'reproduction')
    parser.add_argument('--check-models',action='store_true')
    args=parser.parse_args()
    output=args.output_root.resolve()
    if args.stage!='verify' and output==ROOT:
        parser.error('Choose a separate output root to preserve the submitted results.')
    env=os.environ.copy()
    env['HOTEL_OUTPUT_ROOT']=str(output)
    def run(script,*extra):
        subprocess.run([sys.executable,str(ROOT/'src'/script),*map(str,extra)],cwd=ROOT,env=env,check=True)
    if args.stage=='verify':
        run('verification.py','--project-root',ROOT,*(['--check-models'] if args.check_models else []))
        return
    output.mkdir(parents=True,exist_ok=True)
    if args.stage in ['ab','all']:
        run('reproduce_ab.py','--root',ROOT,'--out',output/'results')
    if args.stage in ['dummy','all']:
        import evaluation as ev
        from preprocessing_source import load_shared_data
        from sklearn.dummy import DummyClassifier
        data=load_shared_data(ROOT)
        data['project_root']=output
        ev.run_standard_experiment(ev.make_pipeline(data,DummyClassifier(strategy='prior')),
            model='dummy',data=data,params={'strategy':'prior'})
    if args.stage in ['logistic','all']:
        run('train_logistic.py','--project-root',ROOT,'--output-root',output)
    if args.stage in ['forest','all']:
        run('train_rf.py','--stage','all')
    if args.stage in ['explain','all']:
        run('explain_rf.py')
    if args.stage in ['xgboost','all']:
        run('train_xgboost.py','--project-root',ROOT,'--output-root',output)
    if args.stage in ['probability','all']:
        run('threshold_calibration.py','--project-root',ROOT,'--prediction-root',output,'--output-root',output)

if __name__=='__main__':
    main()
