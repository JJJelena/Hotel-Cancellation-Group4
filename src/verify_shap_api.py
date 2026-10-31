"""Fresh small-sample SHAP API/additivity check, separate from frozen E results.

The saved forest is not refitted or changed; no 400-row explanation is claimed.
Example: python src/verify_shap_api.py --output-root runs/shap_api_check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--sample-size', type=int, choices=[5, 20], default=5)
    args = parser.parse_args()
    project, output = args.project_root.resolve(), args.output_root.resolve()
    if output == project or project in output.parents and 'results' in output.relative_to(project).parts:
        parser.error('Use a separate run output directory; frozen results must not be overwritten.')
    os.environ['HOTEL_OUTPUT_ROOT'] = str(output)
    import joblib
    import numpy as np
    import shap
    from preprocessing_source import load_shared_data
    from reproduction_metadata import write_run_environment, sha256
    from shap_compat import rf_probability_explainer, POLICY

    audit = output/'results/audit'
    audit.mkdir(parents=True, exist_ok=True)
    write_run_environment(audit/'shap_api_run_environment.json', project_root=project,
        stage='fresh_small_sample_shap_api_check', seed=2026,
        thread_settings={'saved_rf_n_jobs':10},
        algorithm_settings={'background_n':100,'background_seed':42,
            'explained_n':args.sample_size,'explained_seed':2026,
            'numeric_policy':POLICY,'model_refitted':False})
    started = time.perf_counter()
    data = load_shared_data(project)
    path = project/'results/models/random_forest_main.joblib'
    before = sha256(path)
    saved = joblib.load(path)
    pipeline = saved['pipeline'] if isinstance(saved, dict) else saved
    pre, model = pipeline.named_steps['preprocessing'], pipeline.named_steps['model']
    background = data['random_train'].sample(n=100, random_state=42)
    # This is the first 5/20 rows of the exact 400-row frozen sampling protocol.
    sample = data['random_test'].sample(n=400, random_state=2026).iloc[:args.sample_size]
    def dense(frame):
        transformed = pre.transform(frame[data['features']])
        return np.asarray(transformed.toarray() if hasattr(transformed, 'toarray') else transformed,
            dtype=np.float32)
    explainer, numeric = rf_probability_explainer(model, dense(background))
    raw = explainer.shap_values(dense(sample), approximate=False, check_additivity=True)
    class_index = int(np.flatnonzero(model.classes_ == 1)[0])
    values = raw[:, :, class_index] if isinstance(raw, np.ndarray) and raw.ndim == 3 else raw[class_index]
    base = float(np.asarray(explainer.expected_value).reshape(-1)[class_index])
    probability = pipeline.predict_proba(sample[data['features']])[:, class_index]
    reconstructed = base + values.sum(axis=1)
    errors = np.abs(reconstructed-probability)
    after = sha256(path)
    result = dict(schema_version='small-sample-shap-api-check-v1',
        status='passed' if errors.max() < 1e-5 and before == after else 'failed',
        scope='Fresh API execution and probability additivity on a saved random RF; not model retraining or full 400-row E reproduction.',
        package_version=shap.__version__, background_n=100, explained_n=args.sample_size,
        background_row_ids=background.row_id.astype(int).tolist(),
        explained_row_ids=sample.row_id.astype(int).tolist(),
        model_sha256_before=before,model_sha256_after=after,model_file_unchanged=before==after,
        numeric_policy=numeric, expected_value=base,
        additivity_check_tolerance=1e-5,max_additivity_error=float(errors.max()),
        mean_additivity_error=float(errors.mean()),
        rows=[dict(row_id=int(row),probability=float(p),reconstructed_probability=float(q),absolute_error=float(e))
            for row,p,q,e in zip(sample.row_id,probability,reconstructed,errors)],
        seconds=time.perf_counter()-started,
        frozen_400_row_results_overwritten=False,
        limitations=['5/20-row smoke check does not replace the frozen 400-row global ranking or stability analysis.',
            'This check establishes compatibility only in the recorded current environment.'])
    (audit/'shap_api_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:result[k] for k in ('status','explained_n','max_additivity_error','model_file_unchanged','seconds')},indent=2))
    return 0 if result['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
