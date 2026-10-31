"""Record an observed training environment before fitting without altering results."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys

PACKAGES = ('numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib',
    'xgboost', 'shap', 'matplotlib', 'nbformat', 'threadpoolctl')
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OMP_DYNAMIC', 'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def capture_run_environment(project_root, stage, seed, thread_settings=None,
                            algorithm_settings=None):
    """Capture current execution facts; this is not a historical artifact manifest."""
    root = Path(project_root).resolve()
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    files = [root / 'data/hotel_bookings_clean.csv', root / 'data/data_dictionary.csv']
    files += sorted((root / 'data/splits').glob('*.csv'))
    files += sorted((root / 'src').glob('*.py'))
    files += sorted(root.glob('requirements*.txt'))
    libraries = []
    try:
        from threadpoolctl import threadpool_info
        libraries = [{key: item.get(key) for key in
            ('user_api', 'internal_api', 'prefix', 'version', 'num_threads',
             'threading_layer', 'architecture')} for item in threadpool_info()]
    except ImportError:
        pass
    return dict(schema_version='training-environment-v1',
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        metadata_scope='observed environment at run start, not the creating environment of saved reference artifacts',
        stage=stage, seed=seed,
        python=dict(version=platform.python_version(), implementation=platform.python_implementation(),
                    compiler=platform.python_compiler(), pointer_bits=64 if sys.maxsize > 2**32 else 32),
        platform=dict(system=platform.system(), release=platform.release(), machine=platform.machine(),
                      processor=platform.processor(), logical_cpu_count=os.cpu_count()),
        packages=packages, explicit_thread_settings=thread_settings or {},
        thread_environment={key: os.environ.get(key) for key in THREAD_VARIABLES},
        loaded_numeric_libraries=libraries,
        algorithm_settings=algorithm_settings or {},
        input_and_source_sha256={p.relative_to(root).as_posix(): sha256(p) for p in files if p.is_file()},
        reproducibility_scope='fixed input files, folds, seed and documented algorithm settings; cross-version or cross-platform bitwise equality is not claimed')


def write_run_environment(path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    evidence = capture_run_environment(**kwargs)
    path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    return evidence
