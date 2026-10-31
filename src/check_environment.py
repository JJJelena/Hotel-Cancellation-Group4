"""Check the selected Python/dependency profile before a project run.

Base: python src/check_environment.py
Full: python src/check_environment.py --full
Add --notebooks to check the optional notebook interface too.
The command reads installed metadata and imports packages. It fits no models,
modifies no reference results, and writes JSON only when --output is supplied.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import re
import sys

MODULES = {
    'numpy': 'numpy', 'pandas': 'pandas', 'scikit-learn': 'sklearn',
    'scipy': 'scipy', 'matplotlib': 'matplotlib', 'joblib': 'joblib',
    'xgboost': 'xgboost', 'nbformat': 'nbformat', 'shap': 'shap',
    'seaborn': 'seaborn', 'jupyterlab': 'jupyterlab',
}


def read_pins(path: Path, visited=None):
    visited = set() if visited is None else visited
    path = path.resolve()
    if path in visited:
        return {}
    visited.add(path)
    pins = {}
    for original in path.read_text(encoding='utf-8-sig').splitlines():
        line = original.split('#', 1)[0].strip()
        if not line:
            continue
        if line.startswith('-r '):
            pins.update(read_pins(path.parent / line[3:].strip(), visited))
            continue
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s;]+)', line)
        if not match:
            raise ValueError(f'Unsupported requirement in {path.name}: {line}')
        name = match.group(1).lower().replace('_', '-')
        expected = match.group(2)
        if name in pins and pins[name] != expected:
            raise ValueError(f'Conflicting pins for {name}')
        pins[name] = expected
    return pins


def check(project_root: Path, full=False, notebooks=False, imports=True):
    errors, notices, packages = [], [], {}
    minimum = (3, 12) if full else (3, 11)
    profile = 'full' if full else 'base'
    if notebooks:
        profile += '+notebooks'
    requirements = project_root / ('requirements-full.txt' if full else 'requirements.txt')
    pins = read_pins(requirements)
    if notebooks:
        pins.update(read_pins(project_root / 'requirements-notebooks.txt'))
    if sys.version_info < minimum:
        errors.append(f'{profile} requires Python {minimum[0]}.{minimum[1]} or newer; detected {platform.python_version()}')
    if sys.version_info[:2] not in [(3, 11), (3, 12), (3, 13)]:
        notices.append('Direct-dependency release metadata may not provide wheels for this interpreter. Python 3.12 is the recommended complete-workflow interpreter.')
    try:
        from packaging.specifiers import SpecifierSet
    except ImportError:
        SpecifierSet = None
        notices.append('packaging is absent, so installed Requires-Python metadata was not evaluated; missing project packages still fail below.')
    for name, expected in pins.items():
        entry = {'required': expected, 'installed': None, 'requires_python': None,
                 'version_matches': False, 'import_ok': None}
        try:
            dist = metadata.distribution(name)
            entry['installed'] = dist.version
            entry['requires_python'] = dist.metadata.get('Requires-Python')
            entry['version_matches'] = dist.version == expected
            if not entry['version_matches']:
                errors.append(f'{name}: installed {dist.version}; profile requires {expected}')
            requirement = entry['requires_python']
            if requirement and SpecifierSet is not None:
                if not SpecifierSet(requirement).contains(platform.python_version(), prereleases=True):
                    errors.append(f'{name} {dist.version} requires Python {requirement}')
            if imports:
                try:
                    importlib.import_module(MODULES[name])
                    entry['import_ok'] = True
                except Exception as exc:
                    entry['import_ok'] = False
                    entry['import_error'] = f'{type(exc).__name__}: {exc}'
                    errors.append(f'{name} failed to import: {type(exc).__name__}: {exc}')
                    if name == 'xgboost' and platform.system() == 'Darwin':
                        errors.append('macOS XGBoost needs the OpenMP runtime. Run brew install libomp, then rerun this check in the same environment.')
        except metadata.PackageNotFoundError:
            errors.append(f'{name} is not installed; expected {expected}')
        packages[name] = entry
    if not full:
        notices.append('SHAP is optional in the base profile. For RF explanations, use Python 3.12 and pip install -r requirements-full.txt.')
    if platform.system() == 'Darwin':
        notices.append('macOS: install LLVM OpenMP with brew install libomp before loading the pinned XGBoost wheel. macOS execution has not been certified by the Windows integration check.')
    notices.append('Matching these direct pins does not guarantee bit-identical retraining across Python, operating system, CPU, BLAS/OpenMP or transitive-library builds. Verify the saved artifacts separately.')
    return {'status': 'PASS' if not errors else 'FAIL', 'profile': profile,
            'python': platform.python_version(), 'implementation': platform.python_implementation(),
            'platform': platform.system(), 'machine': platform.machine(),
            'requirements_file': requirements.name, 'imports_checked': imports,
            'packages': packages, 'errors': errors, 'notices': notices}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--notebooks', action='store_true')
    parser.add_argument('--metadata-only', action='store_true', help='Do not import modules; this cannot verify native libraries.')
    parser.add_argument('--output', type=Path, help='Optional JSON report; use a new run/audit path.')
    args = parser.parse_args()
    result = check(args.project_root.resolve(), args.full, args.notebooks, not args.metadata_only)
    serialised = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialised + '\n', encoding='utf-8')
    print(serialised)
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
