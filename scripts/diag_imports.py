import sys
import importlib.util

print('python_version:', sys.version)

for pkg in ('joblib', 'sklearn', 'mlflow'):
    spec = importlib.util.find_spec(pkg)
    print(f"{pkg} spec:", spec)

try:
    import _multiprocessing as mp
    print('_multiprocessing: OK')
except Exception as e:
    print('_multiprocessing import error:', type(e).__name__, e)

try:
    import joblib
    print('joblib imported OK')
except Exception as e:
    import traceback
    print('joblib import failed:', type(e).__name__)
    traceback.print_exc()
