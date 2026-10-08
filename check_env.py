import importlib, sys
print("Python", sys.version.split()[0])
for m in ["pandas","numpy","pyarrow","sklearn","xgboost","matplotlib","scipy"]:
    try:
        mod = importlib.import_module(m)
        print(f"OK   {m} {getattr(mod, '__version__', '')}")
    except ImportError:
        print(f"MISSING {m}  -> pip install -r requirements.txt")
