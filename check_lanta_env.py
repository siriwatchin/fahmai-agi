import importlib
import sys

print("python", sys.version.replace("\n", " "))

mods = [
    "torch",
    "transformers",
    "accelerate",
    "duckdb",
    "pandas",
    "qdrant_client",
    "sentence_transformers",
]

for mod in mods:
    try:
        module = importlib.import_module(mod)
        print(mod, getattr(module, "__version__", "ok"))
    except Exception as exc:
        print(mod, "MISSING", exc)
