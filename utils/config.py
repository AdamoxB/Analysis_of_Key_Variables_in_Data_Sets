# Configuration constants used by the whole project

import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# These are just placeholders – no actual files are required for this demo.
DATA_FILE  = DATA_DIR / "sample.csv"   # Not used in the example, but handy if you add real data
MODEL_FILE = MODEL_DIR / "model.pkl"
