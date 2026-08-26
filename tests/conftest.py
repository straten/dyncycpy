import pathlib
import sys

# Ensure the repository root (where pycyc.py and plotting.py live) is
# importable regardless of the directory pytest is invoked from.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib

# pycyc.py imports matplotlib.pyplot at module scope; force a headless
# backend before that import happens so the suite runs without a display.
matplotlib.use("Agg")
