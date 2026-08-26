"""
Command-line entry point, preserving the original `python pycyc.py <file>
[profile]` usage as `python -m pycyc <file> [profile]` now that pycyc is a
package.
"""

import sys

import numpy as np

from . import CyclicSolver

if __name__ == "__main__":
    fname = sys.argv[1]
    CS = CyclicSolver(filename=fname)
    if len(sys.argv) > 2:
        CS.initProfile(loadFile=sys.argv[2])
    else:
        CS.initProfile()
    np.save(("%s_profile.npy" % CS.source), CS.pp_intrinsic)
    CS.loop(make_plots=True, tolfact=20)
    CS.saveResults()
