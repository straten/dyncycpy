#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.fft import fftshift

def list_nonzero(filename, threshold) -> None:

    with open (filename, "rb") as fh:
        wavefield = fftshift(pickle.load(fh))

    nchan=wavefield.shape[1]
    nsub=wavefield.shape[0]

    abs_wav = np.abs(wavefield)
    abs_max = np.max(abs_wav)

    indeces = np.where(abs(abs_wav) > abs_max*threshold)
    coords = np.transpose(indeces)
    print(f"{filename} nsub={nsub} nchan={nchan}")
    for coord in coords:
        print(f"{coord[0]-nsub//2} {coord[1]-nchan//2} {wavefield[coord[0],coord[1]]}")

    minx = np.min(indeces[0])
    maxx = np.max(indeces[0]) + 1
    miny = np.min(indeces[1])
    maxy = np.max(indeces[1]) + 1
    pextent = [minx-nsub//2, maxx-nsub//2, miny-nchan//2, maxy-nchan//2]

    print(f"x:{minx}->{maxx} y:{miny}->{maxy}")

    plt.rc('figure', figsize=[16,8])
    fig, axs = plt.subplots(1,2)

    plotthis = np.real(wavefield[minx:maxx,miny:maxy])
    pmax = np.max(abs(plotthis))
    img = axs[0].imshow(plotthis.T, aspect="auto", origin="lower", cmap="bwr", interpolation=None, extent=pextent, vmin=-pmax, vmax=pmax)
    fig.colorbar(img)

    plotthis = np.imag(wavefield[minx:maxx,miny:maxy])
    pmax = np.max(abs(plotthis))
    img = axs[1].imshow(plotthis.T, aspect="auto", origin="lower", cmap="bwr", interpolation=None, extent=pextent, vmin=-pmax, vmax=pmax)
    fig.colorbar(img)

    filename = "nonzero.png"
    fig.savefig(filename)
    plt.close()

def main() -> None:
    """Plot the cyclic spectrum in four different ways."""
    import argparse

    # do arg parsing here
    p = argparse.ArgumentParser()
    p.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="the threshold to apply",
    )

    args, files = p.parse_known_args()

    for file in files:
        list_nonzero(file, args.threshold)

if __name__ == "__main__":
  main()
