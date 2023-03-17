import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import matplotlib as mpl

mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["font.size"] = 14
mpl.rcParams["xtick.major.size"] = 7
mpl.rcParams["ytick.major.size"] = 7
mpl.rcParams["xtick.minor.size"] = 4
mpl.rcParams["ytick.minor.size"] = 4
mpl.rcParams["figure.figsize"] = [8.0, 6.0]


def plot_intrinsic_vs_observed(CS, pp_ref=None):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))
    offset = 100

    off_start = 0
    off_end = 200
    target_max = 100
    if pp_ref is None:
        pp_ref = CS.pp_ref
    base_ref = np.mean(pp_ref[off_start:off_end])
    ref = pp_ref - base_ref
    ref /= np.max(ref) / target_max
    roll = -np.argmax(ref) + offset
    ref = np.roll(ref, roll)
    base_int = np.mean(CS.pp_int[off_start:off_end])
    _int = CS.pp_int - base_int
    _int /= np.max(_int) / target_max
    _int = np.roll(_int, roll)

    axs[0].plot(
        np.arange(2048) / 1024,
        np.array((ref, ref)).ravel() + 1,
        label="as usual",
        c="black",
    )
    axs[0].plot(
        np.arange(2048) / 1024,
        np.array((_int, _int)).ravel() + 1,
        label="intrinsic",
        c="red",
    )

    axs[0].set_xticks(
        np.arange(offset / 1024, 2048 / 1024, 512 / 1024),
        labels=("0.0", "0.5", "1.0", "1.5"),
    )

    axs[0].tick_params(which="both", direction="in")
    axs[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0].yaxis.set_minor_locator(MultipleLocator(5.0))
    axs[0].set_ylim(-1.0, 100.0)
    axs[0].spines.right.set_visible(False)
    axs[0].spines.top.set_visible(False)

    axs[1].plot(
        np.arange(2048) / 1024,
        np.array((ref, ref)).ravel() + 1,
        label="as usual",
        c="black",
    )
    axs[1].plot(
        np.arange(2048) / 1024,
        np.array((_int, _int)).ravel() + 1,
        label="intrinsic",
        c="red",
    )

    axs[1].set_xticks(
        np.arange(offset / 1024, 2048 / 1024, 512 / 1024),
        labels=("0.0", "0.5", "1.0", "1.5"),
    )
    axs[1].tick_params(which="both", direction="in")
    axs[1].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[1].set_yticks((0, 1, 2, 3, 4))
    axs[1].yaxis.set_minor_locator(MultipleLocator(0.2))
    axs[1].set_ylim(0.0, 4.0)
    axs[1].spines.right.set_visible(False)
    axs[1].spines.top.set_visible(False)
    _ = axs[1].set_xlabel("Pulse Phase [Turns]")
