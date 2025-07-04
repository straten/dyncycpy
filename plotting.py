import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.fft import fftshift
import matplotlib as mpl

mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["font.size"] = 14
mpl.rcParams["xtick.major.size"] = 7
mpl.rcParams["ytick.major.size"] = 7
mpl.rcParams["xtick.minor.size"] = 4
mpl.rcParams["ytick.minor.size"] = 4
mpl.rcParams["figure.figsize"] = [8.0, 6.0]


def plot_intrinsic_vs_observed(CS, pp_ref=None,savefig=None):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))

    target_max = 100
    if pp_ref is None:
        pp_ref = CS.pp_ref

    nbin = pp_ref.size

    offset = 100 * nbin//1024
    off_start = 0
    off_end = 200 * nbin//1024

    base_ref = np.mean(pp_ref[off_start:off_end])
    ref = pp_ref - base_ref
    ref /= np.max(ref) / target_max
    roll = -np.argmax(ref) + offset
    ref = np.roll(ref, roll)
    base_int = np.mean(CS.pp_intrinsic[off_start:off_end])
    _int = CS.pp_intrinsic - base_int
    _int /= np.max(_int) / target_max
    _int = np.roll(_int, roll)

    # print(f'plot_intrinsic_vs_observed nbin={nbin} ref.shape={pp_ref.shape}')

    axs[0].plot(
        np.arange(nbin*2) / nbin,
        np.array((ref, ref)).ravel() + 1,
        label="as usual",
        c="black",
    )
    axs[0].plot(
        np.arange(nbin*2) / nbin,
        np.array((_int, _int)).ravel() + 1,
        label="intrinsic",
        c="red",
    )

    axs[0].set_xticks(
        np.arange(offset / nbin, 2, 0.5),
        labels=("0.0", "0.5", "1.0", "1.5"),
    )

    axs[0].tick_params(which="both", direction="in")
    axs[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0].yaxis.set_minor_locator(MultipleLocator(5.0))
    axs[0].set_ylim(-1.0, 100.0)
    axs[0].spines.right.set_visible(False)
    axs[0].spines.top.set_visible(False)

    axs[1].plot(
        np.arange(nbin*2) / nbin,
        np.array((ref, ref)).ravel() + 1,
        label="as usual",
        c="black",
    )
    axs[1].plot(
        np.arange(nbin*2) / nbin,
        np.array((_int, _int)).ravel() + 1,
        label="intrinsic",
        c="red",
    )

    axs[1].set_xticks(
        np.arange(offset / nbin, 2, 0.5),
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
    if savefig is not None:
        fig.savefig(savefig)


def plot_Doppler_vs_delay (h_doppler_delay, dT, bw, filename):

    ntime, ntap = h_doppler_delay.shape

    if dT == 0 or bw == 0:
        extent = None
    else:
        delta_delay_mus = np.abs(1.0 / bw)
        max_delay_ms = delta_delay_mus * ntap * 0.5e-3
        max_Doppler_Hz = .5 / dT
        
        extent=[-max_Doppler_Hz, max_Doppler_Hz, -max_delay_ms, max_delay_ms]
    
    plotthis = np.log10(np.abs(fftshift(h_doppler_delay)) + 1e-2)
    plotmed = np.median(plotthis)
    fig, ax = plt.subplots(figsize=(8, 9))

    ax.set_xlabel("Cycle Frequency [Hz]")
    ax.set_ylabel("Delay [ms]")
    img = ax.imshow(plotthis.T, aspect="auto", origin="lower", cmap="cubehelix_r", vmin=plotmed, extent=extent, interpolation='none')
    fig.colorbar(img)
    fig.savefig(filename)
    plt.close()