import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from scipy.fft import fftshift
import matplotlib as mpl

# match_two_filters/time2freq (needed only by plot_simulation, below) are
# imported lazily inside that function rather than here: pycyc/solver.py
# imports this module at load time (for plot_Doppler_vs_delay), so a
# module-level `from pycyc... import ...` here would be a circular import
# whenever `plotting` happens to be the first of the two imported.

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

    # estimate the baseline, assuming that on-pulse phase bins are in the minority
    # such that they are effectively outliers that do not significantly impact the median
    base_ref = np.median(pp_ref)
    ref = pp_ref - base_ref
    ref /= np.max(ref) / target_max
    roll = -np.argmax(ref) + offset
    ref = np.roll(ref, roll)
    base_int = np.median(CS.pp_intrinsic)
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


def plot_Doppler_vs_delay (h_doppler_delay, dT, bw, filename=None):

    ntime, ndelay = h_doppler_delay.shape

    if dT == 0:
        max_Doppler_Hz = ntime/2
        xlabel="Temporal Cycles"
    else:
        max_Doppler_Hz = .5 / dT
        xlabel="Cycle Frequency [Hz]"

    if bw == 0:
        print("plot_Doppler_vs_delay bandwidth is zero")
        max_delay_ms = ndelay/2
        ylabel="Spectral Cycles"
    else:
        delta_delay_mus = np.abs(1.0 / bw)
        max_delay_ms = delta_delay_mus * ndelay * 0.5e-3
        ylabel="Delay [ms]"

    extent=[-max_Doppler_Hz, max_Doppler_Hz, -max_delay_ms, max_delay_ms]
    
    plotthis = np.log10(np.abs(fftshift(h_doppler_delay)) + 1e-6)
    plotmed = np.median(plotthis[:,ndelay//2:])
    fig, ax = plt.subplots(figsize=(8, 9))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()

    img = ax.imshow(plotthis.T, aspect="auto", origin="lower", cmap="cubehelix_r", vmin=plotmed, extent=extent, interpolation='none')
    fig.colorbar(img)

    if filename is not None:
        fig.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_power_vs_delay (h_doppler_delay, bw, filename=None):

    ntime, ndelay = h_doppler_delay.shape

    if bw == 0:
        x = np.linspace(0, ndelay, ndelay)
    else:
        delta_delay_mus = np.abs(1.0 / bw)
        max_delay_ms = delta_delay_mus * ndelay * 0.5e-3        
        x = np.linspace(-max_delay_ms, max_delay_ms, ndelay)

    toplot = np.log10(np.sum(np.abs(h_doppler_delay) ** 2, axis=0) + 1e-16)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlabel("Delay [ms]")
    ax.set_ylabel("$\log_{10}$(Power)")
    ax.plot(x, fftshift(toplot))
    if filename is not None:
        fig.savefig(filename)
        plt.close()
    else:
        plt.show()

def plot_current_solution(
    plot_cs,
    cs_model,
    grad,
    hf,
    ht,
    mlag,
    rf,
    bw,
    rindex,
    noise,
    nchan,
    pref,
    psopt,
    psmeas,
    objval,
    filename,
    isub,
    nopt,
    source,
    niter,
    plotdir,
):
    """
    Diagnostic 9-panel figure comparing the model and measured cyclic
    spectra, the impulse/frequency response, the merit trajectory, and the
    reference/intrinsic/measured profiles for one subint/pol.

    Pure function: CyclicSolver.plotCurrentSolution computes sopt/smeas
    (via optimize_profile) and pref/psopt/psmeas (via harm2phase) and passes
    them in, so this has no CyclicSolver dependency.
    """
    fig = Figure()
    ax1 = fig.add_subplot(3, 3, 1)
    csextent = [1, mlag - 1, rf + bw / 2.0, rf - bw / 2.0]
    im = ax1.imshow(
        np.log10(np.abs(plot_cs[:, 1:mlag])),
        aspect="auto",
        interpolation="nearest",
        extent=csextent,
    )
    # im = ax1.imshow(cs2ps(plot_cs),aspect='auto',interpolation='nearest',extent=csextent)
    ax1.set_xlim(0, mlag)
    ax1.text(
        0.9,
        0.9,
        "log|CS|",
        fontdict=dict(size="small"),
        va="top",
        ha="right",
        transform=ax1.transAxes,
        bbox=dict(alpha=0.75, fc="white"),
    )
    im.set_clim(-4, 2)

    ax1b = fig.add_subplot(3, 3, 2)
    im = ax1b.imshow(
        np.angle(plot_cs[:, :mlag]) - np.median(np.angle(plot_cs[:, :mlag]), axis=0)[None, :],
        cmap="hsv",
        aspect="auto",
        interpolation="nearest",
        extent=csextent,
    )
    # im = ax1b.imshow(plot_cs[:,:mlag].imag,aspect='auto',interpolation='nearest',extent=csextent)

    im.set_clim(-np.pi, np.pi)
    ax1b.set_xlim(0, mlag)
    ax1b.text(
        0.9,
        0.9,
        "angle(CS)",
        fontdict=dict(size="small"),
        va="top",
        ha="right",
        transform=ax1b.transAxes,
        bbox=dict(alpha=0.75, fc="white"),
    )
    for tl in ax1b.yaxis.get_ticklabels():
        tl.set_visible(False)
    ax2 = fig.add_subplot(3, 3, 4)
    im = ax2.imshow(
        np.log10(np.abs(cs_model[:, 1:mlag])),
        aspect="auto",
        interpolation="nearest",
        extent=csextent,
    )
    # im = ax2.imshow(cs2ps(cs_model),aspect='auto',interpolation='nearest',extent=csextent)
    im.set_clim(-4, 2)
    ax2.set_xlim(0, mlag)
    ax2.set_ylabel("RF (MHz)")
    ax2.text(
        0.9,
        0.9,
        "log|CS model|",
        fontdict=dict(size="small"),
        va="top",
        ha="right",
        transform=ax2.transAxes,
        bbox=dict(alpha=0.75, fc="white"),
    )

    ax2b = fig.add_subplot(3, 3, 5)
    im = ax2b.imshow(
        np.angle(cs_model[:, :mlag]) - np.median(np.angle(cs_model[:, :mlag]), axis=0)[None, :],
        cmap="hsv",
        aspect="auto",
        interpolation="nearest",
        extent=csextent,
    )
    # im = ax2b.imshow(cs_model[:,:mlag].imag,aspect='auto',interpolation='nearest',extent=csextent)
    im.set_clim(-np.pi, np.pi)
    ax2b.set_xlim(0, mlag)
    ax2b.text(
        0.9,
        0.9,
        "angle(CS model)",
        fontdict=dict(size="small"),
        va="top",
        ha="right",
        transform=ax2b.transAxes,
        bbox=dict(alpha=0.75, fc="white"),
    )
    for tl in ax2b.yaxis.get_ticklabels():
        tl.set_visible(False)

    ax3 = fig.add_subplot(3, 3, 7)
    #        ax3.imshow(np.log(np.abs(cs_model0)[:,1:]),aspect='auto')
    err = np.abs(plot_cs - cs_model)[:, 1:mlag]
    # err = cs2ps(plot_cs) - cs2ps(normalize_cs(cs_model,bw,self.ref_freq))
    im = ax3.imshow(err, aspect="auto", interpolation="nearest", extent=csextent)
    ax3.set_xlim(0, mlag)
    #        im.set_clim(err[1:-1,1:-1].min(),err[1:-1,1:-1].max())
    im.set_clim(0, 3 * noise)
    ax3.text(
        0.9,
        0.9,
        "|error|",
        fontdict=dict(size="small"),
        va="top",
        ha="right",
        transform=ax3.transAxes,
        bbox=dict(alpha=0.75, fc="white"),
    )
    ax3.set_xlabel("Harmonic")

    ax3b = fig.add_subplot(3, 3, 8)
    im = ax3b.imshow(
        np.angle((plot_cs[:, :mlag] / cs_model[:, :mlag])),
        cmap="hsv",
        aspect="auto",
        interpolation="nearest",
        extent=csextent,
    )
    im.set_clim(-np.pi / 2.0, np.pi / 2.0)
    ax3b.set_xlim(0, mlag)
    ax3b.text(
        0.9,
        0.9,
        "angle(error)",
        fontdict=dict(size="small"),
        va="top",
        ha="right",
        transform=ax3b.transAxes,
        bbox=dict(alpha=0.75, fc="white"),
    )
    for tl in ax3b.yaxis.get_ticklabels():
        tl.set_visible(False)
    ax3b.set_xlabel("Harmonic")

    ax4 = fig.add_subplot(4, 3, 3)
    t = np.arange(ht.shape[0]) / bw
    ax4.plot(t, np.roll(20 * np.log10(np.abs(ht)), int(ht.shape[0] / 2) - rindex))
    ax4.plot(
        t,
        np.roll(
            20 * np.log10(np.convolve(np.ones((10,)) / 10.0, np.abs(ht), mode="same")),
            int(ht.shape[0] / 2) - rindex,
        ),
        linewidth=2,
        color="r",
        alpha=0.4,
    )

    ax4.set_ylim(0, 80)
    ax4.set_xlim(t[0], t[-1])
    ax4.text(
        0.9,
        0.9,
        "dB|h(t)|$^2$",
        fontdict=dict(size="small"),
        va="top",
        ha="right",
        transform=ax4.transAxes,
    )
    ax4.text(
        0.95,
        0.01,
        "$\\mu$s",
        fontdict=dict(size="small"),
        va="bottom",
        ha="right",
        transform=ax4.transAxes,
    )
    ax4b = fig.add_subplot(4, 3, 6)
    f = np.linspace(rf + bw / 2.0, rf - bw / 2.0, nchan)
    ax4b.plot(f, np.abs(hf))
    ax4b.text(
        0.9,
        0.9,
        "|H(f)|",
        fontdict=dict(size="small"),
        va="top",
        ha="right",
        transform=ax4b.transAxes,
    )
    ax4b.text(
        0.95,
        0.01,
        "MHz",
        fontdict=dict(size="small"),
        va="bottom",
        ha="right",
        transform=ax4b.transAxes,
    )
    ax4b.set_xlim(f.min(), f.max())
    ax4b.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax5 = fig.add_subplot(4, 3, 9)
    if len(objval) >= 3:
        x = np.abs(np.diff(np.array(objval).flatten()))
        ax5.plot(np.arange(x.shape[0]), np.log10(x))
    ax5.text(
        0.9,
        0.9,
        "log($\\Delta$merit)",
        fontdict=dict(size="small"),
        va="top",
        ha="right",
        transform=ax5.transAxes,
    )
    ax6 = fig.add_subplot(4, 3, 12)
    pref = pref
    ax6.plot(pref, label="Reference", linewidth=2)
    ax6.plot(psopt, "r", label="Intrinsic")
    ax6.plot(psmeas, "g", label="Measured")
    legend = ax6.legend(loc="upper left", prop=dict(size="xx-small"), title="Profiles")
    legend.get_frame().set_alpha(0.5)
    ax6.set_xlim(0, pref.shape[0])
    fname = filename[-50:]
    if len(filename) > 50:
        fname = "..." + fname
    title = "%s isub: %d nopt: %d\n" % (fname, isub, nopt)
    title += "Source: %s Freq: %s MHz Feval #%04d Merit: %.3e Grad: %.3e" % (
        source,
        rf,
        niter,
        objval[-1],
        np.abs(grad).sum(),
    )
    fig.suptitle(title, size="small")
    canvas = FigureCanvasAgg(fig)
    fname = os.path.join(plotdir, ("%s_%04d_%04d.png" % (source, nopt, niter)))
    canvas.print_figure(fname)


def plot_simulation(
    ht0_all,
    hf,
    ht,
    cs0,
    cs_model,
    bw,
    rf,
    nchan,
    isub,
    nphase,
    ref_freq,
    pp_meas,
    pp_intrinsic,
    pref,
    tau,
    cs,
    filename,
    ipol,
    nopt,
    niter,
    objval,
    plotdir,
    source,
    noise,
    pharm=None,
    mlag=100,
):
    """
    Diagnostic figure comparing a simulation's known input wavefield/profile
    against the recovered solution. Orphaned/experimental: no code in this
    repository currently calls this (it references a "simulation mode" --
    ht0_all, pp_meas, tau, pharm -- that no longer exists on CyclicSolver),
    kept and given the same explicit-parameter treatment as
    plot_current_solution in case it's revived. `pharm`/`tau` are optional,
    matching the original's try/except AttributeError pattern for
    attributes that were not always set.

    Pure function: pass cs0 = CyclicSolver.modelCS(ht) and
    cs_model = CyclicSolver.modelCS(ht0) precomputed, and
    pref = CyclicSolver.harm2phase(ph_ref) precomputed (the original called
    an undefined `self.harm2phase` here -- a bug, since this was a bare
    module-level function, not a method).
    """
    from pycyc.transforms import match_two_filters, time2freq

    if ht0_all is None:
        print("Does not appear this is a simulation run")
    ht = ht  # [isub,:]
    t = np.arange(ht.shape[0]) / bw
    f = np.linspace(rf + bw / 2.0, rf - bw / 2.0, nchan)
    csextent = [1, mlag - 1, rf - bw / 2.0, rf + bw / 2.0]
    ht0 = ht0_all[isub]
    hf0 = match_two_filters(hf, time2freq(ht0))
    
    fig = Figure(figsize=(10, 7))
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(f, np.abs(hf), label=r"|$\hat{H}(f)$|")
    ax1.plot(f, np.abs(hf0), label="|$H$(f)|")
    legend = ax1.legend(loc="upper right", prop=dict(size="x-small"))
    legend.get_frame().set_alpha(0.5)
    ax1.text(
        0.9,
        0.1,
        "MHz",
        fontdict=dict(size="small"),
        va="top",
        ha="right",
        transform=ax1.transAxes,
        bbox=dict(alpha=0.75, fc="white"),
    )

    ax1.set_xlim(f.min(), f.max())

    ax2 = fig.add_subplot(3, 3, 4)
    ax2.plot(f, np.abs(hf / hf0), label=r"$\left|\frac{\hat{H}(f)}{H(f)}\right|$")
    ax2.plot(
        f,
        np.angle(hf / hf0),
        alpha=0.7,
        label=r"$\angle\left(\frac{\hat{H}(f)}{H(f)}\right)$",
    )
    legend = ax2.legend(loc="lower left", prop=dict(size="x-small"))
    legend.get_frame().set_alpha(0.5)
    ax2.text(
        0.9,
        0.1,
        "MHz",
        fontdict=dict(size="small"),
        va="top",
        ha="right",
        transform=ax2.transAxes,
        bbox=dict(alpha=0.75, fc="white"),
    )

    # ax2.plot(f[:-1],np.diff(np.angle(hf/hf0)))
    # ax2.plot(f[:-1],np.diff(np.angle(minphase(np.abs(hf))/hf0)))
    ax2.set_ylim(-3.2, 3.2)
    ax2.set_xlim(f.min(), f.max())

    ax3 = fig.add_subplot(3, 3, 2)
    #    im = ax3.imshow(np.abs(cs_model[:,:mlag]/cs0[:,:mlag]),
    #                     aspect='auto',interpolation='nearest',extent=csextent)
    #    im.set_clim(0.5,2)
    pt = 1e3 * np.linspace(0, 1, nphase) / ref_freq
    ax3.plot(pt, fftshift(pp_meas), "r", label="measured")
    ax3.plot(pt, fftshift(pp_intrinsic), "cyan", alpha=0.7, label="deconvolved")
    ax3.plot(pt, fftshift(pref), "k", label="original")
    ax3.errorbar(
        [pt[len(pt) / 4]],
        [pp_meas.max() / 2.0],
        xerr=tau / 1e3,
        capsize=5,
        linewidth=2,
    )
    ax3.text(pt[len(pt) / 4], pp_meas.max() * 0.55, "tau", fontdict=dict(size="small"))
    ax3.set_xlim(0, pt[-1])
    legend = ax3.legend(loc="upper right", prop=dict(size="x-small"))
    legend.get_frame().set_alpha(0.5)
    ax3.text(
        0.9,
        0.1,
        "ms",
        fontdict=dict(size="small"),
        va="top",
        ha="right",
        transform=ax3.transAxes,
        bbox=dict(alpha=0.75, fc="white"),
    )

    ax4 = fig.add_subplot(3, 3, 3)
    im = ax4.imshow(
        np.angle(cs_model[:, :mlag] / cs0[:, :mlag]),
        cmap="hsv",
        aspect="auto",
        interpolation="nearest",
        extent=csextent,
    )
    im.set_clim(-3.2, 3.2)
    ax4.text(
        0.9,
        0.9,
        "angle cs_model/cs0",
        fontdict=dict(size="small"),
        va="top",
        ha="right",
        transform=ax4.transAxes,
        bbox=dict(alpha=0.75, fc="white"),
    )

    ax5 = fig.add_subplot(3, 3, 5)
    #    im = ax5.imshow(np.abs(cs[:,:mlag]/cs0[:,:mlag]),
    #                     aspect='auto',interpolation='nearest',extent=csextent)
    #    im.set_clim(0.5,2)
    ax5.plot(
        t / 1e3,
        fftshift(20 * np.log10(np.abs(ht0) / np.abs(ht0).max())),
        label="dB($|h(t)|^2$)",
    )
    ax5.plot(
        t / 1e3,
        fftshift(20 * np.log10(np.abs(ht) / np.abs(ht).max())) - 40.0,
        "r",
        label=r"dB($|\hat{h}(t)|^2$)-40",
    )
    ax5.set_ylim(-80.0, 0)
    ax5.set_xlim(0, t[-1] / 1e3)
    legend = ax5.legend(loc="upper left", prop=dict(size="x-small"))
    legend.get_frame().set_alpha(0.5)
    ax5.text(
        0.9,
        0.1,
        "ms",
        fontdict=dict(size="small"),
        va="top",
        ha="right",
        transform=ax5.transAxes,
        bbox=dict(alpha=0.75, fc="white"),
    )

    ax8 = fig.add_subplot(3, 3, 8)
    maxt0 = t[fftshift(np.abs(ht0)).argmax()]
    maxt = t[fftshift(np.abs(ht)).argmax()]
    if maxt0 < maxt:
        maxt = maxt0
    #    ax8.plot(t-maxt,np.fft.fftshift(20*np.log10(np.abs(ht0)/np.abs(ht0).max())),label='dB($|h(t)|^2$)')
    #    ax8.plot(t-maxt,np.fft.fftshift(20*np.log10(np.abs(ht)/np.abs(ht).max())),'r',label=r'dB($|\hat{h}(t)|^2$)')
    #    ax8.set_ylim(-80.,0)

    ax8.plot(t - maxt, fftshift((np.abs(ht0) / np.abs(ht0).max())), label="$|h(t)|$")
    ax8.plot(
        t - maxt,
        fftshift((np.abs(ht) / np.abs(ht).max())),
        "r",
        label=r"$|\hat{h}(t)|$",
    )

    left = -5 * tau
    right = 20 * tau
    if right > t[-1] - maxt:
        right = t[-1] - maxt
    ax8.set_xlim(left, right)
    legend = ax8.legend(loc="upper right", prop=dict(size="x-small"))
    legend.get_frame().set_alpha(0.5)
    ax8.set_xlabel(r"$\mu$s")

    ax6 = fig.add_subplot(3, 3, 6)
    im = ax6.imshow(
        np.angle(cs[:, :mlag] / cs0[:, :mlag]),
        cmap="hsv",
        aspect="auto",
        interpolation="nearest",
        extent=csextent,
    )
    im.set_clim(-3.2, 3.2)
    ax6.text(
        0.9,
        0.9,
        "angle cs_meas/cs0",
        fontdict=dict(size="small"),
        va="top",
        ha="right",
        transform=ax6.transAxes,
        bbox=dict(alpha=0.75, fc="white"),
    )

    ax7 = fig.add_subplot(3, 3, 7)
    im = ax7.imshow(
        np.log10(np.abs(cs[:, :mlag])),
        aspect="auto",
        interpolation="nearest",
        extent=csextent,
    )
    ax7.text(
        0.9,
        0.9,
        "log|CS meas|",
        fontdict=dict(size="small"),
        va="top",
        ha="right",
        transform=ax7.transAxes,
        bbox=dict(alpha=0.75, fc="white"),
    )
    ax7.set_xlabel("harmonic")
    ax7.set_ylabel("MHz")

    ax9 = fig.add_subplot(3, 3, 9)
    im = ax9.imshow(
        np.angle(cs[:, :mlag]),
        cmap="hsv",
        aspect="auto",
        interpolation="nearest",
        extent=csextent,
    )

    ax9.text(
        0.9,
        0.9,
        "angle(CS meas)",
        fontdict=dict(size="small"),
        va="top",
        ha="right",
        transform=ax9.transAxes,
        bbox=dict(alpha=0.75, fc="white"),
    )
    ax9.set_xlabel("harmonic")

    fname = filename[-50:]
    if len(filename) > 50:
        fname = "..." + fname
    if pharm is not None:
        harmstr = "Harmonics: %d" % pharm
    else:
        harmstr = ""

    snrstr = ""
    taustr = ""
    if tau is not None:
        taustr = "h(t) tau: %.1f" % tau
        if noise is not None:
            snrstr = "snr: %.3f" % noise

    title = "%s isub: %d ipol: %d nopt: %d\n" % (fname, isub, ipol, nopt)
    title += (
        harmstr + " " + taustr + " " + snrstr + " " + (" Feval #%04d Merit: %.3e" % (niter, objval[-1]))
    )
    fig.suptitle(title, size="small")
    canvas = FigureCanvasAgg(fig)
    fname = os.path.join(
        plotdir,
        ("sim_SNR_%.1f_%s_%04d_%04d.pdf" % (noise, source, nopt, niter)),
    )
    canvas.print_figure(fname)