"""
pycyc.objective - the cyclic-spectrum forward model, merit function, and its
gradient (pycyc.tex Appendix "Gradient of the Merit Function"), as pure
functions of explicit parameters rather than a CyclicSolver instance.

This is Stage 2 of the pycyc.py refactor plan: replaces
make_model_cs(CS, ...)/complex_cyclic_merit_lag(ht, CS, ...)/
cyclic_merit_lag(x, CS)/get_params/get_ht, which read bw/ref_freq/
shear_phasors/pad_cyclic_spectra/include_Nyquist/maxharm/exclude_DC/nlag off
a CyclicSolver ("CS") instance. CyclicSolver.modelCS/.updateWavefieldSubint/
.loop now build a CyclicModelParams (pycyc.model) from self and call these
directly; the merit and gradient math is unchanged (verified by the
pytest suite migrated alongside this module, plus new tests below).
"""

__all__ = [
    "make_model_cs",
    "pack_real_params",
    "unpack_real_params",
    "cyclic_merit_and_grad",
    "cyclic_merit_lag_x",
]

import pickle

import numpy as np

from .model import chan_limits_cs, cyclic_padding, total_cyclic_power
from .transforms import cs2cc, shear_spectra, time2freq


def make_model_cs(params, hf, s0):
    """
    Compute the model cyclic spectrum S'_mu(alpha, nu) = H(nu+alpha/2)
    H*(nu-alpha/2) S_mu(alpha) (pycyc.tex eqn 1, before the g(t) gain
    factor), and the sheared H(nu+alpha/2)/H(nu-alpha/2) used to build its
    gradient.
    """
    bw = params.bw
    ref_freq = params.ref_freq
    phasors = params.shear_phasors
    padding = params.pad_cyclic_spectra

    nchan = hf.shape[0]
    nharm = s0.shape[0]
    # profile2cs
    cs = np.repeat(
        s0[np.newaxis, :], nchan, axis=0
    )  # fill the cs model with the harmonic profile for each freq chan

    hfplus, hfminus = shear_spectra(hf, phasors)

    cs = cs * hfplus * np.conj(hfminus)

    # force the Nyquist harmonic to be real-valued
    if params.include_Nyquist:
        cs[:, nharm - 1] = np.abs(cs[:, nharm - 1])

    if padding:
        cs = cyclic_padding(cs, bw, ref_freq)

    return cs, hfplus, hfminus


def pack_real_params(ht, rindex):
    """Pack a complex impulse response `ht` into the real-valued parameter
    vector used by scipy.optimize.fmin_l_bfgs_b: interleaved (Re, Im) pairs
    for every lag except `rindex`, whose imaginary part is dropped (fixing
    the degenerate absolute phase/delay described in pycyc.tex, "Degenerate
    phase")."""
    nlag = ht.shape[0]
    params = np.zeros((2 * nlag - 1,), dtype="float")
    if rindex > 0:
        params[: 2 * (rindex)] = ht[:rindex].view("float")
    params[2 * rindex] = ht[rindex].real
    if rindex < nlag - 1:
        params[2 * rindex + 1 :] = ht[rindex + 1 :].view("float")
    return params


def unpack_real_params(params, rindex):
    """Inverse of pack_real_params."""
    nlag = int((params.shape[0] + 1) / 2)
    ht = np.zeros((nlag,), dtype=np.complex128)
    ht[:rindex] = params[: 2 * rindex].view(np.complex128)
    ht[rindex] = params[2 * rindex]
    ht[rindex + 1 :] = params[2 * rindex + 1 :].view(np.complex128)
    return ht


def _active_mask(nchan, nharm, params):
    """
    True wherever the model cyclic spectrum built by make_model_cs actually
    depends on `ht`; False wherever it is forced to a data-independent
    constant (0) by band-edge padding (pad_cyclic_spectra) or harmonic
    truncation (maxharm).

    Every current caller of cyclic_merit_and_grad passes cs_data that is
    already zero at exactly these positions (CyclicSolver.get_cs applies the
    same padding/maxharm truncation to the data before this function ever
    sees it), so residual == 0 there already and this mask changes nothing
    for them. It matters when that invariant doesn't hold: without it, the
    gradient would pick up a spurious contribution from a term whose
    derivative with respect to ht is, by construction, exactly zero.
    """
    active = np.ones((nchan, nharm), dtype=bool)
    if params.pad_cyclic_spectra:
        for ih in range(nharm):
            imin, imax = chan_limits_cs(ih, nchan, params.bw, params.ref_freq)
            active[:imin, ih] = False
            active[imax:, ih] = False
    if params.maxharm is not None:
        active[:, params.maxharm + 1 :] = False
    return active


def cyclic_merit_and_grad(
    ht, params, s0, cs_data, gain=1.0, rindex=0, dump_residual=False, iprint=False
):
    """
    The objective function: computes the merit (pycyc.tex eqn:merit_function)
    and its Wirtinger gradient dM/dh*(tau) (pycyc.tex Appendix, Equations
    dSconj_dh/dS_dh) with respect to the impulse response `ht`.

    Returns (merit, grad, nonzero), where `nonzero` is the number of nonzero
    entries in the (harmonic-truncated) model, used by CyclicSolver for a
    reduced-chi-squared diagnostic.
    """
    hf = time2freq(ht)
    cs_model, hfplus, hfminus = make_model_cs(params, hf, s0)
    cs_model = cs_model * gain

    if params.maxharm is not None:
        cs_model = cs_model.copy()
        cs_model[:, params.maxharm + 1 :] = 0.0

    extract = cs_model[:, params.exclude_DC :]
    nonzero = np.count_nonzero(extract)

    # WDvS13 Equation 19 and eqn:merit_function of appendix
    merit = (np.abs(extract - cs_data[:, params.exclude_DC :]) ** 2).sum()

    # residual, R = model - data
    residual = cs_model - cs_data

    if dump_residual:
        P_residual = total_cyclic_power(residual)
        P_model = total_cyclic_power(cs_model)
        P_data = total_cyclic_power(cs_data)
        print(
            f"cyclic_merit_and_grad nharm={residual.shape[1]} power in residual={P_residual} model={P_model} data={P_data}"
        )
        filename = f"complex_cyclic_merit_lag_residual_{rindex:03d}.pkl"
        with open(filename, "wb") as fh:
            pickle.dump(residual, fh)
        filename = f"complex_cyclic_merit_lag_data_{rindex:03d}.pkl"
        with open(filename, "wb") as fh:
            pickle.dump(cs_data, fh)

    # zero the gradient's view of the residual wherever the model can't
    # depend on ht in the first place (see _active_mask) -- a no-op under
    # every current caller's invariant that cs_data is already zero there
    active = _active_mask(residual.shape[0], residual.shape[1], params)
    gradient_residual = np.where(active, residual, 0)

    phasors = params.shear_phasors

    # make nchan / nlag copies of the intrinsic profile
    cs0 = np.repeat(s0[np.newaxis, :], params.nlag, axis=0)

    cc1 = cs2cc(gradient_residual * hfminus)
    grad2 = cc1 * phasors * np.conj(cs0)  # WDvS Equation 37
    grad_sum1 = grad2[:, params.exclude_DC :].sum(1)  # sum over all harmonics

    cc1 = cs2cc(np.conj(gradient_residual) * hfplus)
    grad2 = cc1 * np.conj(phasors) * cs0
    grad_sum2 = grad2[:, params.exclude_DC :].sum(1)  # sum over all harmonics

    # WDvS13 Appendix, Equations dSconj_dh/dS_dh: each term carries an
    # explicit factor of g(t_l) from differentiating through the gain-scaled
    # model, independent of the gain already folded into `residual` above.
    grad = gain * (grad_sum1 + grad_sum2)

    if iprint:
        print("merit= %.7e  grad= %.7e" % (merit, (np.abs(grad) ** 2).sum()))

    return merit, grad, nonzero


def cyclic_merit_lag_x(x, params, rindex, s0, cs_data, gain=1.0, dump_residual=False, iprint=False):
    """
    Real-parameter (pack_real_params-packed) wrapper around
    cyclic_merit_and_grad, matching the `func(x, *args) -> (f, g)` signature
    scipy.optimize.fmin_l_bfgs_b expects.

    Unlike the CyclicSolver-threading cyclic_merit_lag this replaces, this
    does not record the merit trace anywhere -- a caller that wants that
    (e.g. CyclicSolver.loop, for its diagnostic plot titles) wraps this in
    its own callback.
    """
    ht = unpack_real_params(x, rindex)
    merit, grad, _nonzero = cyclic_merit_and_grad(
        ht, params, s0, cs_data, gain=gain, rindex=rindex, dump_residual=dump_residual, iprint=iprint
    )
    # multiply by 2 when going from Wirtinger to real/imag derivatives
    grad = pack_real_params(2.0 * grad, rindex)
    return merit, grad
