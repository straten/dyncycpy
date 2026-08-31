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

from __future__ import annotations

__all__ = [
    "make_model_cs",
    "make_model_cs_batched",
    "pack_real_params",
    "unpack_real_params",
    "cyclic_merit_and_grad",
    "cyclic_merit_and_grad_batched",
    "cyclic_merit_lag_x",
]

import logging
import pickle
from typing import Callable

import numpy as np

from .model import CyclicModelParams, chan_limits_cs, cyclic_padding, total_cyclic_power
from .transforms import cs2cc, shear_spectra, shear_spectra_batched, time2freq

logger = logging.getLogger(__name__)

# Signature of the on_residual callback: (residual, cs_model, cs_data) -> None
ResidualCallback = Callable[[np.ndarray, np.ndarray, np.ndarray], None]


def make_model_cs(
    params: CyclicModelParams, hf: np.ndarray, s0: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    # force the Nyquist harmonic to be real-valued: a real intrinsic
    # profile's cyclic spectrum must have a real Nyquist harmonic (Hermitian
    # symmetry of the profile's Fourier transform, same as an FFT's own
    # Nyquist bin). Uses .real (not .abs, which also discards the sign) --
    # see cyclic_merit_and_grad's matching gradient correction, which this
    # must stay paired with: taking .real here changes the Wirtinger
    # derivative of this column with respect to ht, and the gradient must
    # account for that or it silently computes the derivative of the
    # unconstrained (pre-.real) value instead.
    #
    # The trailing .copy() matters on the GPU (xp=cupy, see
    # make_model_cs_batched/cyclic_merit_and_grad_batched): assigning
    # xp.real(cs[:, nharm-1]) (a view aliased with cs itself) directly back
    # into cs[:, nharm-1] left the imaginary component un-zeroed on cupy in
    # one reproduction and correctly zeroed in another with the same shapes
    # and dtype -- inconsistent across otherwise-identical runs, consistent
    # with an unsynchronized read-after-write hazard on the aliased buffer
    # rather than a deterministic cupy behavior difference from numpy.
    # Forcing the real part into an independent array before the assignment
    # (materializing it, rather than assigning a lazy/aliased view) removes
    # the aliasing and was reliably correct in every reproduction attempted.
    # numpy is unaffected either way; the copy costs one small array alloc.
    if params.include_Nyquist:
        cs[:, nharm - 1] = np.real(cs[:, nharm - 1]).copy()

    if padding:
        cs = cyclic_padding(cs, bw, ref_freq)

    return cs, hfplus, hfminus


def make_model_cs_batched(
    params: CyclicModelParams, hf_batch: np.ndarray, s0_batch: np.ndarray, xp=np, fft_module=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batched sibling of make_model_cs: builds the model cyclic spectrum for a
    whole batch of subints (e.g. all of them, or one chunk) in a single
    call. CuPy-by-Claude Stage 1 (see
    /home/willem/.claude/plans/stateful-dreaming-wall.md).

    hf_batch: (batch, nchan). s0_batch: (nharm,) [a single profile shared
    across the batch, e.g. self.ph_ref] or (batch, nharm) [a distinct
    profile per subint, e.g. self.jitter_profiles] -- both broadcast
    correctly against (batch, nchan, nharm) with no reshaping by the
    caller. xp/fft_module: see shear_spectra_batched.

    Returns (cs, hfplus, hfminus), each (batch, nchan, nharm).

    pad_cyclic_spectra=True is not supported here (raises
    NotImplementedError): cyclic_padding's per-harmonic Python loop isn't
    batched/GPU-friendly yet, and cycfista.py/cycsolve.py (this function's
    intended caller, via the FISTA gradient path) both run with
    pad_cyclic_spectra=False.
    """
    if params.pad_cyclic_spectra:
        raise NotImplementedError(
            "make_model_cs_batched does not support pad_cyclic_spectra=True "
            "yet -- see the CuPy port plan's Stage 1 scope note."
        )

    phasors = params.shear_phasors
    nharm = s0_batch.shape[-1]

    hfplus, hfminus = shear_spectra_batched(hf_batch, phasors, xp=xp, fft_module=fft_module)

    # s0_batch[..., newaxis, :] is (1, nharm) for a shared (nharm,) profile
    # or (batch, 1, nharm) for a per-subint (batch, nharm) profile -- either
    # way it broadcasts against (batch, nchan, nharm) with no repeat/copy.
    cs = s0_batch[..., xp.newaxis, :] * hfplus * xp.conj(hfminus)

    # force the Nyquist harmonic to be real-valued -- see make_model_cs's
    # matching comment; must stay paired with cyclic_merit_and_grad_batched's
    # gradient correction below. The .copy() matters on the GPU -- see
    # make_model_cs's comment on the aliased-assignment hazard it works
    # around.
    if params.include_Nyquist:
        cs[:, :, nharm - 1] = xp.real(cs[:, :, nharm - 1]).copy()

    return cs, hfplus, hfminus


def pack_real_params(ht: np.ndarray, rindex: int) -> np.ndarray:
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


def unpack_real_params(params: np.ndarray, rindex: int) -> np.ndarray:
    """Inverse of pack_real_params."""
    nlag = int((params.shape[0] + 1) / 2)
    ht = np.zeros((nlag,), dtype=np.complex128)
    ht[:rindex] = params[: 2 * rindex].view(np.complex128)
    ht[rindex] = params[2 * rindex]
    ht[rindex + 1 :] = params[2 * rindex + 1 :].view(np.complex128)
    return ht


def _active_mask(nchan: int, nharm: int, params: CyclicModelParams) -> np.ndarray:
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
    ht: np.ndarray,
    params: CyclicModelParams,
    s0: np.ndarray,
    cs_data: np.ndarray,
    gain: float = 1.0,
    rindex: int = 0,
    dump_residual: bool = False,
    iprint: bool = False,
    on_residual: ResidualCallback | None = None,
) -> tuple[float, np.ndarray, int]:
    """
    The objective function: computes the merit (pycyc.tex eqn:merit_function)
    and its Wirtinger gradient dM/dh*(tau) (pycyc.tex Appendix, Equations
    dSconj_dh/dS_dh) with respect to the impulse response `ht`.

    Returns (merit, grad, nonzero), where `nonzero` is the number of nonzero
    entries in the (harmonic-truncated) model, used by CyclicSolver for a
    reduced-chi-squared diagnostic.

    dump_residual=True logs a power summary and pickles (residual, cs_data)
    to disk, as before; on_residual, if given, is additionally called as
    on_residual(residual, cs_model, cs_data) so a caller can route the same
    arrays anywhere (a callback, an in-memory buffer, ...) without going
    through the filesystem.
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
        logger.debug(
            "cyclic_merit_and_grad nharm=%d power in residual=%s model=%s data=%s",
            residual.shape[1],
            P_residual,
            P_model,
            P_data,
        )
        filename = f"complex_cyclic_merit_lag_residual_{rindex:03d}.pkl"
        with open(filename, "wb") as fh:
            pickle.dump(residual, fh)
        filename = f"complex_cyclic_merit_lag_data_{rindex:03d}.pkl"
        with open(filename, "wb") as fh:
            pickle.dump(cs_data, fh)

    if on_residual is not None:
        on_residual(residual, cs_model, cs_data)

    # zero the gradient's view of the residual wherever the model can't
    # depend on ht in the first place (see _active_mask) -- a no-op under
    # every current caller's invariant that cs_data is already zero there
    active = _active_mask(residual.shape[0], residual.shape[1], params)
    gradient_residual = np.where(active, residual, 0)

    # make_model_cs forced the Nyquist harmonic's model value to Re(z)
    # instead of the unconstrained complex z = gain*s0*hfplus*conj(hfminus)
    # -- w = Re(z) = (z+z*)/2 has dw/dz = dw/dz* = 1/2, not the identity
    # relationship grad_sum1/grad_sum2 below assume for every other column.
    # Working through the chain rule for |w - data|^2 (data may itself carry
    # a nonzero imaginary part at this harmonic, e.g. measurement noise, even
    # though w is exactly real by construction) collapses to exactly this:
    # replace the residual (and, since it's now real, its own conjugate) at
    # this one column with its real part before it feeds grad_sum1/grad_sum2
    # below. Skipping this previously left a spurious i*Im(residual)*(dz* -
    # dz) term in the gradient wherever the data's Nyquist harmonic wasn't
    # exactly real -- confirmed by finite difference to be a ~36% relative
    # error on synthetic data with no such constraint, and present (though
    # smaller, from real data's own residual-noise imaginary part) on every
    # real run through cycsolve.py --gpu, which requires include_Nyquist.
    # .copy() -- see make_model_cs's comment on the aliased-assignment
    # hazard this works around on the GPU (cyclic_merit_and_grad_batched
    # below shares this exact pattern).
    if params.include_Nyquist:
        nharm = s0.shape[0]
        gradient_residual[:, nharm - 1] = np.real(gradient_residual[:, nharm - 1]).copy()

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
        logger.debug("merit= %.7e  grad= %.7e", merit, (np.abs(grad) ** 2).sum())

    return merit, grad, nonzero


def cyclic_merit_and_grad_batched(
    ht_batch: np.ndarray,
    params: CyclicModelParams,
    s0_batch: np.ndarray,
    cs_data_batch: np.ndarray,
    gain_batch=1.0,
    xp=np,
    fft_module=None,
    dump_residual: bool = False,
    on_residual: ResidualCallback | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batched sibling of cyclic_merit_and_grad: computes the merit and
    Wirtinger gradient for a whole batch of subints (e.g. all of them, or
    one chunk) in a single call. CuPy-by-Claude Stage 1 (see
    /home/willem/.claude/plans/stateful-dreaming-wall.md) -- intended
    caller is CyclicSolver.compute_gradient_batched (the FISTA gradient
    path), not the classical per-subint loop()/outer_loop path.

    ht_batch: (batch, nlag). s0_batch: (nharm,) [shared profile] or
    (batch, nharm) [per-subint, e.g. jitter profiles] -- see
    make_model_cs_batched. cs_data_batch: (batch, nchan, nharm).
    gain_batch: scalar or (batch,).

    Returns (merit, grad, nonzero) as (batch,), (batch, nlag), (batch,)
    arrays -- per-subint, unlike cyclic_merit_and_grad's scalars; a caller
    that wants a batch total (as CyclicSolver.compute_gradient's
    self.merit/self.nterm_merit accumulate today) sums merit/nonzero
    itself.

    dump_residual/on_residual are not supported here (raises
    NotImplementedError if either is passed) -- diagnostics-only, out of
    scope for the initial batched path; nothing on the FISTA path uses
    them today.
    """
    if dump_residual or on_residual is not None:
        raise NotImplementedError(
            "cyclic_merit_and_grad_batched does not support dump_residual/"
            "on_residual (diagnostics-only, out of scope for the initial "
            "batched path -- see the CuPy port plan's Stage 1 scope note)."
        )

    if fft_module is None:
        from .backend import get_fft

        fft_module = get_fft(xp)

    # time2freq's convention (norm="ortho"), batched over the nlag axis
    hf_batch = fft_module.fft(ht_batch, axis=1, norm="ortho")
    cs_model, hfplus, hfminus = make_model_cs_batched(params, hf_batch, s0_batch, xp=xp, fft_module=fft_module)

    gain_arr = xp.asarray(gain_batch)
    cs_model = cs_model * gain_arr[..., xp.newaxis, xp.newaxis]

    if params.maxharm is not None:
        cs_model = cs_model.copy()
        cs_model[:, :, params.maxharm + 1 :] = 0.0

    extract = cs_model[:, :, params.exclude_DC :]
    nonzero = xp.count_nonzero(extract, axis=(1, 2))

    # WDvS13 Equation 19 and eqn:merit_function of appendix
    merit = (xp.abs(extract - cs_data_batch[:, :, params.exclude_DC :]) ** 2).sum(axis=(1, 2))

    # residual, R = model - data
    residual = cs_model - cs_data_batch

    # zero the gradient's view of the residual wherever the model can't
    # depend on ht in the first place (see _active_mask) -- shared across
    # the whole batch (depends only on params, not on data), so computed
    # once here rather than per subint
    active = xp.asarray(_active_mask(residual.shape[1], residual.shape[2], params))
    gradient_residual = xp.where(active[xp.newaxis, :, :], residual, 0)

    # see cyclic_merit_and_grad's matching comment: make_model_cs_batched
    # forced the Nyquist harmonic's model value to Re(z), so its residual's
    # gradient contribution must use Re(residual) here too, not the general
    # complex-residual formula grad_sum1/grad_sum2 below otherwise assume.
    # .copy() -- see make_model_cs's comment on the aliased-assignment
    # hazard this works around on the GPU.
    if params.include_Nyquist:
        nharm = s0_batch.shape[-1]
        gradient_residual[:, :, nharm - 1] = xp.real(gradient_residual[:, :, nharm - 1]).copy()

    phasors = params.shear_phasors

    # (1, nharm) for a shared profile or (batch, 1, nharm) for a per-subint
    # one -- either way broadcasts against grad2's (batch, nlag, nharm),
    # standing in for the non-batched version's explicit
    # np.repeat(s0[np.newaxis, :], params.nlag, axis=0)
    cs0 = s0_batch[..., xp.newaxis, :]

    # cs2cc's convention (norm="ortho"), batched over the radio-frequency/lag axis
    cc1 = fft_module.ifft(gradient_residual * hfminus, axis=1, norm="ortho")
    grad2 = cc1 * phasors * xp.conj(cs0)  # WDvS Equation 37
    grad_sum1 = grad2[:, :, params.exclude_DC :].sum(axis=2)  # sum over all harmonics

    cc1 = fft_module.ifft(xp.conj(gradient_residual) * hfplus, axis=1, norm="ortho")
    grad2 = cc1 * xp.conj(phasors) * cs0
    grad_sum2 = grad2[:, :, params.exclude_DC :].sum(axis=2)  # sum over all harmonics

    # WDvS13 Appendix, Equations dSconj_dh/dS_dh: each term carries an
    # explicit factor of g(t_l) from differentiating through the gain-scaled
    # model, independent of the gain already folded into `residual` above.
    grad = gain_arr[..., xp.newaxis] * (grad_sum1 + grad_sum2)

    return merit, grad, nonzero


def cyclic_merit_lag_x(
    x: np.ndarray,
    params: CyclicModelParams,
    rindex: int,
    s0: np.ndarray,
    cs_data: np.ndarray,
    gain: float = 1.0,
    dump_residual: bool = False,
    iprint: bool = False,
    on_residual: ResidualCallback | None = None,
) -> tuple[float, np.ndarray]:
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
        ht,
        params,
        s0,
        cs_data,
        gain=gain,
        rindex=rindex,
        dump_residual=dump_residual,
        iprint=iprint,
        on_residual=on_residual,
    )
    # multiply by 2 when going from Wirtinger to real/imag derivatives
    grad = pack_real_params(2.0 * grad, rindex)
    return merit, grad
