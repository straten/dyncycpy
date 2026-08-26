"""
pycyc.regularization - wavefield regularization and degenerate-DOF removal.

Covers noise thresholding/shrinkage of the 2D Doppler-delay wavefield,
spectral-entropy minimization of the degenerate per-subint phase (pycyc.tex
Appendix "Minimal Spectral Entropy"), removal of the degenerate
phase/delay/shift projections from the wavefield gradient, and
frequency-response alignment between neighbouring subints.
"""

__all__ = ["apply_threshold", "apply_shrinkage_threshold", "apply_delay_shrinkage_threshold", "noise_power_wavefield", "delay_noise_power_wavefield", "rms_wavefield", "normalize_cs_by_noise_rms", "normalize_cs", "rms_cs", "find_n_largest_indices", "spectral_entropy_grad", "spectral_entropy", "minimize_spectral_entropy", "spectral_entropy_grad_with_delay", "spectral_entropy_with_delay", "minimize_spectral_entropy_with_delay", "circular", "minimize_temporal_phase_noise", "subtract_degenerate_delay_and_phase", "subtract_degenerate_dof", "spectral_shift", "spectral_distance", "minimize_difference", "align_to_neighbour"]

import logging

import numpy as np
from scipy.fft import fft, ifft
from scipy.optimize import minimize
from scipy.signal import fftconvolve

from .transforms import time2freq, freq2time
from .model import chan_limits_cs

logger = logging.getLogger(__name__)

def apply_threshold(x: np.ndarray, threshold: float, kernel=None):
    """
    Any value with abs(x) < threshold is set to zero
    """
    x_power = np.abs(x) ** 2
    if kernel is not None:
        print("apply threshold: smoothing power using supplied kernel")
        x_power = fftconvolve(x_power, kernel, mode="same")
    var_noise = noise_power_wavefield(x_power)
    limit = var_noise * threshold**2
    out = np.heaviside(x_power - limit, 1) * x
    nonz = np.count_nonzero(out)
    sz = np.size(out)
    print(f"apply_threshold: zero={(sz-nonz)*100.0/sz} %")
    return out



def apply_shrinkage_threshold(x: np.ndarray, threshold: float, kernel=None, decay=None):
    """
    abs(x) is decreased by threshold.
    Any resulting value with abs(x) < threshold is set to zero
    """
    x_power = np.abs(x) ** 2
    if kernel is not None:
        print("apply shrinkage threshold: smoothing power using supplied kernel")
        x_power = fftconvolve(x_power, kernel, mode="same")

    var_noise = noise_power_wavefield(x_power)
    limit = np.sqrt(var_noise) * threshold
    shrinkage = limit

    if decay is not None:
        shrinkage = limit * np.exp(-(np.sqrt(x_power) - limit) / (decay * limit))

    # add a small offset to absx to avoid division by zero in next step
    absx = np.abs(x) + np.sqrt(limit) * 1e-6
    out = np.maximum(absx - shrinkage, 0) * x / absx
    nonz = np.count_nonzero(out)
    sz = np.size(out)
    print(f"apply_shrinkage_threshold: zero={(sz-nonz)*100.0/sz} %")
    return out



def apply_delay_shrinkage_threshold(x: np.ndarray, threshold: float, baseline_threshold: float, kernel=None):
    """
    abs(x) is decreased by threshold * delay_noise_power
    Any resulting value with abs(x) < threshold * delay_noise_power is set to zero
    """
    x_power = np.abs(x) ** 2
    if kernel is not None:
        print("apply delay shrinkage threshold: smoothing power using supplied kernel")
        x_power = fftconvolve(x_power, kernel, mode="same")

    var_noise = delay_noise_power_wavefield(x_power, baseline_threshold)
    shrinkage = np.sqrt(var_noise) * threshold

    # add a small offset to absx to avoid division by zero in next step
    absx = np.abs(x) + shrinkage * 1e-6
    out = np.maximum(absx - shrinkage, 0) * x / absx
    nonz = np.count_nonzero(out)
    sz = np.size(out)
    print(f"apply_delay_shrinkage_threshold: zero={(sz-nonz)*100.0/sz} %")
    return out



def noise_power_wavefield(h_power):
    # compute the mean wavefield power over all doppler shifts and a range of negative delays
    nchan = h_power.shape[1]
    start_chan = nchan * 5 // 8
    end_chan = nchan * 7 // 8
    noise_power = h_power[:, start_chan:end_chan]
    norm = np.maximum(np.count_nonzero(noise_power), 1)
    return np.sum(noise_power) / norm



def delay_noise_power_wavefield(power, threshold):
    bias = 1.0 - threshold * np.exp(-threshold) / (1.0 - np.exp(-threshold))
    # print(f"delay_noise_power_wavefield threshold={threshold} bias={bias}")

    ndoppler = power.shape[0]
    power.shape[1]
    # print(f"delay_noise_power_wavefield ndelay={ndelay} ndoppler={ndoppler}")

    # for the initial estimate of noise power as a function of delay,
    # extract a 10-doppler-shift-wide strip at the +/- extrema of Doppler shift
    # (where signal is expected to be low)
    width = 10
    min = (ndoppler - width) // 2
    max = (ndoppler + width) // 2
    edge = power[min:max, :]

    sum_edge = np.sum(edge, axis=0)
    count_edge = np.maximum(np.count_nonzero(edge, axis=0), 1)

    masked_delay_power = sum_edge / count_edge
    for i in range(10):
        masked = np.heaviside(threshold * masked_delay_power - power, 1) * power
        sum_masked = np.sum(masked, axis=0)
        count_masked = np.maximum(np.count_nonzero(masked, axis=0), 1)
        masked_delay_power = sum_masked / (bias * count_masked)

    return masked_delay_power



def rms_wavefield(h):
    # compute rms wavefield rms over all doppler shifts and a range of negative delays
    return np.sqrt(noise_power_wavefield(np.abs(h) ** 2))



def normalize_cs_by_noise_rms(cs, bw, ref_freq):
    nchan = cs.shape[0]
    nharm = cs.shape[1]
    cmin, cmax = chan_limits_cs(nharm, nchan, bw, ref_freq)
    hmin = nharm // 2
    extracted_noise = cs[cmin:cmax, hmin:]
    # print(f'normalize_cs_by_noise_rms nonzero={np.count_nonzero(noise)} size={noise.size}')
    rms = np.sqrt((np.abs(extracted_noise) ** 2).mean())
    return cs / rms, rms



def normalize_cs(cs, bw, ref_freq):
    rms1 = rms_cs(cs, ih=1, bw=bw, ref_freq=ref_freq)
    rmsn = rms_cs(cs, ih=cs.shape[1] - 1, bw=bw, ref_freq=ref_freq)
    normfac = np.sqrt(np.abs(rms1**2 - rmsn**2))
    # print(f"normalize_cs: normfac={normfac}")
    return cs / normfac, normfac



def rms_cs(cs, ih, bw, ref_freq):
    nchan = cs.shape[0]
    imin, imax = chan_limits_cs(ih, nchan, bw, ref_freq)
    rms = np.sqrt((np.abs(cs[imin:imax, ih]) ** 2).mean())
    return rms



def find_n_largest_indices(arr: np.ndarray, n: int) -> list:
    """
    Given a 2D NumPy array and an integer N, this function returns a list of
    the (row, column) pairs for the N largest values in the array.

    This routine is optimized for memory efficiency by using `np.argpartition`
    which avoids a full sort of the flattened array. It finds the indices of
    the N largest values and then converts them to 2D coordinates.

    Written by Gemini.

    Args:
        arr (np.ndarray): A 2D NumPy array of values.
        n (int): The number of largest values to find.

    Returns:
        list: A list of tuples, where each tuple is an (x, y) coordinate
              (row, column) of one of the N largest values.
    """
    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        print("Error: Input must be a 2D NumPy array.")
        return []
    
    if n <= 0:
        print("Error: N must be a positive integer.")
        return []

    # Get a flattened view of the array. This is an O(1) operation
    # that does not create a copy of the data.
    arr_flat = arr.ravel()

    # Determine the number of elements to partition. If N is larger than
    # the total number of elements, we just take all of them.
    kth_element = max(0, arr_flat.size - n)

    # Use `np.argpartition` to find the indices of the N largest elements.
    # This is much more efficient than a full sort as it only guarantees
    # that the kth element is in its sorted position. The elements to the
    # right of it are the largest ones, but not necessarily sorted among
    # themselves. This is all done without creating a copy of the data.
    top_n_flattened_indices = np.argpartition(arr_flat, kth_element)[kth_element:]

    # Map the flattened indices back to 2D coordinates.
    # np.unravel_index is a handy function for this. It takes a
    # flattened index and the shape of the original array and
    # returns the multi-dimensional index.
    row_indices, col_indices = np.unravel_index(
        top_n_flattened_indices, arr.shape
    )

    # Combine the row and column indices into a list of (row, column) tuples.
    result_indices = list(zip(row_indices, col_indices))
    
    # Sort the results by value to match the previous behavior, if desired.
    # This step is optional but provides a consistent output. It can be
    # commented out if sorting is not needed for your use case.
    result_indices.sort(key=lambda coord: arr[coord], reverse=True)

    return result_indices



def spectral_entropy_grad(phi, h_time_delay):
    """
    Calculates the total spectral entropy of the time-to-Doppler forward Fourier transform
    of the input h_time_delay after phase shifting each row/time (except the first) by phi

    Args:
    phi: A 1D array of Ntime-1 real-valued phase shifts (radians) to be applied to each row except the first
    h_time_delay: A 2D array of Ntime * Ndelay complex-values;

    Each row of the dynamic impulse response, h_time_delay, is multiplied by a phasor defined by phi

    Returns:
    The spectral entropy and its gradient with respect to the phase shifts
    """

    Ntime, M = h_time_delay.shape

    phs = np.zeros(Ntime)
    phs[1:] = phi
    phasors = np.exp(1.0j * phs)
    h_time_delay_prime = np.multiply(h_time_delay, phasors[:, np.newaxis])
    h_doppler_delay_prime = fft(h_time_delay_prime, axis=0, norm="ortho")
    power_spectrum = np.abs(h_doppler_delay_prime) ** 2

    total_power = np.sum(power_spectrum)
    power_spectrum /= total_power
    log_power_spectrum = np.log2(power_spectrum + 1e-16)
    entropy = -np.sum(power_spectrum * log_power_spectrum)

    weighted_ifft = ifft((1.0 + log_power_spectrum) * h_doppler_delay_prime, axis=0, norm="ortho")

    gradient = 2.0 / total_power * np.sum(np.imag(np.conj(weighted_ifft) * h_time_delay_prime), axis=1)
    np.sum(gradient**2)
    np.sqrt(np.sum(phi**2) / (Ntime - 1))
    # print(f"rms={rms:.4g} rad; S={entropy} grad power={grad_power:.4}")

    return entropy, gradient[1:]



def spectral_entropy(h_time_delay):
    ntime = h_time_delay.shape[0]
    phi = np.zeros(ntime - 1)
    entropy, grad = spectral_entropy_grad(phi, h_time_delay)
    return entropy



def minimize_spectral_entropy(h_time_delay):
    ntime = h_time_delay.shape[0]
    initial_guess = np.zeros(ntime - 1)

    S_init = spectral_entropy(h_time_delay)

    result = minimize(
        spectral_entropy_grad,
        initial_guess,
        args=(h_time_delay,),
        method="BFGS",
        jac=True,
        callback=circular,
    )

    optimal_phases = np.zeros(ntime)
    optimal_phases[1:] = result.x
    phasors = np.exp(1.0j * optimal_phases)
    h_time_delay *= phasors[:, np.newaxis]

    S_final = spectral_entropy(h_time_delay)

    print(f"minimize_spectral_entropy initial={S_init} final={S_final}")



def spectral_entropy_grad_with_delay(params, h_time_freq):
    """
    Jointly optimizes the per-subint absolute phase phi_l AND delay
    epsilon_l (pycyc.tex "Minimal Spectral Entropy", extended to the
    epsilon_l gradient the tex derives symbolically but never carries to a
    simplified closed form -- see pycyc.tex for the completed derivation).
    Adapted from notebooks/Minimum_Spectral_Entropy_Test.ipynb, the
    author's own validated prototype; verified algebraically identical, for
    the phi part, to spectral_entropy_grad above via the orthonormal-FFT
    identity conj(ifft(X)) = fft(conj(X)).

    Unlike spectral_entropy_grad (phi-only, operating on h_time_delay and
    applying phi as a per-row scalar multiply), this takes h_time_freq --
    H(nu;t), the *frequency*-domain dynamic response -- because epsilon is a
    delay ramp applied in frequency (exp(i*epsilon*nu)), equivalent by the
    Fourier shift theorem to shifting h(tau;t) along tau; the delay-domain
    transform is done internally.

    Args:
    params: A 1D array of 2*(Ntime-1) values: (Ntime-1) real-valued phase
        shifts phi_l (radians), followed by (Ntime-1) real-valued delay
        ramps epsilon_l (radians per normalized frequency bin), applied to
        every row except the first (which is held fixed, breaking the
        overall phi/epsilon degeneracy the same way rindex does for the
        wavefield fit itself).
    h_time_freq: the dynamic frequency response H(nu;t), a 2D array of
        Ntime * Nchan complex values.

    Returns:
    The spectral entropy of h(tau,omega) and its gradient with respect to
    (phi_l, epsilon_l), same concatenated layout as params.
    """
    Ntime, Nchan = h_time_freq.shape

    phs = np.zeros(Ntime)
    eps = np.zeros(Ntime)
    phs[1:] = params[: Ntime - 1]
    eps[1:] = params[Ntime - 1 :]

    nus = np.fft.fftfreq(Nchan)
    h_time_freq_prime = h_time_freq * np.exp(1j * np.outer(eps, nus))
    # d/deps of h_time_freq_prime, with the explicit "i" factor dropped --
    # matches how the phi gradient below never applies its own explicit "i"
    # either, both instead relying on Re[-i*z] = Im[z] at the end.
    dh_time_freq_prime_deps_oni = h_time_freq_prime * nus[np.newaxis, :]

    h_time_delay_prime = ifft(h_time_freq_prime, axis=1, norm="ortho")
    dh_time_delay_prime_deps_oni = ifft(dh_time_freq_prime_deps_oni, axis=1, norm="ortho")

    phasors = np.exp(1.0j * phs)
    h_time_delay_prime = h_time_delay_prime * phasors[:, np.newaxis]
    dh_time_delay_prime_deps_oni = dh_time_delay_prime_deps_oni * phasors[:, np.newaxis]

    h_doppler_delay_prime = fft(h_time_delay_prime, axis=0, norm="ortho")
    power_spectrum = np.abs(h_doppler_delay_prime) ** 2

    total_power = np.sum(power_spectrum)
    power_spectrum = power_spectrum / total_power
    log_power_spectrum = np.log2(power_spectrum + 1e-16)
    entropy = -np.sum(power_spectrum * log_power_spectrum)

    weighted_ifft = ifft((1.0 + log_power_spectrum) * h_doppler_delay_prime, axis=0, norm="ortho")
    conj_weighted = np.conj(weighted_ifft)

    gradient_phs = 2.0 / total_power * np.sum(np.imag(conj_weighted * h_time_delay_prime), axis=1)
    gradient_eps = 2.0 / total_power * np.sum(np.imag(conj_weighted * dh_time_delay_prime_deps_oni), axis=1)

    return entropy, np.concatenate((gradient_phs[1:], gradient_eps[1:]))



def spectral_entropy_with_delay(h_time_freq):
    ntime = h_time_freq.shape[0]
    params = np.zeros(2 * (ntime - 1))
    entropy, _grad = spectral_entropy_grad_with_delay(params, h_time_freq)
    return entropy



def minimize_spectral_entropy_with_delay(h_time_freq, guess_phi=None, guess_eps=None, maxiter=1000):
    """
    In-place joint (phi_l, epsilon_l) spectral-entropy minimization of
    h_time_freq -- see spectral_entropy_grad_with_delay. Off by default in
    CyclicSolver (self.minimize_spectral_entropy_delay); the epsilon fit
    roughly doubles the parameter count of the shipped phi-only
    minimize_spectral_entropy and BFGS convergence at realistic subint
    counts is not guaranteed (observed directly, adapting this from the
    prototype: ~468 subints / 934 parameters left BFGS at maxiter=1000
    still visibly decreasing, not converged) -- hence the explicit maxiter
    and the warning (not silence) on non-convergence, both absent from the
    prototype.

    guess_phi/guess_eps: optional (Ntime-1)-length initial guesses (e.g.
    warm-started from a previous outer-loop pass, or from
    pycyc.profile.fit_profile_shift's per-subint epsilon estimates as a
    cross-check/comparison).

    Returns (phi, epsilon), the fitted (Ntime-1)-length arrays (the fixed
    first-row phi[0]=epsilon[0]=0 convention is not included) -- unlike
    minimize_spectral_entropy, which returns nothing, since here the fitted
    epsilon is itself useful as a diagnostic to compare against the
    profile-domain estimate.
    """
    ntime, nchan = h_time_freq.shape
    initial_guess = np.zeros(2 * (ntime - 1))
    if guess_phi is not None:
        initial_guess[: ntime - 1] = guess_phi
    if guess_eps is not None:
        initial_guess[ntime - 1 :] = guess_eps

    S_init = spectral_entropy_with_delay(h_time_freq)

    result = minimize(
        spectral_entropy_grad_with_delay,
        initial_guess,
        args=(h_time_freq,),
        method="BFGS",
        jac=True,
        callback=circular,
        options={"maxiter": maxiter},
    )
    if not result.success:
        logger.warning(
            "minimize_spectral_entropy_with_delay: BFGS did not converge within maxiter=%d (%s)",
            maxiter,
            result.message,
        )

    params = result.x
    phs = np.zeros(ntime)
    eps = np.zeros(ntime)
    phs[1:] = params[: ntime - 1]
    eps[1:] = params[ntime - 1 :]

    nus = np.fft.fftfreq(nchan)
    h_time_freq_prime = h_time_freq * np.exp(1j * np.outer(eps, nus))
    phasors = np.exp(1.0j * phs)
    h_time_freq[:, :] = h_time_freq_prime * phasors[:, np.newaxis]

    S_final = spectral_entropy_with_delay(h_time_freq)
    logger.info("minimize_spectral_entropy_with_delay initial=%s final=%s", S_init, S_final)

    return phs[1:], eps[1:]



def circular(x):
    x[:] = np.fmod(x, 2.0 * np.pi)



def minimize_temporal_phase_noise(x):
    nspec = x.shape[0]
    xprev = x[0]
    zero = 1.0 + 0.0j
    power = 0.0
    for isub in range(1, nspec):
        z = (np.conj(x[isub]) * xprev).sum()
        z /= np.abs(z)
        x[isub] *= z
        diff = z - zero
        power += np.abs(diff) ** 2
        xprev = x[isub]



def subtract_degenerate_delay_and_phase(h_delay_grad,h_delay):
    """
    Subtract phase and delay terms from the gradient
    """
    h_freq_grad = time2freq(h_delay_grad)
    h_freq = time2freq(h_delay)
    nchan = h_freq.shape[0]

    # phase basis vector
    v_phase = 1.0j * h_freq
    v_phase /= np.sqrt(np.sum(np.abs(v_phase) ** 2))
    # linear phase gradient along frequency basis vector
    v_delay = 1.0j * np.fft.fftfreq(nchan) * h_freq
    v_delay /= np.sqrt(np.sum(np.abs(v_delay) ** 2))

    # verify orthogonality of basis vectors
    dot12 = np.sum(np.conj(v_phase) * v_delay)
    print(f"subtract_degenerate_delay_and_phase dot product of basis vectors: {dot12}")

    # project gradient onto basis vectors, then subtract the projections from the gradient
    a_phase = np.sum(np.conj(v_phase) * h_freq_grad)
    h_freq_grad -= a_phase * v_phase

    b_phase = np.sum(np.conj(v_phase) * h_freq_grad)
    print(f"subtract_degenerate_delay_and_phase projections: {a_phase=} {b_phase=}")

    a_delay = np.sum(np.conj(v_delay) * h_freq_grad)
    h_freq_grad -= a_delay * v_delay

    b_delay = np.sum(np.conj(v_delay) * h_freq_grad)
    print(f"subtract_degenerate_delay_and_phase projections: {a_delay=} {b_delay}")

    return freq2time(h_freq_grad)



def subtract_degenerate_dof(h_time_delay_grad,h_time_delay):
    """
    Subtract phase and two linear phase gradients from the gradient
    """
    ntime = h_time_delay.shape[0]
    for itime in range(ntime):
        h_time_delay_grad[itime, :] = subtract_degenerate_delay_and_phase(h_time_delay_grad[itime, :],h_time_delay[itime, :])
    return h_time_delay_grad



def spectral_shift(theta, hf, index=None):
    """
    Return the input multiplied by a phase and phase gradient

    Args:
    theta: A 1D array of two values: phase, and slope
    hf: the complex-valued spectra that is transformed
    index: the array the slope multiplies. Defaults to the FFT bin
        frequencies np.fft.fftfreq(hf.size), appropriate when hf is a
        frequency-domain spectrum and the slope represents a delay. Pass
        e.g. np.arange(hf.size) instead for a harmonic-indexed profile,
        where the slope represents a rigid pulse-phase-shift rate.

    Returns:
    The transformed input, and the index array used
    """

    phase = theta[0]
    slope = theta[1]
    if index is None:
        index = np.fft.fftfreq(hf.size)
    return hf * np.exp(1j * (phase + slope * index)), index



def spectral_distance(theta, hf_ref, hf, index=None):
    """
    Calculates the magnitude of the difference between the two spectra in hf_ref and hf
    after mupltiplying the second one by a phase and phase gradient

    Args:
    theta: A 1D array of two values: phase, and slope
    hf_ref: the complex-valued spectra used as a reference
    hf: the complex-valued spectra that is aligned to hf_ref by minimizing distance
    index: see spectral_shift

    Returns:
    The distance and its gradient with respect to the 2 parameters in theta
    """

    hfprime, index = spectral_shift(theta, hf, index=index)
    delta = hf_ref - hfprime

    diff = np.sum(np.abs(delta) ** 2)

    del_phase = 1j * hfprime
    del_slope = 1j * index * hfprime

    ddiff_dphs = -2 * np.sum(np.real(np.conj(delta) * del_phase))
    ddiff_dslo = -2 * np.sum(np.real(np.conj(delta) * del_slope))

    return diff, [ddiff_dphs, ddiff_dslo]



def minimize_difference(hf_ref, hf):
    Nchan = hf_ref.size
    initial_guess = np.zeros(2)

    # This value was found with a bit of trial and error, and should be a parameter
    # This limits the duration of the pulse used to estimate delay/shift
    # Including too many samples reduces the S/N of the delay detection
    Nchan_use = 512
    if Nchan > Nchan_use:
        ht_ref = freq2time(hf_ref)
        ht = freq2time(hf)
        use_hf_ref = time2freq(ht_ref[:Nchan_use])
        use_hf = time2freq(ht[:Nchan_use])
    else:
        Nchan_use = Nchan
        use_hf_ref = hf_ref
        use_hf = hf

    align_power = False
    if align_power:
        ht = np.abs(ifft(use_hf)) ** 2
        ht_ref = np.abs(ifft(use_hf_ref)) ** 2
        ccf_power = np.correlate(ht, ht_ref, "same")
        imax = np.argmax(ccf_power) + Nchan // 2
        ph_max = 0
    else:
        ccf = fft(np.conj(use_hf) * use_hf_ref)
        ccf_power = np.abs(ccf) ** 2
        imax = np.argmax(ccf_power)
        ph_max = np.angle(ccf[imax])

    print(f"{imax=} {ph_max=} {Nchan=}")

    limit = 0
    if limit > 0 and imax > limit and imax < Nchan_use - limit:
        print(f"{imax=} beyond {limit=}")
        imax = 0
        ph_max = 0

    initial_guess[0] = ph_max

    if imax < Nchan_use // 2:
        slope = imax
    else:
        slope = imax - Nchan_use

    initial_guess[1] = slope * 2.0 * np.pi

    options = {"maxiter": 1000, "gtol": 1e-9}

    result = minimize(
        spectral_distance,
        initial_guess,
        args=(use_hf_ref, use_hf),
        method="BFGS",
        jac=True,
        options=options,
    )

    alpha = result.x

    best_imax = alpha[1] / (2.0 * np.pi)
    if best_imax > Nchan_use / 2:
        best_imax = best_imax - Nchan_use
        alpha[1] = best_imax * 2.0 * np.pi
        print(f"{best_imax=}")

    hf[:], nus = spectral_shift(alpha, hf)

    # cross-correlation at lag 0
    ccf0 = np.sum(np.conj(hf) * hf_ref)
    # total spectral power in reference spectrum
    tsp0 = np.sum(np.abs(hf_ref) ** 2)
    tsp = np.sum(np.abs(hf) ** 2)
    R = ccf0 / np.sqrt(tsp0 * tsp)
    return R



def align_to_neighbour(h_time_freq):
    nt, nf = h_time_freq.shape
    hf0 = h_time_freq[0]
    for it in range(1, nt):
        hf = h_time_freq[it]
        R = minimize_difference(hf0, hf)
        # try to skip intervals with bad fit / low S/N
        if np.abs(R) > 0.05:
            hf0 = hf
        else:
            print(f"align_to_neighbour i={it} {R=}")

        h_time_freq[it, :] = hf

