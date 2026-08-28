"""
pycyc.solver - CyclicSolver orchestration.

The merit/gradient/model math is in pycyc.objective/pycyc.profile (Stage 2
of the refactor plan) as pure functions of an explicit CyclicModelParams
rather than a CyclicSolver instance; PSRFITS/pickle I/O is in pycyc.io
(_IOMixin, Stage 3) and diagnostic plotting is in the top-level plotting.py
(also Stage 3). What's left here is orchestration: CyclicSolver's own
state and the methods that tie the above together.
"""

import logging

logger = logging.getLogger(__name__)

try:
    import psrchive
except Exception:
    logger.info("pycyc.py: psrchive python libraries not found. You will not be able to load psrchive files.")
import concurrent.futures
import dataclasses
import os
import pickle

import numpy as np
import scipy
import scipy.optimize
from scipy.fft import fftshift, irfft
from scipy.signal.windows import kaiser

from plotting import plot_current_solution, plot_Doppler_vs_delay

from .backend import get_fft, get_xp, to_device, to_host
from .io import _IOMixin
from .io_utils import *
from .jitter import (
    compute_residuals,
    fit_jitter_basis,
    reconstruct_jittered_profile,
    subspace_principal_angle,
)
from .model import *
from .objective import (
    cyclic_merit_and_grad,
    cyclic_merit_and_grad_batched,
    cyclic_merit_lag_x,
    make_model_cs,
    pack_real_params,
    unpack_real_params,
)
from .profile import solve_profile_and_gain, solve_profile_and_gain_batched
from .regularization import *
from .transforms import *


class CyclicSolver(_IOMixin):
    def __init__(
        self,
        filename=None,
        statefile=None,
        offp=None,
        tscrunch=None,
        zap_edges=None,
        pscrunch=False,
        maxchan=None,
        maxharm=None,
    ):
        """
        *offp* : passed to the load method for selecting an off pulse region (optional).
        *tscrunch* : passed to the load method for averaging subintegrations
        *offp*: tuple (start,end) off-pulse phase bin range for normalizing the bandpass
        *maxchan*: Top channel index to use.
            Can be used to pull out one subband from a file which contains multiple subbands
        *tscrunch* : average down by a factor of 1/tscrunch
            (e.g. if tscrunch = 2, average every pair of subints)
        *pscrunch* : average the polarisations
        """

        self.zap_edges = zap_edges
        self.pscrunch = pscrunch
        self.tscrunch = tscrunch
        self.offp = offp
        self.maxchan = maxchan
        self.maxharm = maxharm
        self.save_cyclic_spectra = False
        self.pad_cyclic_spectra = True
        self.filenames = []
        self.nspec = 0
        self.nsubint = 0
        # intrinsic pulse profile in phase domain
        self.pp_intrinsic = None
        self.pp_scattered = None
        # intrinsic profile harmonics (Fourier transform of intrinsic profile)
        self.intrinsic_ph = None
        self.intrinsic_ph_sum = None
        self.intrinsic_ph_sumsq = None
        self.ph_numer = None
        self.ph_denom = None
        self.shear_phasors = None
        self.optimal_gains = None
        self.cs_norm = None
        self.hf_prev = None
        self.iprint = False
        self.make_plots = False
        self.dump_residual = False
        self.dump_gradient = False
        self.niter = 0

        self.mean_time_offset = 0
        self.nthread = 1

        # modelling options

        # By default, use the integrated pulse profile for all sub-integrations
        self.use_integrated_profile = True

        # Remove the baseline from input data
        self.remove_baseline = True

        # omit the DC spin harmonic from the definition of the merit function
        # set to 0 or 1; this flag is used as an integer
        self.exclude_DC = 1

        # incldue the Nyquist spin harmonic
        # set to 0 or 1; this flag is used as an integer
        self.include_Nyquist = 1

        # maintain constant total power in the wavefield
        self.conserve_wavefield_energy = False

        # set the wavefield at all negative delays to zero
        self.enforce_causality = False

        # set the wavefield to a real value at the origin
        self.enforce_real_at_origin = False

        # set harmonically related elements of the wavefield gradient to zero
        self.zap_gradient_harmonics = 0

        # reduce temporal phase noise by minimizing the spectral entropy
        self.minimize_spectral_entropy = False

        # also jointly fit a per-subint delay (epsilon_t) during spectral
        # entropy minimization, not just the per-subint phase -- off by
        # default: roughly doubles the parameter count of the phi-only fit
        # above, and BFGS convergence at realistic subint counts is not
        # guaranteed (see pycyc.regularization.minimize_spectral_entropy_with_delay).
        # Only takes effect when self.minimize_spectral_entropy is also True.
        self.minimize_spectral_entropy_delay = False

        # multiply the wavefiled by a phase that makes real and imaginary parts orthognonal
        self.enforce_orthogonal_real_imag = False

        # align the phases of time-adjacent impulse responses computed from the wavefield
        self.reduce_temporal_phase_noise = False

        # align the phases of time-adjacent impulse responses computed from the wavefield gradient
        self.reduce_temporal_phase_noise_grad = False

        # align the phase and delay of time-adjacent frequency responses computed from the wavefield
        self.align_frequency_responses = False

        # set all wavefield components less than theshold*rms to zero
        # the rms is computed over all doppler shifts between 5/8 and 7/8 of the largest delay
        self.noise_threshold = None

        # set all wavefield components less than theshold*rms to zero, after shrinking them by the same amount
        # the rms is computed over all doppler shifts between 5/8 and 7/8 of the largest delay
        self.noise_shrinkage_threshold = None

        # set all wavefield components less than theshold*delay_noise to zero,
        # after shrinking them by the same amount.  For a given delay, delay_noise is the standard deviation
        # over all doppler shifts below delay_noise_selection_threshold times the mean (corrected for bias)
        self.delay_noise_shrinkage_threshold = None
        self.delay_noise_selection_threshold = None

        # exponential decay scale for the amount of shrinkage
        self.noise_shrinkage_decay = None

        # when thresholding, smooth wavefield power using a Kaiser window with the specified duty cycle
        self.noise_smoothing_duty_cycle = None
        # default Kaiser smoothing beta factor (6 = similar to Hann)
        # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.kaiser.html)
        self.noise_smoothing_beta = 6

        # simultaneously fit for the instrinsic cyclic spectrum (not recommended - introduces degeneracies)
        self.ml_profile = False

        # include separate temporal gain variations in the model
        self.model_gain_variations = False

        # model genuine per-subint pulse jitter with a reduced-rank (PCA)
        # basis (pycyc.jitter) instead of forcing a single profile shared
        # across every subint -- see outer_loop. Off by default: this is
        # new methodology (the jitter/degeneracy plan), not a drop-in
        # replacement for the existing fixed-profile behavior.
        self.model_jitter = False
        # outer_loop passes before the jitter model switches on -- lets the
        # wavefield fit reach a rough convergence first, so the jitter model
        # isn't fit against transient wavefield-model error rather than
        # genuine jitter (see the jitter/degeneracy plan's Stage 4 design).
        self.jitter_warmup_passes = 2
        # optional cap on the PCA rank chosen by pycyc.jitter.fit_jitter_basis
        self.jitter_max_rank = None
        # current per-subint jitter-aware profile (nsubint, nharm), or None
        # to use the ordinary pooled/per-subint profile selection below.
        # Set by outer_loop's jitter refresh step; consulted by loop() and
        # updateWavefieldSubint() ahead of self.use_integrated_profile.
        self.jitter_profiles = None
        # diagnostics from the most recent jitter refresh (see outer_loop)
        self.jitter_rank = 0
        self._jitter_basis = None
        self.jitter_principal_angle = None

        # CuPy port (see /home/willem/.claude/plans/stateful-dreaming-wall.md):
        # compute_gradient_batched's device/chunking knobs. use_gpu is off
        # by default -- experimental, opt-in; requires cupy (see
        # pycyc.backend.get_xp) only when actually set True.
        self.use_gpu = False
        # subints processed per batched call; None means all self.nspec at
        # once (fine on CPU). GPU VRAM budgets require a real chunk size --
        # see the CuPy port plan's Stage 3 memory-budget notes.
        self.gpu_chunk_size = None
        # dtype used on-device when self.use_gpu is True (irrelevant
        # otherwise): complex64 by default for VRAM headroom, validated
        # against complex128 CPU results in tests/test_batched_gpu.py.
        # None means "don't downcast" (use whatever dtype the source
        # arrays already are, i.e. complex128).
        self.gpu_dtype = np.complex64
        # device-resident cache of self.cyclic_spectra/self.shear_phasors
        # (populated lazily by compute_gradient_batched, keyed on the host
        # array's id() plus gpu_dtype so a reload/reassignment invalidates
        # it automatically) -- avoids re-transferring these across every
        # FISTA iteration. Assumes self.cyclic_spectra/self.shear_phasors
        # aren't mutated *in place* once the FISTA loop starts calling
        # compute_gradient_batched (today's in-place tapering happens
        # earlier, during initProfile/load).
        self._gpu_cyclic_spectra = None
        self._gpu_cyclic_spectra_key = None
        self._gpu_phasors = None
        self._gpu_phasors_key = None

        # normalize each cyclic spectrum
        self.normalize_cyclic_spectra = False

        # derive a first guest for the wavefield using the harmonic with the highest S/N
        self.first_wavefield_from_best_harmonic = 0

        # delay the initial wavefield estimate by this many samples
        self.first_wavefield_delay = 0

        # taper data (cyclic spectra) in frequency using the specified window
        # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html
        # Examples:
        #   spectral_window = ('kaiser', 8.0)
        #   spectral_window = ('tukey' 0.25)
        self.spectral_window = None

        # taper data (cyclic spectra) in time using the specified window
        self.temporal_window = None

        # taper wavefield along Doppler axis using the specified window
        self.doppler_window = None
        self.doppler_taper = None

        # by default, do not filter
        self.low_pass_filter_Doppler = 1.0

        # taper wavefield along delay axis using the specified window
        self.delay_window = None
        self.delay_taper = None

        # initial guess for the dynamic response
        self.initial_h_time_freq = None

        # roll the initial guess to place peak in power at zero delay
        self.zero_initial_delay = False

        # roll the initial guess to place peak in power at specified delay
        self.roll_initial_guess = 0

        # zap the edges of the initial_guess
        self.zap_initial_guess = False

        # project the gradient onto the subspace that is orthogonal to the null space direction defined by degenerate phase
        self.subtract_degenerate_projections = False

        if filename:
            self.load(filename)

        elif statefile:
            self.loadState(statefile)

        self.statefile = statefile

    def _model_params(self):
        """Build the CyclicModelParams (pycyc.model) describing the current
        forward model, for make_model_cs/cyclic_merit_and_grad/
        solve_profile_and_gain (pycyc.objective/pycyc.profile)."""
        return CyclicModelParams(
            bw=self.bw,
            ref_freq=self.ref_freq,
            shear_phasors=self.shear_phasors,
            pad_cyclic_spectra=self.pad_cyclic_spectra,
            include_Nyquist=self.include_Nyquist,
            maxharm=self.maxharm,
            exclude_DC=self.exclude_DC,
            nlag=self.nlag,
        )

    def modelCS(self, ht=None, hf=None):
        """
        Convenience function for computing modelCS using ref profile

        Call as modelCS(ht) for time domain or modelCS(hf=hf) for freq domain
        """
        if ht is not None:
            hf = time2freq(ht)
        cs, _hfplus, _hfminus = make_model_cs(self._model_params(), hf, self.ph_ref)

        return cs

    def initProfile(self, loadFile=None, maxinitharm=None, maxsubint=None):
        """
        Initialize the reference profile

        If loadFile is not specified, will compute an initial profile from the data
        If loadFile ends with .txt, it is assumed to be a filter_profile output file
        If loadFile ends with .npy, it is assumed to be a numpy data file
        If loadFile ends with .fits, it is assumed to be a PSRFITS file

        Resulting profile is assigned to self.pp_intrinsic
        The results of this routine have been checked to agree with filter_profile -i

        *maxinitharm* : zero harmonics above this one in the initial profile
            (acts to smooth/denoise) (optional)

        """

        if maxsubint is not None:
            self.nsubint = maxsubint

        self.nspec = self.nsubint

        self.hf_prev = np.ones((self.nchan,), dtype=np.complex128)

        if self.initial_h_time_freq is None:
            self.expected_power = self.nchan * self.nspec
            self.h_doppler_delay = np.zeros((self.nspec, self.nchan), dtype=np.complex128)
            self.h_doppler_delay[0, self.first_wavefield_delay] = np.sqrt(self.expected_power)
            self.h_time_delay = freq2time(self.h_doppler_delay, axis=0)

            # ensure that delta-function yields expected frequency response at all times
            for isub in range(self.nspec):
                ht = self.h_time_delay[isub]
                hf = time2freq(ht)
                for ichan in range(self.nchan):
                    if np.abs(hf[ichan] - 1.0) > 1e-6:
                        logger.info(f"unexpected initial response[{ichan}]={hf[ichan]}")

        else:
            self.expected_power = np.sum(np.abs(self.initial_h_time_freq) ** 2)
            current_shape = (self.nspec, self.nchan)
            if self.initial_h_time_freq.shape != current_shape:
                logger.info(f"padding input shape={self.initial_h_time_freq.shape} to {current_shape=}")
                h_time_freq = pad_wavefield(self.initial_h_time_freq, current_shape)
                self.h_time_delay = freq2time(h_time_freq, axis=1)
                self.h_doppler_delay = time2freq(self.h_time_delay, axis=0)
                plot_Doppler_vs_delay(
                    self.h_doppler_delay, self.mean_time_offset, self.bw, "input_wavefield_after_padding.png"
                )
            else:
                h_time_freq = self.initial_h_time_freq
                self.h_time_delay = freq2time(h_time_freq, axis=1)
                self.h_doppler_delay = time2freq(self.h_time_delay, axis=0)

            if self.zero_initial_delay:
                h_time_power = np.sum(np.abs(self.h_time_delay) ** 2, axis=0)
                peak = np.argmax(h_time_power)
                logger.info(f"{peak=}")
                self.h_time_delay = np.roll(self.h_time_delay, -peak, axis=1)
                self.h_doppler_delay = time2freq(self.h_time_delay, axis=0)

            if self.enforce_real_at_origin:
                self.h_doppler_delay[0, 0] = np.real(self.h_doppler_delay[0, 0])

        if self.iprint:
            logger.info(f"ORIGIN AMPLITUDE: {self.h_doppler_delay[0,0]}")

        self.noise_smoothing_kernel = None
        if self.noise_smoothing_duty_cycle is not None:
            ashape = np.asarray(self.h_doppler_delay.shape)
            wshape = np.round(ashape * self.noise_smoothing_duty_cycle)
            logger.info(f"noise smoothing kernel shape: {wshape}")
            kernel = np.outer(
                kaiser(wshape[0], self.noise_smoothing_beta),
                kaiser(wshape[1], self.noise_smoothing_beta),
            )
            self.noise_smoothing_kernel = kernel / np.sum(kernel)  # Normalize the kernel

        if self.spectral_window is not None:
            spectral_taper = scipy.signal.get_window(self.spectral_window, self.nchan)
            for ichan in range(self.nchan):
                self.cyclic_spectra[:, :, ichan, :] *= spectral_taper[ichan]

        if self.temporal_window is not None:
            temporal_taper = scipy.signal.get_window(self.temporal_window, self.nsubint)
            for ispec in range(self.nsubint):
                self.cyclic_spectra[ispec] *= temporal_taper[ispec]

        if self.doppler_window is not None:
            self.doppler_taper = fftshift(scipy.signal.get_window(self.doppler_window, self.nsubint))

        if self.delay_window is not None:
            self.delay_taper = fftshift(scipy.signal.get_window(self.delay_window, self.nchan))

        if self.first_wavefield_from_best_harmonic:
            self.compute_first_wavefield_from_best_harmonic()

        self.dynamic_spectrum = np.zeros((self.nsubint, self.npol, self.nchan))
        self.first_harmonic_spectrum = np.zeros((self.nsubint, self.npol, self.nchan), dtype=np.complex128)
        self.optimized_filters = np.zeros((self.nsubint, self.nchan), dtype=np.complex128)
        self.intrinsic_profiles = np.zeros((self.nsubint, self.npol, self.nbin))
        self.scattered_profiles = np.zeros((self.nsubint, self.nbin))

        if (self.optimal_gains is None) or (len(self.optimal_gains) != self.nsubint):
            self.optimal_gains = np.ones(self.nsubint)

        self.maxinitharm = maxinitharm
        self.save_dynamic_spectrum = True
        self.save_cs_norm = True
        self.updateProfile()
        self.save_dynamic_spectrum = False
        self.save_cs_norm = False

        # The first guess for the intrinsic profile should be loaded afer updateProfile creates various arrays
        if loadFile:
            if loadFile.endswith(".npy"):
                self.pp_intrinsic = np.load(loadFile)
            elif loadFile.endswith(".txt"):
                self.pp_intrinsic = loadProfile(loadFile)
            elif loadFile.endswith(".fits") or loadFile.endswith(".ar"):
                self.load_initial_profile(loadFile)
            else:
                raise Exception("Filename must end with .txt or .npy to indicate type")

        if self.roll_initial_guess != 0:
            delay = self.roll_initial_guess / (self.bw * 1e6)
            phase = delay * self.ref_freq
            bins = phase * self.nbin
            phase_roll = -int(bins)
            logger.info(
                f"roll initial guesses by {self.roll_initial_guess} time samples and {phase_roll} phase bins"
            )
            self.intrinsic_ph = np.roll(self.intrinsic_ph, phase_roll, axis=1)
            self.intrinsic_profiles = np.roll(self.intrinsic_profiles, phase_roll, axis=2)
            self.scattered_profiles = np.roll(self.scattered_profiles, phase_roll, axis=1)
            self.pp_intrinsic = np.roll(self.pp_intrinsic, phase_roll)
            self.pp_scattered = np.roll(self.pp_scattered, phase_roll)
            self.h_time_delay = np.roll(self.h_time_delay, self.roll_initial_guess, axis=1)
            self.h_doppler_delay = time2freq(self.h_time_delay, axis=0)

    def compute_first_wavefield_from_best_harmonic(self):
        initial_total_power = np.sum(np.abs(self.h_doppler_delay) ** 2)
        maxharm = np.minimum(self.first_wavefield_from_best_harmonic, self.nharm)
        sn = np.zeros(maxharm)

        # search for the harmonic with the highest Doppler/delay power S/N
        for harm in range(maxharm):
            logger.info(f"harmonic={harm}")
            # extract the harmonic and sum over polarizations
            time_freq = np.sum(self.cyclic_spectra[:, :, :, harm], axis=1)
            trial_wavefield = time2freq(freq2time(time_freq, axis=1), axis=0)
            power = np.abs(trial_wavefield) ** 2

            # estimate the S/N for this trial wavefield

            # first, take a slice of noise at extreme Doppler shift
            width = 10
            imin = (self.nspec - width) // 2
            imax = (self.nspec + width) // 2
            noise_slice = power[imin:imax, :]

            noise_power = np.mean(noise_slice)
            total_power = np.mean(power)
            sn[harm] = np.sqrt(total_power / noise_power)
            logger.info(f"harmonic={harm} S/N={sn[harm]}")

        best_harmonic = np.argmax(sn)
        logger.info(f"best harmonic={best_harmonic}")

        # extract the harmonic and sum over polarizations
        time_freq = np.sum(self.cyclic_spectra[:, :, :, best_harmonic], axis=1)
        self.h_doppler_delay = time2freq(freq2time(time_freq, axis=1), axis=0)
        power = np.abs(self.h_doppler_delay) ** 2

        # set the power at the origin equal to the logarithmic/geometric mean of its neighbours
        log_sum = 0
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i != 0 or j != 0:
                    log_sum += np.log10(power[i, j])
        log_mean = log_sum / 8
        mean_amp = pow(10, 0.5 * log_mean)
        zero_amp = np.abs(self.h_doppler_delay[0, 0])
        logger.info(f"amplitude[0,0] current={zero_amp} new={mean_amp}")

        if self.delay_noise_shrinkage_threshold is not None:
            logger.info(f"delay_noise_shrinkage_threshold={self.delay_noise_shrinkage_threshold}")
            np.copyto(
                self.h_doppler_delay,
                apply_delay_shrinkage_threshold(
                    self.h_doppler_delay,
                    self.delay_noise_shrinkage_threshold,
                    self.delay_noise_selection_threshold,
                    self.noise_smoothing_kernel,
                ),
            )

        self.h_doppler_delay[0, 0] *= mean_amp / zero_amp
        self.h_doppler_delay[:, self.nchan // 2 :] = 0.0

        power = np.abs(self.h_doppler_delay) ** 2
        total_power = np.sum(power)
        scale_factor = np.sqrt(initial_total_power / total_power)
        logger.info(f"total power original={initial_total_power} new={total_power} scale={scale_factor}")
        self.h_doppler_delay *= scale_factor

    def solve(self, **kwargs):
        """
        Construct an iterative solution to the IRF using multiple subintegrations
        """
        if kwargs.pop("restart", False):
            self.nopt = 0
        savefile = kwargs.pop(
            "savebase",
            os.path.abspath(self.filename) + ("_%02d.cysolve.pkl" % self.nloop),
        )

        for isub in range(self.nsubint):
            kwargs["isub"] = isub
            self.loop(**kwargs)
            logger.info("Saving after nopt: %s", self.nopt)
            self.saveState(savefile)

        self.nloop += 1

    def get_dof(self):
        # while experimenting with maxharm, nfree can be greater than nterm
        if self.nfree_parameters < self.nterm_merit:
            return self.nterm_merit - self.nfree_parameters
        return 1

    def get_reduced_chisq(self):
        return self.merit / self.get_dof()

    def updateProfileSubint(self, isub):
        for ipol in range(self.npol):
            if self.save_cyclic_spectra:
                cs = self.cyclic_spectra[isub, ipol]
            else:
                ps = self.data[isub, ipol]  # dimensions will now be (nchan,nbin)
                cs, norm = self.get_cs(ps)
                self.cs_norm[isub, ipol] = norm

            if self.save_dynamic_spectrum:
                self.dynamic_spectrum[isub, ipol, :] = np.real_if_close(cs[:, 0])
                self.first_harmonic_spectrum[isub, ipol, :] = cs[:, 1]

            ht = self.h_time_delay[isub]
            hf = time2freq(ht)

            update_gain = self.model_gain_variations and ipol == 0

            if self.compute_scattered_profile:
                ph = fscrunch_cs(cs, bw=self.bw, ref_freq=self.ref_freq, padding=self.pad_cyclic_spectra)
                # print(f"updateProfileSubint {ph.shape=}")
                pp = self.harm2phase(ph)
                self.scattered_profiles[isub, :] += pp

            ph, gain, ph_numer, ph_denom = self.optimize_profile(cs, hf, self.bw, self.ref_freq, update_gain)

            self.ph_numer[ipol, isub] = ph_numer
            self.ph_denom[ipol, isub] = ph_denom

            if update_gain:
                self.optimal_gains[isub] = gain

            if self.exclude_DC:
                ph[0] = 0.0
            if self.maxinitharm:
                ph[self.maxinitharm :] = 0.0
            pp = self.harm2phase(ph)

            if self.iprint:
                logger.info(f"update profile isub={isub}/{self.nsubint}")

            self.intrinsic_profiles[isub, ipol, :] = pp
            return 0

    def updateProfile_batched(self):
        """
        Batched sibling of the per-subint updateProfileSubint loop inside
        updateProfile -- dispatched there when self.use_gpu is True,
        mirroring updateWavefield's dispatch to compute_gradient_batched.
        CuPy port Stage 5 (see
        /home/willem/.claude/plans/stateful-dreaming-wall.md).

        Requires self.save_cyclic_spectra=True (same as
        compute_gradient_batched) and self.include_Nyquist=True: with
        include_Nyquist=False, self.harm2phase's else-branch assumes a
        single 1-D profile (ph.shape[0] is treated as a harmonic count),
        not a (batch, nharm) array.

        Does not support self.compute_scattered_profile or
        self.save_dynamic_spectrum (raises NotImplementedError if either
        is True): both are one-time-setup-only features, only ever True
        on initProfile()'s own first updateProfile() call -- cycsolve.py
        sets self.use_gpu only after initProfile() completes, specifically
        so this method never has to see either True.
        """
        if not self.save_cyclic_spectra:
            raise NotImplementedError("updateProfile_batched requires self.save_cyclic_spectra=True.")
        if not self.include_Nyquist:
            raise NotImplementedError(
                "updateProfile_batched requires self.include_Nyquist=True "
                "(self.harm2phase's include_Nyquist=False branch assumes a "
                "single 1-D profile, not batched)."
            )
        if self.compute_scattered_profile:
            raise NotImplementedError(
                "updateProfile_batched does not support compute_scattered_profile "
                "-- a one-time-setup-only feature; set self.use_gpu after "
                "initProfile() completes, not before."
            )
        if self.save_dynamic_spectrum:
            raise NotImplementedError(
                "updateProfile_batched does not support save_dynamic_spectrum "
                "-- a one-time-setup-only feature; set self.use_gpu after "
                "initProfile() completes, not before."
            )

        xp = get_xp(self.use_gpu)
        fft_module = get_fft(xp)
        chunk_size = self.gpu_chunk_size or self.nspec
        dtype = self.gpu_dtype if self.use_gpu else None

        params = self._model_params()
        params_xp = dataclasses.replace(params, shear_phasors=self._device_phasors(xp, dtype))
        cyclic_spectra_dev = self._device_cyclic_spectra(xp, dtype)

        for ipol in range(self.npol):
            update_gain = self.model_gain_variations and ipol == 0
            cs_full = cyclic_spectra_dev[:, ipol]

            # intrinsic_ph_sum/intrinsic_ph_sumsq are (nharm,), shared
            # across the whole batch (running sums from the *previous*
            # pass, not per-subint) -- transfer once per ipol, not per chunk
            intrinsic_ph_sum_dev = (
                self._to_device_dtype(self.intrinsic_ph_sum, xp, dtype) if update_gain else None
            )
            intrinsic_ph_sumsq_dev = (
                self._to_device_dtype(self.intrinsic_ph_sumsq, xp, dtype) if update_gain else None
            )

            for start in range(0, self.nspec, chunk_size):
                stop = min(start + chunk_size, self.nspec)
                sl = slice(start, stop)

                ht_chunk = self._to_device_dtype(self.h_time_delay[sl], xp, dtype)
                hf_chunk = fft_module.fft(ht_chunk, axis=1, norm="ortho")  # time2freq's convention, batched
                cs_chunk = cs_full[sl]

                ph_chunk, gain_chunk, ph_numer_chunk, ph_denom_chunk = solve_profile_and_gain_batched(
                    cs_chunk,
                    hf_chunk,
                    params_xp,
                    update_gain,
                    intrinsic_ph_sum_dev,
                    intrinsic_ph_sumsq_dev,
                    xp=xp,
                    fft_module=fft_module,
                )

                self.ph_numer[ipol, sl] = to_host(ph_numer_chunk)
                self.ph_denom[ipol, sl] = to_host(ph_denom_chunk)

                if update_gain:
                    self.optimal_gains[sl] = to_host(xp.real(gain_chunk))

                ph_host = to_host(ph_chunk)
                if self.exclude_DC:
                    ph_host[:, 0] = 0.0
                if self.maxinitharm:
                    ph_host[:, self.maxinitharm :] = 0.0

                # harm2phase (scipy.fft-bound, host-only) transforms along
                # the last axis by default, which is exactly the batched
                # semantics needed here when include_Nyquist=True (see the
                # guard above) -- no changes to harm2phase itself needed
                self.intrinsic_profiles[sl, ipol, :] = self.harm2phase(ph_host)

    def updateProfile(self):
        """
        Update the reference profile

        Resulting profile is assigned to self.pp_intrinsic
        """

        self.compute_scattered_profile = False
        if self.pp_scattered is None:
            self.compute_scattered_profile = True

        if (self.ph_numer is None) or (self.ph_numer.shape != (self.npol, self.nspec, self.nharm)):
            self.ph_numer = np.zeros((self.npol, self.nspec, self.nharm), dtype=np.complex128)
        else:
            self.ph_numer.fill(0.0)

        if (self.ph_denom is None) or (self.ph_denom.shape != (self.npol, self.nspec, self.nharm)):
            self.ph_denom = np.zeros((self.npol, self.nspec, self.nharm), dtype=np.complex128)
        else:
            self.ph_denom.fill(0.0)

        if (self.intrinsic_ph is None) or (self.intrinsic_ph.shape != (self.npol, self.nharm)):
            self.intrinsic_ph = np.zeros((self.npol, self.nharm), dtype=np.complex128)
        else:
            self.intrinsic_ph.fill(0.0)

        if self.cs_norm is None:
            self.cs_norm = np.zeros((self.nsubint, self.npol))

        if self.shear_phasors is None:
            self.shear_phasors = create_shear_phasors(self.nchan, self.nharm, self.bw, self.ref_freq)

        # initialize profile from data
        # the results of this routine have been checked against filter_profile and they perform the same

        self.normalize(self.h_doppler_delay)

        self.h_time_delay = freq2time(self.h_doppler_delay)

        if self.align_frequency_responses:
            logger.info(f"reduce temporal phase and delay noise")
            h_time_freq = time2freq(self.h_time_delay, axis=1)
            align_to_neighbour(h_time_freq)
            self.h_time_delay = freq2time(h_time_freq, axis=1)
            self.h_doppler_delay = time2freq(self.h_time_delay, axis=0)

        if self.reduce_temporal_phase_noise:
            logger.info(f"reduce temporal phase noise")
            minimize_temporal_phase_noise(self.h_time_delay)
            self.h_doppler_delay = time2freq(self.h_time_delay, axis=0)

        if self.minimize_spectral_entropy:
            if self.minimize_spectral_entropy_delay:
                logger.info(f"minimize spectral entropy (phi + epsilon)")
                h_time_freq = time2freq(self.h_time_delay, axis=1)
                minimize_spectral_entropy_with_delay(h_time_freq)
                self.h_time_delay = freq2time(h_time_freq, axis=1)
            else:
                logger.info(f"minimize spectral entropy")
                minimize_spectral_entropy(self.h_time_delay)
            self.h_doppler_delay = time2freq(self.h_time_delay, axis=0)

        if self.enforce_orthogonal_real_imag:
            z = (self.h_doppler_delay * self.h_doppler_delay).sum()
            ph = z / np.abs(z)
            ph = np.sqrt(ph)
            abs_origin = np.abs(self.h_doppler_delay[0, 0])
            logger.info(f"enforce_orthogonal_real_imag z={z} ph={ph} abs_origin={abs_origin}")
            self.h_doppler_delay *= np.conj(ph)
            self.h_time_delay = freq2time(self.h_doppler_delay)

        if self.enforce_real_at_origin:
            self.h_doppler_delay[0, 0] = np.real(self.h_doppler_delay[0, 0])
            self.h_time_delay = freq2time(self.h_doppler_delay)

        # CuPy port (see /home/willem/.claude/plans/stateful-dreaming-wall.md):
        # same single-flag dispatch as updateWavefield/compute_gradient_batched.
        if self.use_gpu:
            self.updateProfile_batched()
        elif self.nthread == 1:
            for isub in range(self.nspec):
                self.updateProfileSubint(isub)

        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.nthread) as executor:
                future_subint = {
                    executor.submit(self.updateProfileSubint, isub): isub for isub in range(self.nspec)
                }

                for future in concurrent.futures.as_completed(future_subint):
                    isub = future_subint[future]
                    try:
                        future.result()
                    except Exception as exc:
                        logger.info(f"updateProfile isub={isub} exception: {exc}")

        self.pp_intrinsic = np.average(self.intrinsic_profiles, axis=(0, 1))

        if self.compute_scattered_profile:
            self.pp_scattered = np.average(self.scattered_profiles, axis=0)

        mean_gain = 1

        if self.model_gain_variations:
            # keep the gains from wandering
            mean_gain = self.optimal_gains.mean()
            logger.info(f"updateProfile mean gain: {mean_gain}")
            self.optimal_gains /= mean_gain

        self.intrinsic_ph_sum = np.zeros(self.nharm, dtype=np.complex128)
        self.intrinsic_ph_sumsq = np.zeros(self.nharm, dtype=np.complex128)

        for ipol in range(self.npol):
            ph_numer = np.average(self.ph_numer[ipol], axis=0)
            ph_denom = np.average(self.ph_denom[ipol], axis=0)

            self.intrinsic_ph[ipol] = safely_divide(ph_numer, ph_denom)
            self.intrinsic_ph[ipol] *= mean_gain
            self.intrinsic_ph_sum += self.intrinsic_ph[ipol]
            self.intrinsic_ph_sumsq += np.abs(self.intrinsic_ph[ipol]) ** 2

        # sum over all polarizations and sub-integrations
        ph_numer = np.average(self.ph_numer, axis=(0, 1))
        ph_denom = np.average(self.ph_denom, axis=(0, 1))

        self.pp_intrinsic = self.harm2phase(safely_divide(ph_numer, ph_denom)) * mean_gain
        logger.info(f"updateProfile intrinsic profile range: {np.ptp(self.pp_intrinsic)}")

    def _refresh_jitter_model(self):
        """
        Recompute the reduced-rank jitter model (pycyc.jitter) against the
        freshly-pooled reference profile: fits epsilon_t/gain_t per subint
        against it (pycyc.profile.fit_profile_shift, via
        pycyc.jitter.compute_residuals), refreshes the PCA basis from the
        residuals (pycyc.jitter.fit_jitter_basis), and rebuilds
        self.jitter_profiles for the next pass's wavefield fit to use.

        Called by outer_loop once self.model_jitter is on and warmup has
        elapsed; assumes self.updateProfile() has just run, so
        self.pp_intrinsic/self.ph_numer/self.ph_denom are current.

        The per-harmonic profile noise floor needed by fit_jitter_basis is
        Var(S_t(alpha)) = sigma_D^2 / ph_denom(alpha) -- see
        pycyc.jitter.fit_jitter_basis's docstring for the derivation --
        using a single representative sigma_D^2 from self.cyclic_variance
        on one subint's cyclic spectrum (not a per-subint noise estimate;
        a simplifying approximation, consistent with fit_profile_shift's
        gain-fit approximation elsewhere in this plan).
        """
        ph_ref = phase2harm(self.pp_intrinsic)
        ph_ref = normalize_profile(ph_ref)
        if self.exclude_DC:
            ph_ref[0] = 0

        ph_numer_per_subint = np.average(self.ph_numer, axis=0)
        ph_denom_per_subint = np.average(self.ph_denom, axis=0)
        s_t_all = safely_divide(ph_numer_per_subint, ph_denom_per_subint)

        epsilon_t, gain_t, residuals = compute_residuals(ph_ref, s_t_all, fit_gain=self.model_gain_variations)

        if self.save_cyclic_spectra:
            cs0 = self.cyclic_spectra[0, 0]
        else:
            cs0, _norm = self.get_cs(self.data[0, 0])
        sigma_D2, _nvalid = self.cyclic_variance(cs0)

        ph_denom_avg = np.real(np.mean(ph_denom_per_subint, axis=0))
        noise_variance_per_harmonic = sigma_D2 / np.maximum(ph_denom_avg, 1e-300)

        rank, weights, basis, eigenvalues, threshold = fit_jitter_basis(
            residuals, noise_variance_per_harmonic, max_rank=self.jitter_max_rank
        )

        angle = (
            subspace_principal_angle(self._jitter_basis, basis) if self._jitter_basis is not None else None
        )
        logger.info(
            "outer_loop jitter refresh: rank=%d threshold=%.4g max_eigenvalue=%.4g principal_angle=%s",
            rank,
            threshold,
            eigenvalues.max() if eigenvalues.size else float("nan"),
            angle,
        )

        self._jitter_basis = basis
        self.jitter_rank = rank
        self.jitter_principal_angle = angle
        self.jitter_profiles = reconstruct_jittered_profile(ph_ref, epsilon_t, gain_t, weights, basis)

    def outer_loop(
        self,
        n_passes,
        warmup_passes=None,
        loop_kwargs=None,
        merit_tol=1e-4,
        jitter_angle_tol=1e-3,
        patience=2,
        use_last_soln=False,
    ):
        """
        Alternates a per-subint wavefield fit (self.loop, called once per
        subint) with profile re-estimation (self.updateProfile) across up
        to n_passes outer iterations -- the "outer loop" this module's own
        original docstring flagged as missing ("the next step will be to
        build the outer loop which uses the new guess at the intrinsic
        profile to reoptimize the IRF... this isn't yet implemented but
        the machinery is all there").

        For the first warmup_passes passes (default
        self.jitter_warmup_passes), this is exactly today's existing
        behavior: S(alpha) pooled/shared across every subint via the
        ordinary self.use_integrated_profile selection. After warmup, if
        self.model_jitter is True, each pass additionally refreshes a
        reduced-rank jitter model of the profile residuals
        (self._refresh_jitter_model) and feeds a per-subint jitter-aware
        profile into the next pass's wavefield fit (loop() and
        updateWavefieldSubint() both consult self.jitter_profiles ahead of
        the ordinary profile selection).

        loop_kwargs: extra keyword arguments passed to every self.loop()
        call (e.g. maxfun, tolfact) -- see loop()'s own docstring. Do not
        include isub, hf_prev, ht0, use_last_soln or freeze_profile here --
        outer_loop manages those itself (see below).

        Per-subint fits within a pass run in parallel via
        concurrent.futures.ThreadPoolExecutor when self.nthread > 1 (mirroring
        updateProfile/compute_gradient), else serially -- same as those two
        methods. This requires each subint's fit to be independent of every
        other subint's fit *within the same pass*, which rules out loop()'s
        own default of warm-starting isub from whatever self.hf_prev/
        self.rindex happened to be left by the immediately-preceding
        self.loop() call (well-defined only under strictly serial
        execution -- see loop()'s use_last_soln guard). So by default
        (use_last_soln=False) outer_loop instead warm-starts each subint
        from *its own* previous pass's solution (self.h_time_delay[isub] /
        self.optimized_filters[isub], already stored) rather than from
        whichever subint happened to run immediately before it in scan
        order -- independent per isub, safe for any self.nthread, and
        arguably more principled for this multi-pass setting anyway (each
        subint's fit smoothly refines pass over pass instead of chasing a
        neighbor's possibly-unrelated solution within one pass). The
        reference profile (self.ph_ref) is likewise frozen for the whole
        pass -- computed once here from self.pp_intrinsic before dispatching
        any subint fits, rather than the online per-subint refinement
        loop() does when called directly (freeze_profile=True suppresses
        that; see loop()'s docstring).

        Passing use_last_soln=True restores today's original behavior
        exactly (loop()'s own self.hf_prev/self.rindex chain, each subint
        warm-started from the immediately-preceding one in scan order, and
        self.pp_intrinsic refined online across the pass) -- but loop()
        itself will raise if self.nthread != 1 in that case, since that
        chain is meaningless under concurrent execution; outer_loop always
        runs its per-subint loop serially when use_last_soln=True; set
        self.nthread = 1 to use it.

        Stopping condition: a pass counts as "stable" when the total
        merit's relative change from the previous pass is below merit_tol,
        AND -- only once the jitter model is active (self.model_jitter and
        past warmup) -- the jitter basis has also stopped moving, i.e.
        self.jitter_principal_angle (set by _refresh_jitter_model, the
        principal angle in radians between this pass's and the previous
        pass's jitter subspace) is below jitter_angle_tol. Requiring both
        avoids declaring convergence just because the wavefield merit has
        locally plateaued while the jitter subspace is still shifting
        underneath it. The loop stops once `patience` consecutive passes
        are stable; n_passes is always a hard cap regardless. Set
        patience<=0 to disable early stopping entirely and always run
        exactly n_passes passes.
        """
        if warmup_passes is None:
            warmup_passes = self.jitter_warmup_passes
        loop_kwargs = dict(loop_kwargs or {})

        prev_merit = None
        stable_count = 0

        for pass_idx in range(n_passes):
            ph_ref = phase2harm(self.pp_intrinsic)
            ph_ref = normalize_profile(ph_ref)
            if self.exclude_DC:
                ph_ref[0] = 0
            self.ph_ref = ph_ref

            def _fit_subint(isub, pass_idx=pass_idx):
                if use_last_soln:
                    ht0 = None
                    hf_prev = None
                elif pass_idx == 0:
                    ht0 = None
                    hf_prev = np.ones((self.nchan,), dtype=np.complex128)
                else:
                    ht0 = self.h_time_delay[isub].copy()
                    hf_prev = self.optimized_filters[isub].copy()
                return self.loop(
                    isub=isub,
                    hf_prev=hf_prev,
                    ht0=ht0,
                    use_last_soln=use_last_soln,
                    freeze_profile=True,
                    **loop_kwargs,
                )

            total_merit = 0.0
            if self.nthread == 1:
                for isub in range(self.nspec):
                    total_merit += _fit_subint(isub)
            else:
                # unlike compute_gradient/updateProfile, exceptions here are
                # not caught-and-printed: a failed subint fit (in particular
                # the use_last_soln/nthread guard above) must stop outer_loop
                # outright rather than silently proceeding a pass short of a
                # subint's contribution.
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.nthread) as executor:
                    future_subint = {executor.submit(_fit_subint, isub): isub for isub in range(self.nspec)}

                    for future in concurrent.futures.as_completed(future_subint):
                        total_merit += future.result()

            self.updateProfile()

            jitter_active = self.model_jitter and pass_idx >= warmup_passes
            if jitter_active:
                self._refresh_jitter_model()

            merit_converged = (
                prev_merit is not None
                and prev_merit != 0
                and abs(total_merit - prev_merit) / abs(prev_merit) < merit_tol
            )
            jitter_converged = (
                not jitter_active
                or self.jitter_principal_angle is None
                or self.jitter_principal_angle < jitter_angle_tol
            )
            stable_count = stable_count + 1 if (merit_converged and jitter_converged) else 0
            prev_merit = total_merit

            logger.info(
                "outer_loop pass %d/%d: total merit=%.6e jitter_rank=%d stable=%d/%d",
                pass_idx + 1,
                n_passes,
                total_merit,
                self.jitter_rank,
                stable_count,
                patience,
            )

            if patience > 0 and stable_count >= patience:
                logger.info("outer_loop: converged after %d pass(es)", pass_idx + 1)
                break

    def initWavefield(self):
        """
        First draft of using FISTA to solve the 2D transfer function
        """

        self.h_time_delay_grad = np.zeros((self.nspec, self.nchan), dtype=np.complex128)
        self.h_doppler_delay_grad = np.zeros((self.nspec, self.nchan), dtype=np.complex128)
        self.nopt = 0

        self.updateWavefield(self.h_doppler_delay)

    def get_derivative(self, wavefield):
        return self.h_doppler_delay_grad

    def get_func_val(self, wavefield):
        return self.merit

    def evaluate(self, wavefield):
        self.updateWavefield(wavefield)
        return self.merit, np.copy(self.h_doppler_delay_grad)

    def get_cs(self, ps):
        # cast single-precision input data to double-precision before computing the Fourier transform
        tmp = np.zeros(ps.shape, dtype=np.float64)
        tmp[:, :] = ps[:, :]

        cs = ps2cs(tmp)

        if not self.include_Nyquist:
            cs = cs[:, : self.nharm]

        # print (f"get_cs power in cs={np.sum(np.abs(cs)**2)}")

        if self.model_gain_variations:
            cs, norm = normalize_cs_by_noise_rms(cs, bw=self.bw, ref_freq=self.ref_freq)
        elif self.normalize_cyclic_spectra:
            cs, norm = normalize_cs(cs, bw=self.bw, ref_freq=self.ref_freq)
        else:
            norm = 1
        if self.pad_cyclic_spectra:
            cs = cyclic_padding(cs, self.bw, self.ref_freq)
        if self.maxharm is not None:
            cs[:, self.maxharm + 1 :] = 0.0
        if self.exclude_DC:
            cs[:, 0] = 0.0

        return cs, norm

    def normalize(self, h_doppler_delay):
        if self.conserve_wavefield_energy:
            total_power = np.sum(np.abs(h_doppler_delay) ** 2)
            factor = np.sqrt(self.expected_power / total_power)
            h_doppler_delay *= factor
            # print(f'normalize factor={factor}')
        return h_doppler_delay

    def updateWavefieldSubint(self, ipol, isub):
        if self.save_cyclic_spectra:
            cs = self.cyclic_spectra[isub, ipol]
        else:
            ps = self.data[isub, ipol]  # dimensions will now be (nchan,nbin)
            cs, norm = self.get_cs(ps)

        if self.jitter_profiles is not None:
            ph = self.jitter_profiles[isub]
        elif self.use_integrated_profile:
            ph = self.ph_ref
        else:
            ph_numer = self.ph_numer[ipol, isub]
            ph_denom = self.ph_denom[ipol, isub]
            ph = ph_numer / ph_denom

        ht = self.h_time_delay[isub]

        if self.iprint:
            logger.info(f"update filter isub={isub}/{self.nspec}")

        self.rindex = isub
        _merit, grad, _nterm = cyclic_merit_and_grad(
            ht,
            self._model_params(),
            ph,
            cs,
            gain=self.optimal_gains[isub],
            rindex=self.rindex,
            dump_residual=self.dump_residual,
            iprint=self.iprint,
        )

        self.h_time_delay_grad[isub, :] += grad

        return _merit, _nterm

    def _zap_gradient_harmonics(self):
        """
        Zero out self.zap_gradient_harmonics repeating harmonics of the
        strongest artifact(s) in self.h_doppler_delay_grad, in place.
        """
        grad_power = np.abs(self.h_doppler_delay_grad) ** 2
        largest = find_n_largest_indices(grad_power, self.zap_gradient_harmonics)
        max_power = grad_power[largest[0]]
        for idx in range(self.zap_gradient_harmonics):
            ilargest = largest[idx]
            logger.debug("%s power at %s = %s", f"idx={idx}", ilargest, grad_power[ilargest])
            if grad_power[ilargest] < 1e-6 * max_power:
                logger.debug("stopping zap_gradient_harmonics search")
                break

            for jdx in range(idx + 1, self.zap_gradient_harmonics):
                jlargest = largest[jdx]
                logger.debug("%s power at %s = %s", f"jdx={jdx}", jlargest, grad_power[jlargest])
                if grad_power[jlargest] > 1e-6 * max_power:
                    logger.debug("zeroing harmonics of %s and %s in gradient", ilargest, jlargest)
                    x_offset = abs(jlargest[0] - ilargest[0])
                    y_offset = abs(jlargest[1] - ilargest[1])
                    x = max(jlargest[0], ilargest[0])
                    y = max(jlargest[1], ilargest[1])
                    while (x < grad_power.shape[0]) and (y < grad_power.shape[1]):
                        self.h_doppler_delay_grad[x, y] = 0
                        x += x_offset
                        y += y_offset

    def updateWavefield(self, h_doppler_delay):
        rms_noise = rms_wavefield(h_doppler_delay)

        if rms_noise > 0 and self.noise_threshold is not None:
            # print(f"noise_threshold rms={rms_noise}")
            np.copyto(
                h_doppler_delay,
                apply_threshold(h_doppler_delay, self.noise_threshold, self.noise_smoothing_kernel),
            )

        if rms_noise > 0 and self.noise_shrinkage_threshold is not None:
            # print(f"noise_shrinkage_threshold rms={rms_noise}")
            np.copyto(
                h_doppler_delay,
                apply_shrinkage_threshold(
                    h_doppler_delay,
                    self.noise_shrinkage_threshold,
                    self.noise_smoothing_kernel,
                ),
            )

        # Not gated on rms_noise > 0 like the two blocks above: rms_noise
        # comes from rms_wavefield's fixed reference region (nchan*5//8 :
        # nchan*7//8), which sits entirely inside the acausal (negative-
        # delay) half. When self.enforce_causality holds permanently
        # (rather than expiring after a few warmup iterations), that whole
        # half -- including this reference region -- is exactly zero every
        # iteration, so rms_noise is always 0 and this block would
        # silently never fire if gated the same way. apply_delay_shrinkage_
        # threshold doesn't need rms_noise anyway: it computes its own
        # per-delay noise estimate (delay_noise_power_wavefield, from the
        # Doppler-edge strip) independently of the reference region above.
        if self.delay_noise_shrinkage_threshold is not None:
            # print(f"delay_noise_shrinkage_threshold={self.delay_noise_shrinkage_threshold}")
            np.copyto(
                h_doppler_delay,
                apply_delay_shrinkage_threshold(
                    h_doppler_delay,
                    self.delay_noise_shrinkage_threshold,
                    self.delay_noise_selection_threshold,
                    self.noise_smoothing_kernel,
                ),
            )

        if self.delay_taper is not None:
            h_doppler_delay *= self.delay_taper

        if self.doppler_taper is not None:
            h_doppler_delay *= self.doppler_taper[:, np.newaxis]

        self.h_time_delay = freq2time(h_doppler_delay, axis=0)
        np.copyto(self.h_doppler_delay, h_doppler_delay)

        nonzero = np.count_nonzero(h_doppler_delay)
        # although re & im count as separate terms in sum,
        # normalize_cs_by_noise_rms normalizes by the sum of the variances in re & im
        self.nfree_parameters = nonzero

        self.merit = 0
        self.nterm_merit = 0

        phasor = 1.0 + 0.0j

        if self.shear_phasors is None:
            self.shear_phasors = create_shear_phasors(self.nchan, self.nharm, self.bw, self.ref_freq)

        # CuPy port (see /home/willem/.claude/plans/stateful-dreaming-wall.md):
        # self.use_gpu selects both the batched-vs-threaded gradient path
        # and, within the batched path, numpy vs cupy -- there's no use
        # case for "batched but still on CPU" outside compute_gradient_batched's
        # own tests, so a single flag controls both here.
        if self.use_gpu:
            self.compute_gradient_batched()
        else:
            self.compute_gradient()

        if self.subtract_degenerate_projections:
            self.h_time_delay_grad = subtract_degenerate_dof(self.h_time_delay_grad, self.h_time_delay)

        if self.enforce_causality:
            half_nchan = self.nchan // 2
            self.h_time_delay_grad[:, half_nchan:] = 0

        if self.reduce_temporal_phase_noise_grad:
            minimize_temporal_phase_noise(self.h_time_delay_grad)

        if self.dump_gradient:
            logger.debug("dumping time delay gradient to h_time_delay_grad.pkl")
            with open("h_time_delay_grad.pkl", "wb") as fh:
                pickle.dump(self.h_time_delay_grad, fh)

        self.h_doppler_delay_grad = time2freq(self.h_time_delay_grad)

        if self.zap_gradient_harmonics > 0:
            self._zap_gradient_harmonics()

        if self.dump_gradient:
            logger.debug("dumping doppler delay gradient to h_doppler_delay_grad.pkl")
            with open("h_doppler_delay_grad.pkl", "wb") as fh:
                pickle.dump(self.h_doppler_delay_grad, fh)

        if self.low_pass_filter_Doppler < 1:
            quarter_nsub = round(self.nsubint * self.low_pass_filter_Doppler / 2.0)
            self.h_doppler_delay_grad[quarter_nsub:-quarter_nsub, :] = 0

        align_phase_gradient = False
        if align_phase_gradient:
            logger.info(f"h_doppler_delay_grad[0,0]={self.h_doppler_delay_grad[0,0]}")
            phasor = np.conj(self.h_doppler_delay_grad[0, 0])
            phasor /= np.abs(phasor)
            self.h_doppler_delay_grad *= phasor

    def compute_gradient(self):
        self.h_time_delay_grad[:, :] = 0.0 + 0.0j

        for ipol in range(self.npol):
            self.ph_ref = self.intrinsic_ph[ipol]

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.nthread) as executor:
                future_subint = {
                    executor.submit(self.updateWavefieldSubint, ipol, isub): isub
                    for isub in range(self.nspec)
                }

                for future in concurrent.futures.as_completed(future_subint):
                    isub = future_subint[future]
                    try:
                        _merit, _nterm = future.result()
                        self.merit += _merit
                        self.nterm_merit += _nterm

                    except Exception:
                        logger.exception("compute_gradient isub=%d raised", isub)

    def _to_device_dtype(self, arr, xp, dtype):
        """to_device, additionally downcasting to `dtype` first if given
        (so the smaller array is what actually crosses the host->device
        transfer, not transferred at full precision and downcast after)."""
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return to_device(arr, xp)

    def _device_cyclic_spectra(self, xp, dtype):
        """Device-resident, dtype-cast cache of self.cyclic_spectra, keyed
        on the host array's id() (so a reload/reassignment invalidates it
        automatically) plus dtype. Avoids re-transferring the whole
        (nsubint, npol, nchan, nharm) array across every FISTA iteration --
        see this method's callers' docstrings for the in-place-mutation
        assumption this relies on."""
        if xp is np:
            return self.cyclic_spectra
        cache_key = (id(self.cyclic_spectra), dtype)
        if self._gpu_cyclic_spectra_key != cache_key:
            self._gpu_cyclic_spectra = self._to_device_dtype(self.cyclic_spectra, xp, dtype)
            self._gpu_cyclic_spectra_key = cache_key
        return self._gpu_cyclic_spectra

    def _device_phasors(self, xp, dtype):
        """Device-resident, dtype-cast cache of self.shear_phasors -- same
        rationale as _device_cyclic_spectra, much smaller array but reused
        identically every single subint fit within every iteration."""
        if xp is np:
            return self.shear_phasors
        cache_key = (id(self.shear_phasors), dtype)
        if self._gpu_phasors_key != cache_key:
            self._gpu_phasors = self._to_device_dtype(self.shear_phasors, xp, dtype)
            self._gpu_phasors_key = cache_key
        return self._gpu_phasors

    def compute_gradient_batched(self):
        """
        Batched sibling of compute_gradient: evaluates the merit and
        gradient for a whole chunk of subints in one call to
        pycyc.objective.cyclic_merit_and_grad_batched, instead of
        ThreadPoolExecutor-dispatching updateWavefieldSubint one subint at
        a time. CuPy port Stages 2-3 (see
        /home/willem/.claude/plans/stateful-dreaming-wall.md) -- reads
        self.use_gpu (off by default) and self.gpu_chunk_size (None = all
        self.nspec subints in one call, fine on CPU; a real chunk size is
        needed once self.use_gpu=True, to fit the GPU's memory budget) and
        self.gpu_dtype (downcast precision on-device, default complex64;
        None keeps whatever dtype the source arrays already are).

        self.cyclic_spectra and self.shear_phasors -- read identically by
        every chunk/iteration -- are cached device-resident (see
        _device_cyclic_spectra/_device_phasors) rather than re-transferred
        every call; this assumes neither is mutated *in place* once the
        FISTA loop starts calling this method (today's in-place tapering
        happens earlier, during initProfile/load).

        Requires self.save_cyclic_spectra=True: the self.data/self.get_cs
        per-subint fallback compute_gradient supports isn't batched here,
        and nothing on the FISTA path (cycfista.py/cycsolve.py) runs
        without self.save_cyclic_spectra=True anyway.
        """
        if not self.save_cyclic_spectra:
            raise NotImplementedError(
                "compute_gradient_batched requires self.save_cyclic_spectra=True "
                "(the self.data/self.get_cs per-subint fallback isn't batched here)."
            )

        xp = get_xp(self.use_gpu)
        fft_module = get_fft(xp)
        chunk_size = self.gpu_chunk_size or self.nspec
        dtype = self.gpu_dtype if self.use_gpu else None

        params = self._model_params()
        params_xp = dataclasses.replace(params, shear_phasors=self._device_phasors(xp, dtype))

        cyclic_spectra_dev = self._device_cyclic_spectra(xp, dtype)

        self.h_time_delay_grad[:, :] = 0.0 + 0.0j

        for ipol in range(self.npol):
            self.ph_ref = self.intrinsic_ph[ipol]

            if self.jitter_profiles is not None:
                s0_full = self.jitter_profiles  # (nsubint, nharm)
            elif self.use_integrated_profile:
                s0_full = self.ph_ref  # (nharm,) -- shared across the batch
            else:
                s0_full = self.ph_numer[ipol] / self.ph_denom[ipol]  # (nsubint, nharm)

            cs_full = cyclic_spectra_dev[:, ipol]  # (nsubint, nchan, nharm), already device-resident

            for start in range(0, self.nspec, chunk_size):
                stop = min(start + chunk_size, self.nspec)
                sl = slice(start, stop)

                ht_chunk = self._to_device_dtype(self.h_time_delay[sl], xp, dtype)
                cs_chunk = cs_full[sl]
                gain_chunk = to_device(self.optimal_gains[sl], xp)  # tiny, keep full precision
                s0_source = s0_full if s0_full.ndim == 1 else s0_full[sl]
                s0_chunk = self._to_device_dtype(s0_source, xp, dtype)

                merit_chunk, grad_chunk, nonzero_chunk = cyclic_merit_and_grad_batched(
                    ht_chunk,
                    params_xp,
                    s0_chunk,
                    cs_chunk,
                    gain_batch=gain_chunk,
                    xp=xp,
                    fft_module=fft_module,
                )

                self.h_time_delay_grad[sl, :] += to_host(grad_chunk)
                self.merit += float(to_host(merit_chunk).sum())
                self.nterm_merit += int(to_host(nonzero_chunk).sum())

    def optimize_profile(self, cs, hf, bw, ref_freq, update_gain):
        # bw/ref_freq are taken from the explicit arguments (not self.bw/
        # self.ref_freq) to preserve this method's existing signature/
        # behavior exactly, including for a caller that passes values
        # different from self's current ones.
        params = CyclicModelParams(
            bw=bw,
            ref_freq=ref_freq,
            shear_phasors=self.shear_phasors,
            pad_cyclic_spectra=self.pad_cyclic_spectra,
            include_Nyquist=self.include_Nyquist,
            maxharm=self.maxharm,
            exclude_DC=self.exclude_DC,
            nlag=self.nlag,
        )
        return solve_profile_and_gain(
            cs, hf, params, update_gain, self.intrinsic_ph_sum, self.intrinsic_ph_sumsq
        )

    def _setup_plot_directory(self, make_plots, plotdir, max_plot_lag):
        """Set self.make_plots/self.mlag/self.plotdir for loop(), creating
        plotdir (defaulting to "<filename>_plots") if it doesn't exist yet."""
        self.make_plots = make_plots
        if not make_plots:
            return
        self.mlag = max_plot_lag
        if plotdir is None:
            blah, fbase = os.path.split(self.filename)
            plotdir = os.path.join(os.path.abspath(os.path.curdir), ("%s_plots" % fbase))
        if not os.path.exists(plotdir):
            try:
                os.mkdir(plotdir)
            except Exception:
                logger.info("Warning: couldn't make %s, not plotting", plotdir)
                self.make_plots = False
        self.plotdir = plotdir

    def loop(
        self,
        isub=0,
        ipol=0,
        hf_prev=None,
        make_plots=False,
        maxfun=1000,
        tolfact=1,
        iprint=1,
        plotdir=None,
        maxneg=None,
        maxlen=None,
        rindex=None,
        ht0=None,
        max_plot_lag=50,
        use_last_soln=True,
        use_minphase=True,
        onp=None,
        adjust_delay=True,
        plot_every=1,
        freeze_profile=False,
    ):
        """
        Run the non-linear solver to compute the IRF

        maxfun: int
            maximum number of objective function evaluations
        tolfact: float
            factor to multiply the convergence limit by. Default 1
            uses convergence criteria from original filter_profile.
            Try 10 for less stringent (faster) convergence
        iprint: int
            Passed to scipy.optimize.fmin_l_bfgs_b (see docs)
            use 0 for silent, 1 for verbose, 2 for more log info

        max_plot_lag: highest lag to plot in diagnostic plots.
        use_last_soln: If true, warm-start this subint's fit from
            self.hf_prev (or the explicit hf_prev argument) -- i.e. from
            whichever self.loop() call happened to run immediately before
            this one -- and, once fit, fold this subint's own profile
            contribution into self.pp_intrinsic (self.ph_ref for the *next*
            call is then this pooled, still-accumulating profile). This
            chain is only well-defined under strictly serial execution:
            raises RuntimeError if True while self.nthread != 1, since
            self.nthread is the only signal loop() has that calls might be
            dispatched concurrently (e.g. multiple isub all starting before
            any of them has incremented self.nopt, so checking self.nopt
            can't reliably tell them apart from a genuine one-off call).
            outer_loop defaults to False and instead warm-starts each
            subint from its own previous-pass solution, safe for any
            self.nthread -- see outer_loop's docstring.
        use_minphase: if true, use minimum phase IRF as initial guess
                        else use delta function
        freeze_profile: if True, use self.ph_ref exactly as the caller left
            it (skip recomputing it from self.pp_intrinsic) and don't fold
            this subint's profile into self.pp_intrinsic afterwards --
            for callers (outer_loop) that freeze the reference profile for
            an entire pass and later recompute self.pp_intrinsic themselves
            (self.updateProfile) rather than relying on loop()'s own
            within-pass online accumulation.

        Returns the fitted filter's final objective value (scipy's
        fmin_l_bfgs_b return `f`).
        """
        if use_last_soln and self.nthread != 1:
            raise RuntimeError(
                "loop(use_last_soln=True) warm-starts from self.hf_prev, "
                "i.e. from whichever self.loop() call happened to run "
                "immediately before this one -- well-defined only under "
                "strictly serial execution. Set self.nthread = 1 to use "
                "use_last_soln=True, or leave it False (the default) to "
                f"fit subints independently, safe for any self.nthread "
                f"(currently {self.nthread})."
            )

        self.plot_every = plot_every
        self._setup_plot_directory(make_plots, plotdir, max_plot_lag)

        self.isub = isub
        self.iprint = iprint

        if self.save_cyclic_spectra:
            cs = self.cyclic_spectra[isub, ipol]
        else:
            ps = self.data[isub, ipol]  # dimensions will now be (nchan,nbin)
            cs, norm = self.get_cs(ps)

        if hf_prev is None:
            if self.hf_prev is None:
                self.hf_prev = np.ones((self.nchan,), dtype=np.complex128)
            _hf_prev = self.hf_prev
        else:
            _hf_prev = hf_prev

        self.dynamic_spectrum[isub, :] = np.real(cs[:, 0])

        if not freeze_profile:
            self.ph_ref = phase2harm(self.pp_intrinsic)
            self.ph_ref = normalize_profile(self.ph_ref)

            if self.exclude_DC:
                self.ph_ref[0] = 0

        if self.jitter_profiles is not None:
            ph = self.jitter_profiles[isub]
        else:
            ph = self.ph_ref[:]

        if self.nopt == 0 or not use_last_soln:
            if not freeze_profile:
                self.pp_intrinsic = np.zeros(self.nphase)
            if ht0 is None:
                if rindex is None:
                    delay = self.phase_gradient(cs)
                else:
                    delay = rindex
                logger.info("initial filter: delta function at delay = %d", delay)
                ht = np.zeros((self.nlag,), dtype=np.complex128)
                ht[delay] = self.nlag
                if use_minphase:
                    if onp is None:
                        logger.info("onp not specified, so not using minimum phase")
                    else:
                        spect = np.abs(self.data[isub, ipol, :, onp[0] : onp[1]]).mean(1)
                        ht = freq2time(minphase(spect - spect.min()))
                        ht = np.roll(ht, delay)
                        logger.info("using minimum phase with peak at: %s", np.abs(ht).argmax())
            else:
                ht = ht0.copy()
            hf = time2freq(ht)
        else:
            hf = _hf_prev.copy()
        ht = freq2time(hf)

        if self.delay_taper is not None:
            ht *= self.delay_taper

        if self.nopt == 0 or adjust_delay:
            if rindex is None:
                rindex = np.abs(ht).argmax()
            self.rindex = rindex
        else:
            rindex = self.rindex
        logger.info("max filter index = %d", self.rindex)

        if maxneg is not None:
            if maxlen is not None:
                valsamp = maxlen
            else:
                valsamp = int(ht.shape[0] / 2) + maxneg
            minbound = np.zeros_like(ht)
            minbound[:valsamp] = 1 + 1j
            minbound = np.roll(minbound, rindex - maxneg)
            b = pack_real_params(minbound, rindex)
            bchoice = [0, None]
            bounds = [(bchoice[int(x)], bchoice[int(x)]) for x in b]
        else:
            bounds = None
        # rotate phase time
        phasor = np.conj(ht[rindex])
        ht = ht * phasor / np.abs(phasor)

        dim0 = 2 * self.nlag - 1

        var, nvalid = self.cyclic_variance(cs)
        self.noise = np.sqrt(var)
        dof = nvalid - dim0 - self.nphase
        logger.info("variance : %.5e", var)
        logger.info("nsamp    : %.5e", nvalid)
        logger.info("dof      : %.5e", dof)
        logger.info("min obj  : %.5e", dof * var)

        tol = 1e-1 / (dof)
        logger.info("ftol     : %.5e", tol)
        scipytol = (
            tolfact * tol / 2.220e-16
        )  # 2.220E-16 is machine epsilon, which the scipy optimizer uses as a unit
        logger.info("scipytol : %.5e", scipytol)
        x0 = pack_real_params(ht, rindex)

        self.niter = 0
        self.objval = []

        self.cs = cs
        params = self._model_params()

        def _objective(x):
            merit, grad = cyclic_merit_lag_x(
                x, params, rindex, ph, cs, dump_residual=self.dump_residual, iprint=self.iprint
            )
            # the objval list keeps track of how the convergence is going
            self.objval.append(merit)
            return merit, grad

        x, f, d = scipy.optimize.fmin_l_bfgs_b(
            _objective,
            x0,
            m=20,
            iprint=iprint,
            maxfun=maxfun,
            factr=scipytol,
            bounds=bounds,
        )
        ht = unpack_real_params(x, rindex)

        if self.delay_taper is not None:
            ht *= self.delay_taper

        hf = time2freq(ht)

        self.hf_soln = hf[:]

        hf = match_two_filters(_hf_prev, hf)
        self.optimized_filters[isub, :] = hf

        ht = freq2time(hf)
        self.h_time_delay[isub, :] = ht

        self.hf_prev = hf.copy()

        update_gain = False
        ph, gain, ph_numer, ph_denom = self.optimize_profile(cs, hf, self.bw, self.ref_freq, update_gain)

        if self.exclude_DC:
            ph[0] = 0.0

        pp = self.harm2phase(ph)

        self.intrinsic_profiles[isub, :] = pp
        if not freeze_profile:
            self.pp_intrinsic += pp

        self.nopt += 1

        return f

    def cyclic_variance(self, cs):
        ih = self.nharm

        imin, imax = chan_limits_cs(
            iharm=ih, nchan=self.nchan, bw=self.bw, ref_freq=self.ref_freq
        )  # highest harmonic

        var = (np.abs(cs[imin:imax, ih - 1]) ** 2).sum()
        nvalid = imax - imin
        var = var / nvalid

        for ih in range(1, self.nharm):
            imin, imax = chan_limits_cs(iharm=ih, nchan=self.nchan, bw=self.bw, ref_freq=self.ref_freq)
            nvalid += imax - imin
        return var, nvalid * 2

    def phase_gradient(self, cs, ph_ref=None):
        if ph_ref is None:
            ph_ref = self.ph_ref
        ih = 1
        imin, imax = chan_limits_cs(iharm=ih, nchan=self.nchan, bw=self.bw, ref_freq=self.ref_freq)
        grad_sum = cs[:, ih].sum()
        grad_sum /= ph_ref[ih]
        phase_angle = np.angle(grad_sum)
        # ensure -pi < ph < pi
        if phase_angle > np.pi:
            phase_angle = phase_angle - 2 * np.pi
        # express as delay
        phase_angle /= -2 * np.pi * self.ref_freq
        phase_angle *= 1e6 * self.bw

        if phase_angle > self.nchan / 2:
            delay = int(self.nchan / 2)
        elif phase_angle < -(self.nchan / 2):
            delay = int(self.nchan / 2 + 1)
        elif phase_angle < -0.1:
            delay = int(phase_angle) + self.nchan - 1
        else:
            delay = int(phase_angle)

        return delay

    def harm2phase(self, ph, workers=1):
        if self.include_Nyquist:
            tmp = ph
        else:
            nharm = ph.shape[0]
            tmp = np.zeros(nharm + 1, dtype=np.complex128)
            tmp[:nharm] = ph[:]
        return irfft(tmp, workers=workers, norm="ortho")

    def plotCurrentSolution(self, plot_cs):
        # optimize_profile requires update_gain; plotCurrentSolution always
        # plots against the current best profile estimate, not a re-solve.
        sopt, gain, ph_numer, ph_denom = self.optimize_profile(
            plot_cs, self.hf, self.bw, self.ref_freq, False
        )
        sopt = normalize_profile(sopt)
        if self.exclude_DC:
            sopt[0] = 0.0
        smeas = normalize_profile(plot_cs.mean(0))
        if self.exclude_DC:
            smeas[0] = 0.0

        plot_current_solution(
            plot_cs,
            self.model,
            self.grad,
            self.hf,
            self.ht,
            self.mlag,
            self.rf,
            self.bw,
            self.rindex,
            self.noise,
            self.nchan,
            self.harm2phase(self.ph_ref),
            self.harm2phase(sopt),
            self.harm2phase(smeas),
            self.objval,
            self.filename,
            self.isub,
            self.nopt,
            self.source,
            self.niter,
            self.plotdir,
        )
