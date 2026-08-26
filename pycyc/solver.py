"""
pycyc.solver - CyclicSolver orchestration.

The merit/gradient/model math is in pycyc.objective/pycyc.profile (Stage 2
of the refactor plan) as pure functions of an explicit CyclicModelParams
rather than a CyclicSolver instance; PSRFITS/pickle I/O is in pycyc.io
(_IOMixin, Stage 3) and diagnostic plotting is in the top-level plotting.py
(also Stage 3). What's left here is orchestration: CyclicSolver's own
state and the methods that tie the above together.
"""

try:
    import psrchive
except Exception:
    print("pycyc.py: psrchive python libraries not found. You will not be able to load psrchive files.")
import concurrent.futures
import logging
import os
import pickle

import numpy as np
import scipy
import scipy.optimize
from scipy.fft import fftshift, irfft
from scipy.signal.windows import kaiser

from plotting import plot_Doppler_vs_delay, plot_current_solution

from .transforms import *
from .regularization import *
from .io_utils import *
from .model import *
from .objective import make_model_cs, pack_real_params, unpack_real_params, cyclic_merit_and_grad, cyclic_merit_lag_x
from .profile import solve_profile_and_gain
from .io import _IOMixin, loadCyclicSolver

logger = logging.getLogger(__name__)

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
                        print(f"unexpected initial response[{ichan}]={hf[ichan]}")

        else:
            self.expected_power = np.sum(np.abs(self.initial_h_time_freq) ** 2)
            current_shape = (self.nspec, self.nchan)
            if self.initial_h_time_freq.shape != current_shape:
                print(f"padding input shape={self.initial_h_time_freq.shape} to {current_shape=}")
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
                print(f"{peak=}")
                self.h_time_delay = np.roll(self.h_time_delay, -peak, axis=1)
                self.h_doppler_delay = time2freq(self.h_time_delay, axis=0)

            if self.enforce_real_at_origin:
                self.h_doppler_delay[0, 0] = np.real(self.h_doppler_delay[0, 0])

        if self.iprint:
            print(f"ORIGIN AMPLITUDE: {self.h_doppler_delay[0,0]}")

        self.noise_smoothing_kernel = None
        if self.noise_smoothing_duty_cycle is not None:
            ashape = np.asarray(self.h_doppler_delay.shape)
            wshape = np.round(ashape * self.noise_smoothing_duty_cycle)
            print(f"noise smoothing kernel shape: {wshape}")
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
            print(
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
            print(f"harmonic={harm}")
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
            print(f"harmonic={harm} S/N={sn[harm]}")

        best_harmonic = np.argmax(sn)
        print(f"best harmonic={best_harmonic}")

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
        print(f"amplitude[0,0] current={zero_amp} new={mean_amp}")

        if self.delay_noise_shrinkage_threshold is not None:
            print(f"delay_noise_shrinkage_threshold={self.delay_noise_shrinkage_threshold}")
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
        print(f"total power original={initial_total_power} new={total_power} scale={scale_factor}")
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
            print("Saving after nopt:", self.nopt)
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
                print(f"update profile isub={isub}/{self.nsubint}")

            self.intrinsic_profiles[isub, ipol, :] = pp
            return 0

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
            print(f"reduce temporal phase and delay noise")
            h_time_freq = time2freq(self.h_time_delay, axis=1)
            align_to_neighbour(h_time_freq)
            self.h_time_delay = freq2time(h_time_freq, axis=1)
            self.h_doppler_delay = time2freq(self.h_time_delay, axis=0)

        if self.reduce_temporal_phase_noise:
            print(f"reduce temporal phase noise")
            minimize_temporal_phase_noise(self.h_time_delay)
            self.h_doppler_delay = time2freq(self.h_time_delay, axis=0)

        if self.minimize_spectral_entropy:
            print(f"minimize spectral entropy")
            minimize_spectral_entropy(self.h_time_delay)
            self.h_doppler_delay = time2freq(self.h_time_delay, axis=0)

        if self.enforce_orthogonal_real_imag:
            z = (self.h_doppler_delay * self.h_doppler_delay).sum()
            ph = z / np.abs(z)
            ph = np.sqrt(ph)
            abs_origin = np.abs(self.h_doppler_delay[0, 0])
            print(f"enforce_orthogonal_real_imag z={z} ph={ph} abs_origin={abs_origin}")
            self.h_doppler_delay *= np.conj(ph)
            self.h_time_delay = freq2time(self.h_doppler_delay)

        if self.enforce_real_at_origin:
            self.h_doppler_delay[0, 0] = np.real(self.h_doppler_delay[0, 0])
            self.h_time_delay = freq2time(self.h_doppler_delay)

        if self.nthread == 1:
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
                        print(f"updateProfile isub={isub} exception: {exc}")

        self.pp_intrinsic = np.average(self.intrinsic_profiles, axis=(0, 1))

        if self.compute_scattered_profile:
            self.pp_scattered = np.average(self.scattered_profiles, axis=0)

        mean_gain = 1

        if self.model_gain_variations:
            # keep the gains from wandering
            mean_gain = self.optimal_gains.mean()
            print(f"updateProfile mean gain: {mean_gain}")
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
        print(f"updateProfile intrinsic profile range: {np.ptp(self.pp_intrinsic)}")

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

        if self.use_integrated_profile:
            ph = self.ph_ref
        else:
            ph_numer = self.ph_numer[ipol, isub]
            ph_denom = self.ph_denom[ipol, isub]
            ph = ph_numer / ph_denom

        ht = self.h_time_delay[isub]

        if self.iprint:
            print(f"update filter isub={isub}/{self.nspec}")

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

        if rms_noise > 0 and self.delay_noise_shrinkage_threshold is not None:
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

        self.compute_gradient()

        if self.subtract_degenerate_projections:
            self.h_time_delay_grad = subtract_degenerate_dof(self.h_time_delay_grad, self.h_time_delay)

        if self.enforce_causality:
            half_nchan = self.nchan // 2
            self.h_time_delay_grad[:,half_nchan:] = 0

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
            print(f"h_doppler_delay_grad[0,0]={self.h_doppler_delay_grad[0,0]}")
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
        return solve_profile_and_gain(cs, hf, params, update_gain, self.intrinsic_ph_sum, self.intrinsic_ph_sumsq)

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
                print("Warning: couldn't make", plotdir, "not plotting")
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
        use_last_soln: If true, use last filter as initial guess for this subint
        use_minphase: if true, use minimum phase IRF as initial guess
                        else use delta function
        """
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

        self.ph_ref = phase2harm(self.pp_intrinsic)
        self.ph_ref = normalize_profile(self.ph_ref)

        if self.exclude_DC:
            self.ph_ref[0] = 0

        ph = self.ph_ref[:]

        if self.nopt == 0 or not use_last_soln:
            self.pp_intrinsic = np.zeros(self.nphase)
            if ht0 is None:
                if rindex is None:
                    delay = self.phase_gradient(cs)
                else:
                    delay = rindex
                print("initial filter: delta function at delay = %d" % delay)
                ht = np.zeros((self.nlag,), dtype=np.complex128)
                ht[delay] = self.nlag
                if use_minphase:
                    if onp is None:
                        print("onp not specified, so not using minimum phase")
                    else:
                        spect = np.abs(self.data[isub, ipol, :, onp[0] : onp[1]]).mean(1)
                        ht = freq2time(minphase(spect - spect.min()))
                        ht = np.roll(ht, delay)
                        print("using minimum phase with peak at:", np.abs(ht).argmax())
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
        print("max filter index = %d" % self.rindex)

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
        print("variance : %.5e" % var)
        print("nsamp    : %.5e" % nvalid)
        print("dof      : %.5e" % dof)
        print("min obj  : %.5e" % (dof * var))

        tol = 1e-1 / (dof)
        print("ftol     : %.5e" % (tol))
        scipytol = (
            tolfact * tol / 2.220e-16
        )  # 2.220E-16 is machine epsilon, which the scipy optimizer uses as a unit
        print("scipytol : %.5e" % scipytol)
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
        self.pp_intrinsic += pp

        self.nopt += 1

    def cyclic_variance(self, cs):
        ih = self.nharm

        imin, imax = chan_limits_cs(
            iharm=ih, nchan=self.nchan, bw=self.bw, ref_freq=self.ref_freq
        )  # highest harmonic

        var = (np.abs(cs[imin:imax, ih-1]) ** 2).sum()
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
        sopt, gain, ph_numer, ph_denom = self.optimize_profile(plot_cs, self.hf, self.bw, self.ref_freq, False)
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

