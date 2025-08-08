"""
pycyc - python implementation of Cyclic-Modelling https://github.com/demorest/Cyclic-Modelling
Glenn Jones

This is designed to be a library of python functions to be used interactively with ipython
or in other python scripts. However, for demonstration purposes you can run this as a stand-alone script:

python2.7 pycyc.py input_cs_file.ar   # This will generate an initial profile from the data itself

python2.7 pycyc.py input_cs_file.ar some_profile.txt  # This will use the profile in some_profile.txt

The majority of these routines have been checked against the original Cyclic-Modelling code
and produce identical results to floating point accuracy. The results of the optimization may
not be quite as identical since Cyclic-Modelling uses the nlopt implementation of the L_BFGS solver
while this code uses scipy.optimize.fmin_l_bfgs_b

Here's an example of how I use this on kermit.

$ ipython -pylab

import pycyc

# load some 1713 data at 430 MHz that Nipuni processed
CS = pycyc.CyclicSolver(filename='/psr/gjones/2011-09-19-21:50:00.ar')

CS.initProfile(loadFile='/psr/gjones/pp_1713.npy') # start with a nice precomputed profile.
# Note profile can be in .txt (filter_profile) format or .npy numpy.save format.

# have a look at the profile:
plot(CS.pp_intrinsic)

CS.data.shape
Out: (1, 2, 256, 512)  # 1 subintegration, 2 polarizations, 256 freq channels, 512 phase bins

CS.loop(isub = 0, make_plots=True,ipol=0,tolfact=20) # run the optimzation
# Note this does the "inner loop": sets up the non-linear optimization and runs it
# the next step will be to build the "outer loop" which uses the new guess at the intrinsic profile
# to reoptimize the IRF. This isn't yet implemented but the machinary is all there.
#
# Running this with make_plots will create a <filename>_plots subdirectory with plots of various
# data products. The plots are made each time the objective function is evaluated. Note that there
# can be more than one objective function evaluation per solver iteration.
#
# tolfact can be used to reduce the stringency of the stopping criterion. This seems to be particularly useful
# to avoid overfitting to the noise in real data

# after the optimization runs, have a look at the CS data
clf()
imshow((np.abs(CS.cs))[:,1:],aspect='auto')

# have a look at the IRF
ht = pycyc.freq2time(CS.hf_prev)
t = np.arange(ht.shape[0]) / CS.bw # this will be time in microseconds since CS.bw is in MHz
subplot(211)
plot(t,abs(ht))

# compute the model CS and have a look

csm = CS.modelCS(ht)
subplot(212)
imshow(np.log10(np.abs(csm)),aspect='auto')
colorbar()

# Now examine the effect of zeroing the "noisy" parts of the IRF

ht2 = ht[:] #copy ht
ht2[:114] = 0
ht2[143:] = 0
csm2 = CS.modelCS(ht2)
figure()
subplot(211)
plot(t,abs(ht2))
subplot(212)
imshow(np.log10(np.abs(csm2)),aspect='auto')
colorbar()


# Try a bounded optimizaton, constraining h(t) to have support over a limited range
# the two parameters are:
# maxneg : int or None
#    The number of samples before the delta function which are allowed to be nonzero
#    This value must be given to turn on bounded optimization
# maxlen : int or None
#    The maximum length of the impulse response in samples

# e.g. suppose we want to limit the IRF to only have support from -1 to +10 us and CS.bw ~ 10 MHz
# maxneg = int(1e-6 * 10e6) = 10
# maxlen = int((1e-6 + 10e-6) * 10e6) = 110

CS.loop(make_plots=True, tolfact=10, maxneg=10, maxlen = 110)

"""
try:
    import psrchive
except Exception:
    print("pycyc.py: psrchive python libraries not found. You will not be able to load psrchive files.")
import concurrent.futures
import os
import pickle
import sys

import numpy as np
import scipy
import scipy.optimize
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.fft import fft, fftshift, ifft, irfft, rfft
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.signal.windows import kaiser

from plotting import plot_Doppler_vs_delay


class CyclicSolver:
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
        self.include_Nyquist = 1
        self.save_cyclic_spectra = False
        self.pad_cyclic_spectra = True
        self.filenames = []
        self.nspec = 0
        self.nsubint = 0
        self.intrinsic_ph = None
        self.intrinsic_ph_sum = None
        self.intrinsic_ph_sumsq = None
        self.pp_scattered = None
        self.pp_intrinsic = None
        self.shear_phasors = None
        self.cs_norm = None
        self.hf_prev = None
        self.iprint = False
        self.make_plots = False
        self.dump_residual = False
        self.niter = 0

        self.mean_time_offset = 0
        self.nthread = 1

        # modelling options

        # By default, use the integrated pulse profile for all sub-integrations
        self.use_integrated_profile = True

        # omit the spin harmonic DC bin from the definition of the merit function
        # set to 0 or 1; this flag is used as an integer
        self.omit_dc = 1

        # maintain constant total power in the wavefield
        self.conserve_wavefield_energy = False

        # set the wavefield at all negative delays to zero
        self.enforce_causality = False

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

        # delay the initial wavefield estimate by this many pixels
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

        # project the gradient onto the subspace that is orthogonal to the null space direction defined by degenerate phase
        self.manifold_optimization = False

        if filename:
            self.load(filename)

        elif statefile:
            self.loadState(statefile)

        self.statefile = statefile

    def modelCS(self, ht=None, hf=None):
        """
        Convenience function for computing modelCS using ref profile

        Call as modelCS(ht) for time domain or modelCS(hf=hf) for freq domain
        """
        if ht is not None:
            hf = time2freq(ht)
        cs = make_model_cs(hf, self.s0, self.bw, self.ref_freq, self.shear_phasors)

        return cs[0]

    def load_initial_guess(self, filename):
        """
        Load initial guess for wavefield and intrinsic profile from PSRFITS file
        """

        ar = psrchive.Archive_load(filename)
        bw = abs(ar.get_bandwidth())
        ext = ar.get_dynamic_response()
        data = ext.get_data()
        nchan = ext.get_nchan()
        ntime = ext.get_ntime()

        start_time = ext.get_minimum_epoch()
        end_time = ext.get_maximum_epoch()
        dT = (end_time - start_time).in_seconds() / ntime

        print(
            f"{ntime=} start_time={start_time.printdays(13)} end_time={end_time.printdays(13)} delta-T={dT}"
        )

        data = np.reshape(data, (ntime, nchan))

        h_time_delay = freq2time(data, axis=1)
        h_doppler_delay = time2freq(h_time_delay, axis=0)
        plot_Doppler_vs_delay(h_doppler_delay, dT, bw, "input_wavefield.png")

        if self.zap_edges is not None and self.zap_edges > 0:
            zap_count = int(self.zap_edges * nchan)
            data = data[:, zap_count:-zap_count]
            bw = bw * float(zap_count) / float(nchan)

            h_time_delay = freq2time(data, axis=1)
            h_doppler_delay = time2freq(h_time_delay, axis=0)
            plot_Doppler_vs_delay(h_doppler_delay, dT, bw, "input_wavefield_after_zap_edges.png")

        self.initial_h_time_freq = data

        if ar.get_nsubint() == 1:
            self.pp_intrinsic = np.copy(ar.get_Profile(0, 0, 0).get_amps())

    def load(self, filename):
        """
        Load periodic spectrum from psrchive compatible file (.ar or .fits)
        """

        self.filenames.append(filename)
        ar = psrchive.Archive_load(filename)
        if self.pscrunch:
            ar.pscrunch()

        if not self.omit_dc:
            ar.remove_baseline()

        data = ar.get_data()  # we load all data here, so this should probably change in the long run
        if self.zap_edges is not None and self.zap_edges > 0:
            zap_count = int(self.zap_edges * data.shape[2])
            data = data[:, :, zap_count:-zap_count, :]
            bwfact = 1.0 - self.zap_edges * 2
        elif self.maxchan:
            # bwfact used to indicate the actual bandwidth of the data if we're not using all channels.
            bwfact = self.maxchan / (1.0 * data.shape[2])
            data = data[:, :, : self.maxchan, :]
        else:
            bwfact = 1.0

        if self.offp:
            data = data / (np.abs(data[:, :, :, self.offp[0] : self.offp[1]]).mean(3)[:, :, :, None])

        if self.tscrunch:
            for k in range(1, self.tscrunch):
                data[:-k, :, :, :] += data[k:, :, :, :]

        if self.nsubint == 0:
            # print(f'input data have type={data.dtype}')

            idx = 0  # only used to get parameters of integration, not data itself
            subint = ar.get_Integration(idx)
            self.reference_epoch = subint.get_epoch()
            try:
                self.imjd = np.floor(self.reference_epoch)
                self.fmjd = np.fmod(self.reference_epoch, 1)
            except Exception:  # new version of psrchive has different kind of epoch
                self.imjd = self.reference_epoch.intday()
                self.fmjd = self.reference_epoch.fracday()
            self.ref_phase = 0.0
            self.ref_freq = 1.0 / subint.get_folding_period()
            self.bw = np.abs(subint.get_bandwidth()) * bwfact
            self.rf = subint.get_centre_frequency()

            self.source = ar.get_source()  # source name
            self.nopt = 0
            self.nloop = 0

            self.nsubint, self.npol, self.nchan, self.nbin = data.shape
            self.nlag = self.nchan
            self.nphase = self.nbin
            self.nharm = self.nphase // 2 + self.include_Nyquist
            print(f"load nlag={self.nlag} nharm={self.nharm}")
            if self.maxharm is not None:
                print(f"zeroing all harmonics above {self.maxharm} in each cyclic spectrum")

            self.time_offsets = np.zeros(self.nsubint)
            total_offset = 0
            count_offset = 0
            for isub in range(self.nsubint):
                subint = ar.get_Integration(isub)
                epoch = subint.get_epoch()
                diff = epoch - self.reference_epoch
                self.time_offsets[isub] = diff.in_seconds()
                if isub > 0 and self.time_offsets[isub] > 0 and self.time_offsets[isub - 1] > 0:
                    offset = self.time_offsets[isub] - self.time_offsets[isub - 1]
                    total_offset += offset
                    count_offset += 1
            if count_offset > 1:
                self.mean_time_offset = total_offset / count_offset

            ar = None

            if self.save_cyclic_spectra:
                self.cyclic_spectra = np.zeros(
                    (self.nsubint, self.npol, self.nchan, self.nharm),
                    dtype=np.complex128,
                )
                self.cs_norm = np.zeros((self.nsubint, self.npol))
                for isub in range(self.nsubint):
                    if self.iprint:
                        print(f"load calculating cyclic spectrum for isub={isub}/{self.nsubint}")
                    for ipol in range(self.npol):
                        self.cyclic_spectra[isub, ipol], norm = self.get_cs(data[isub, ipol])
                        self.cs_norm[isub, ipol] = norm
                self.data = None
                data = None
            else:
                self.data = data

        else:
            last_offset = self.time_offsets[self.nsubint - 1]
            subint = ar.get_Integration(0)
            epoch = subint.get_epoch()
            diff = epoch - self.reference_epoch
            next_offset = diff.in_seconds()
            gap = next_offset - last_offset

            missing_subints = 0
            if self.mean_time_offset > 0:
                missing_subints = int(np.round(gap / self.mean_time_offset)) - 1

            if missing_subints > 0:
                print(f"missing {missing_subints} sub-integrations across {gap} seconds")
                print(f"mean sub-integration duration is {self.mean_time_offset} seconds")

            nsubint, npol, nchan, nbin = data.shape

            assert npol == self.npol
            assert nchan == self.nchan
            assert nbin == self.nbin

            new_nsubint = self.nsubint + nsubint + missing_subints
            start_isubint = self.nsubint + missing_subints

            self.time_offsets.resize(new_nsubint)
            total_offset = 0
            count_offset = 0
            for isub in range(new_nsubint):
                if isub >= start_isubint:
                    subint = ar.get_Integration(isub - start_isubint)
                    epoch = subint.get_epoch()
                    diff = epoch - self.reference_epoch
                    self.time_offsets[isub] = diff.in_seconds()
                if isub > 0 and self.time_offsets[isub] > 0 and self.time_offsets[isub - 1] > 0:
                    offset = self.time_offsets[isub] - self.time_offsets[isub - 1]
                    total_offset += offset
                    count_offset += 1
            if count_offset > 1:
                self.mean_time_offset = total_offset / count_offset

            if self.save_cyclic_spectra:
                self.cyclic_spectra.resize(new_nsubint, self.npol, self.nchan, self.nharm)
                self.cs_norm.resize(new_nsubint, self.npol)
                self.time_offsets.resize(new_nsubint)
                for isub in range(nsubint):
                    jsub = isub + self.nsubint + missing_subints
                    if self.iprint:
                        print(f"load calculating cyclic spectrum for isub={jsub}/{new_nsubint}")
                    for ipol in range(self.npol):
                        self.cyclic_spectra[jsub, ipol], norm = self.get_cs(data[isub, ipol])
                        self.cs_norm[jsub, ipol] = norm
                self.data = None
                data = None
                if missing_subints > 0:
                    print("setting missing cyclic spectra to average of bounding spectra")
                    for ipol in range(self.npol):
                        previous_idx = self.nsubint - 1
                        previous_cs = self.cyclic_spectra[previous_idx, ipol]
                        previous_cs_norm = self.cs_norm[previous_idx, ipol]

                        next_idx = self.nsubint + missing_subints
                        next_cs = self.cyclic_spectra[next_idx, ipol]
                        next_cs_norm = self.cs_norm[next_idx, ipol]

                        average_cs = 0.5 * (previous_cs + next_cs)
                        average_cs_norm = 0.5 * (previous_cs_norm + next_cs_norm)

                        for isub in range(missing_subints):
                            jsub = isub + self.nsubint
                            self.cyclic_spectra[jsub, ipol] = average_cs
                            self.cs_norm[jsub, ipol] = average_cs_norm

            else:
                if missing_subints > 0:
                    print("WARNING: patching up missing sub-integrations not implemented")
                    print("WARNING: when not saving cyclic spectra")
                self.data = np.append(self.data, data, axis=0)
                new_nsubint -= missing_subints

            self.nsubint = new_nsubint

    def initProfile(self, loadFile=None, maxinitharm=None, maxsubint=None):
        """
        Initialize the reference profile

        If loadFile is not specified, will compute an initial profile from the data
        If loadFile ends with .txt, it is assumed to be a filter_profile output file
        If loadFile ends with .npy, it is assumed to be a numpy data file

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

        if loadFile:
            if loadFile.endswith(".npy"):
                self.pp_intrinsic = np.load(loadFile)
            elif loadFile.endswith(".txt"):
                self.pp_intrinsic = loadProfile(loadFile)
            else:
                raise Exception("Filename must end with .txt or .npy to indicate type")
            return

        self.maxinitharm = maxinitharm
        self.save_dynamic_spectrum = True
        self.save_cs_norm = True
        self.updateProfile()
        self.save_dynamic_spectrum = False
        self.save_cs_norm = False

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

            if self.omit_dc:
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
            self.pp_scattered = np.zeros(self.nphase)  # scattered profile

        self.optimal_gains = np.ones(self.nsubint)
        self.ph_numer = np.zeros((self.npol, self.nspec, self.nharm), dtype=np.complex128)
        self.ph_denom = np.zeros((self.npol, self.nspec, self.nharm), dtype=np.complex128)
        self.intrinsic_ph = np.zeros((self.npol, self.nharm), dtype=np.complex128)

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

        if self.nthread == 1:
            for isub in range(self.nspec):
                self.updateProfileSubint(isub)

        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.nthread) as executor:
                future_subint = {
                    executor.submit(self.updateProfileSubint, isub): isub
                    for isub in range(self.nspec)
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
            cs = cs[:,:self.nharm]

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
            s0 = self.s0
        else:
            ph_numer = self.ph_numer[ipol, isub]
            ph_denom = self.ph_denom[ipol, isub]
            s0 = ph_numer / ph_denom

        ht = self.h_time_delay[isub]

        if self.iprint:
            print(f"update filter isub={isub}/{self.nspec}")

        self.rindex = isub
        _merit, grad, _nterm = complex_cyclic_merit_lag(ht, self, s0, cs, self.optimal_gains[isub])

        if self.enforce_causality:
            half_nchan = self.nchan // 2
            grad[half_nchan:] = 0

        self.h_time_delay_grad[isub, :] += grad

        return _merit, _nterm

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

        if self.manifold_optimization:
            h_time_delay_grad_0 = self.h_time_delay_grad.copy()
            h_time_delay_0 = self.h_time_delay.copy()

            epsilon = 1e-6

            self.h_time_delay = h_time_delay_0 * (1.0 + 1.0j * epsilon)
            self.compute_gradient()
            h_time_delay_grad_2 = self.h_time_delay_grad.copy()

            self.h_time_delay = h_time_delay_0 * (1.0 - 1.0j * epsilon)
            self.compute_gradient()
            h_time_delay_grad_1 = self.h_time_delay_grad.copy()

            # plot_Doppler_vs_delay(time2freq(h_time_delay_grad_0)*self.nsubint, self.mean_time_offset, self.bw, "h_time_delay_grad_0.png")

            # plot_Doppler_vs_delay(time2freq(self.h_time_delay_grad)*self.nsubint, self.mean_time_offset, self.bw, "h_time_delay_grad.png")

            delta_grad = h_time_delay_grad_2 - h_time_delay_grad_1

            plot_Doppler_vs_delay(
                time2freq(delta_grad) * self.nsubint, self.mean_time_offset, self.bw, "delta_grad.png"
            )

            self.h_time_delay = h_time_delay_0
            self.h_time_delay_grad = h_time_delay_grad_0

            proj = np.vdot(delta_grad, self.h_time_delay_grad)
            print(f"{proj=}")
            self.h_time_delay_grad -= proj * delta_grad / np.sum(np.abs(delta_grad) ** 2)

            # plot_Doppler_vs_delay(time2freq(self.h_time_delay_grad)*self.nsubint, self.mean_time_offset, self.bw, "h_time_delay_grad_again.png")

        if self.reduce_temporal_phase_noise_grad:
            minimize_temporal_phase_noise(self.h_time_delay_grad)

        dumps = False

        if dumps:
            with open("h_time_delay_grad.pkl", "wb") as fh:
                print("DUMPING TIME DELAY GRADIENT")
                pickle.dump(self.h_time_delay_grad, fh)

        self.h_doppler_delay_grad = time2freq(self.h_time_delay_grad)

        if dumps:
            with open("h_doppler_delay_grad.pkl", "wb") as fh:
                print("DUMPING DOPPLER DELAY GRADIENT")
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
            self.s0 = self.intrinsic_ph[ipol]
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

                    except Exception as exc:
                        print(f"compute_gradient isub={isub} exception: {exc}")

    def optimize_profile(self, cs, hf, bw, ref_freq, update_gain):
        hfplus, hfminus = shear_spectra(hf, self.shear_phasors)

        # cs H(-)H(+)*
        cshmhp = cs * hfminus * np.conj(hfplus)
        # |H(-)|^2 |H(+)|^2
        maghmhp = (np.abs(hfminus) * np.abs(hfplus)) ** 2

        if update_gain and self.intrinsic_ph_sum is not None:
            # Equation A11 numerator
            tmp = fscrunch_cs(np.conj(cshmhp) * self.intrinsic_ph_sum, bw=bw, ref_freq=ref_freq, padding=self.pad_cyclic_spectra)
            gain_numer = tmp[1:].sum()  # sum over all harmonics
            # Equation A11 denominator
            tmp = fscrunch_cs(maghmhp * self.intrinsic_ph_sumsq, bw=bw, ref_freq=ref_freq, padding=self.pad_cyclic_spectra)
            gain_denom = tmp[1:].sum()  # sum over all harmonics
            gain = np.real(gain_numer) / np.real(gain_denom)
            print(f'optimize_profile gain={gain}')
        else:
            gain = 1

        # fscrunch
        # print("optimize_profile fscrunch_cs")
        ph_numer = fscrunch_cs(cshmhp, bw=bw, ref_freq=ref_freq, padding=self.pad_cyclic_spectra) * gain
        ph_denom = fscrunch_cs(maghmhp, bw=bw, ref_freq=ref_freq, padding=self.pad_cyclic_spectra) * gain**2
        s0 = ph_numer / ph_denom
        s0[np.real(ph_denom) <= 0.0] = 0
        return s0, gain, ph_numer, ph_denom

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
        self.make_plots = make_plots
        if make_plots:
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

        if self.omit_dc:
            self.ph_ref[0] = 0

        ph = self.ph_ref[:]
        self.s0 = ph

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
            b = get_params(minbound, rindex)
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
        x0 = get_params(ht, rindex)

        self.niter = 0
        self.objval = []

        self.cs = cs
        x, f, d = scipy.optimize.fmin_l_bfgs_b(
            cyclic_merit_lag,
            x0,
            m=20,
            args=(self,),
            iprint=iprint,
            maxfun=maxfun,
            factr=scipytol,
            bounds=bounds,
        )
        ht = get_ht(x, rindex)

        if self.delay_taper is not None:
            ht *= self.delay_taper

        hf = time2freq(ht)

        self.hf_soln = hf[:]

        hf = match_two_filters(_hf_prev, hf)
        self.optimized_filters[isub, :] = hf
        self.hf_prev = hf.copy()

        update_gain = False
        ph, gain, ph_numer, ph_denom = self.optimize_profile(cs, hf, self.bw, self.ref_freq, update_gain)

        if self.omit_dc:
            ph[0] = 0.0

        pp = self.harm2phase(ph)

        self.intrinsic_profiles[isub, :] = pp
        self.pp_intrinsic += pp

        self.nopt += 1

    def saveResults(self, fbase=None):
        if fbase is None:
            fbase = self.filename
        writeProfile(fbase + ".pp_intrinsic.txt", self.pp_intrinsic)
        writeProfile(fbase + ".pp_scattered.txt", self.pp_scattered)
        writeArray(fbase + ".hfs.txt", self.optimized_filters)
        writeArray(fbase + ".dynspec.txt", self.dynamic_spectrum)

    def cyclic_variance(self, cs):
        ih = self.nharm

        imin, imax = chan_limits_cs(
            iharm=ih, nchan=self.nchan, bw=self.bw, ref_freq=self.ref_freq
        )  # highest harmonic
        var = (np.abs(cs[imin:imax, ih]) ** 2).sum()
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

    def saveState(self, filename=None):
        """
        not yet ready for use
        Save current state of this class (inlcuding current CS solution)
        """
        # For now we just use pickle for convenience. In the future, could use np.savez or HDF5 (or FITS)
        if filename is None:
            if self.statefile:
                filename = self.statefile
            else:
                filename = self.filename + ".cysolve.pkl"
        orig_statefile = self.statefile

        fh = open(filename, "w")
        pickle.dump(self, fh, protocol=-1)
        fh.close()

        self.statefile = orig_statefile
        print("Saved state in:", filename)

    def harm2phase(self, ph, workers=1):
        if self.include_Nyquist:
            tmp = ph
        else:
            nharm = ph.shape[0]
            tmp = np.zeros(nharm+1,dtype=np.complex128)
            tmp[:nharm]=ph[:]
        return irfft(tmp, workers=workers, norm="ortho")

    def plotCurrentSolution(self, plot_cs):
        cs_model = self.model
        grad = self.grad
        hf = self.hf
        ht = self.ht
        mlag = self.mlag
        fig = Figure()
        ax1 = fig.add_subplot(3, 3, 1)
        csextent = [1, mlag - 1, self.rf + self.bw / 2.0, self.rf - self.bw / 2.0]
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
        sopt, gain, ph_numer, ph_denom = self.optimize_profile(plot_cs, hf, self.bw, self.ref_freq)
        sopt = normalize_profile(sopt)

        if self.omit_dc:
            sopt[0] = 0.0
        smeas = normalize_profile(plot_cs.mean(0))
        if self.omit_dc:
            smeas[0] = 0.0
        #        cs_model0,hfplus,hfminus,phases = make_model_cs(hf,sopt,self.bw,self.ref_freq)

        ax3 = fig.add_subplot(3, 3, 7)
        #        ax3.imshow(np.log(np.abs(cs_model0)[:,1:]),aspect='auto')
        err = np.abs(plot_cs - cs_model)[:, 1:mlag]
        # err = cs2ps(plot_cs) - cs2ps(normalize_cs(cs_model,self.bw,self.ref_freq))
        im = ax3.imshow(err, aspect="auto", interpolation="nearest", extent=csextent)
        ax3.set_xlim(0, mlag)
        #        im.set_clim(err[1:-1,1:-1].min(),err[1:-1,1:-1].max())
        im.set_clim(0, 3 * self.noise)
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
        t = np.arange(ht.shape[0]) / self.bw
        ax4.plot(t, np.roll(20 * np.log10(np.abs(ht)), int(ht.shape[0] / 2) - self.rindex))
        ax4.plot(
            t,
            np.roll(
                20 * np.log10(np.convolve(np.ones((10,)) / 10.0, np.abs(ht), mode="same")),
                int(ht.shape[0] / 2) - self.rindex,
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
        f = np.linspace(self.rf + self.bw / 2.0, self.rf - self.bw / 2.0, self.nchan)
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
        if len(self.objval) >= 3:
            x = np.abs(np.diff(np.array(self.objval).flatten()))
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
        pref = self.harm2phase(self.s0)
        ax6.plot(pref, label="Reference", linewidth=2)
        ax6.plot(self.harm2phase(sopt), "r", label="Intrinsic")
        ax6.plot(self.harm2phase(smeas), "g", label="Measured")
        legend = ax6.legend(loc="upper left", prop=dict(size="xx-small"), title="Profiles")
        legend.get_frame().set_alpha(0.5)
        ax6.set_xlim(0, pref.shape[0])
        fname = self.filename[-50:]
        if len(self.filename) > 50:
            fname = "..." + fname
        title = "%s isub: %d nopt: %d\n" % (fname, self.isub, self.nopt)
        title += "Source: %s Freq: %s MHz Feval #%04d Merit: %.3e Grad: %.3e" % (
            self.source,
            self.rf,
            self.niter,
            self.objval[-1],
            np.abs(grad).sum(),
        )
        fig.suptitle(title, size="small")
        canvas = FigureCanvasAgg(fig)
        fname = os.path.join(self.plotdir, ("%s_%04d_%04d.png" % (self.source, self.nopt, self.niter)))
        canvas.print_figure(fname)


def safely_divide(numer,denom):
    safe_denom = np.where(denom == 0, 1, denom)
    result = numer / safe_denom
    result[denom == 0] = 0
    return result

def plotSimulation(CS, mlag=100):
    if CS.ht0 is None:
        print("Does not appear this is a simulation run")
    CS.grad
    hf = CS.hf
    ht = CS.ht  # [CS.isub,:]
    cs0 = CS.modelCS(ht)
    t = np.arange(ht.shape[0]) / CS.bw
    f = np.linspace(CS.rf + CS.bw / 2.0, CS.rf - CS.bw / 2.0, CS.nchan)
    csextent = [1, mlag - 1, CS.rf - CS.bw / 2.0, CS.rf + CS.bw / 2.0]
    ht0 = CS.ht0[CS.isub]
    hf0 = match_two_filters(hf, time2freq(ht0))
    cs_model = CS.modelCS(ht0)

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
    pt = 1e3 * np.linspace(0, 1, CS.nphase) / CS.ref_freq
    ax3.plot(pt, fftshift(CS.pp_meas), "r", label="measured")
    ax3.plot(pt, fftshift(CS.pp_intrinsic), "cyan", alpha=0.7, label="deconvolved")
    ax3.plot(pt, fftshift(self.harm2phase(CS.s0)), "k", label="original")
    ax3.errorbar(
        [pt[len(pt) / 4]],
        [CS.pp_meas.max() / 2.0],
        xerr=CS.tau / 1e3,
        capsize=5,
        linewidth=2,
    )
    ax3.text(pt[len(pt) / 4], CS.pp_meas.max() * 0.55, "tau", fontdict=dict(size="small"))
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
    #    im = ax5.imshow(np.abs(CS.cs[:,:mlag]/cs0[:,:mlag]),
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

    left = -5 * CS.tau
    right = 20 * CS.tau
    if right > t[-1] - maxt:
        right = t[-1] - maxt
    ax8.set_xlim(left, right)
    legend = ax8.legend(loc="upper right", prop=dict(size="x-small"))
    legend.get_frame().set_alpha(0.5)
    ax8.set_xlabel(r"$\mu$s")

    ax6 = fig.add_subplot(3, 3, 6)
    im = ax6.imshow(
        np.angle(CS.cs[:, :mlag] / cs0[:, :mlag]),
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
        np.log10(np.abs(CS.cs[:, :mlag])),
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
        np.angle(CS.cs[:, :mlag]),
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

    fname = CS.filename[-50:]
    if len(CS.filename) > 50:
        fname = "..." + fname
    try:
        harmstr = "Harmonics: %d" % CS.pharm
    except AttributeError:
        harmstr = ""

    snrstr = ""
    try:
        taustr = "h(t) tau: %.1f" % CS.tau
        if CS.noise is not None:
            snrstr = "snr: %.3f" % CS.noise
    except AttributeError:
        taustr = ""

    title = "%s isub: %d ipol: %d nopt: %d\n" % (fname, CS.isub, CS.ipol, CS.nopt)
    title += (
        harmstr + " " + taustr + " " + snrstr + " " + (" Feval #%04d Merit: %.3e" % (CS.niter, CS.objval[-1]))
    )
    fig.suptitle(title, size="small")
    canvas = FigureCanvasAgg(fig)
    fname = os.path.join(
        CS.plotdir,
        ("sim_SNR_%.1f_%s_%04d_%04d.pdf" % (CS.noise, CS.source, CS.nopt, CS.niter)),
    )
    canvas.print_figure(fname)


def fold(v):
    """
    Fold negative response onto positive time for minimum phase calculation
    """
    n = v.shape[0]
    nt = int(n / 2)
    rf = np.zeros_like(v[:nt])
    rf[:-1] = v[1:nt]
    rf += np.conj(v[: nt - 1 : -1])
    rw = np.zeros_like(v)
    rw[0] = v[0]
    rw[1 : nt + 1] = rf
    return rw


def minphase(v, workers=1):
    clipped = v.copy()
    thresh = 1e-5
    clipped[np.abs(v) < thresh] = thresh
    return np.exp(fft(fold(ifft(np.log(clipped), workers=workers)), workers=workers))


def loadArray(fname):
    """
    Load array from txt file in format generated by filter_profile,
    useful for filters.txt, dynamic_spectrum.txt
    """
    fh = open(fname, "r")
    try:
        x = int(fh.readline())
    except Exception:
        raise Exception("couldn't read first dimension")
    try:
        y = int(fh.readline())
    except Exception:
        raise Exception("couldn't read second dimension")
    raw = np.loadtxt(fh)
    if raw.shape[0] != x * y:
        raise Exception(
            "number of rows of data=",
            raw.shape[0],
            " not equal to product of dimensions:",
            x,
            y,
        )
    if len(raw.shape) > 1:
        data = raw[:, 0] + raw[:, 1] * 1j
    else:
        data = raw[:]
    data.shape = (x, y)
    fh.close()
    return data


def writeArray(fname, arr):
    """
    Write array to ascii file in same format as filter_profile does
    """
    fh = open(fname, "w")
    fh.write("%d\n" % arr.shape[0])
    fh.write("%d\n" % arr.shape[1])
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            if arr.dtype == np.complex128:
                fh.write("%.7e %.7e\n" % (arr[x, y].real, arr[x, y].imag))
            else:
                fh.write("%.7e\n" % (arr[x, y]))
    fh.close()


def writeProfile(fname, prof):
    """
    Write profile to ascii file in same format as filter_profile does
    """
    t = np.linspace(0, 1, prof.shape[0], endpoint=False)
    fh = open(fname, "w")
    for x in range(prof.shape[0]):
        fh.write("%.7e %.7e\n" % (t[x], prof[x]))
    fh.close()


def loadProfile(fname):
    """
    Load profile in format generated by filter_profile
    """
    x = np.loadtxt(fname)
    return x[:, 1]

#
# norm="ortho" is used in the following definitions in order to preserve power
#

def ps2cs(ps, workers=2):
    """
    real-valued periodic spectrum to complex-valued cyclic spectrum
    real-to-complex forward FFT transforms pulse phase to cycle frequency
    the Nyquist harmonic is removed because it typically contains little information,
    and its removal facilitates comparison with simulated cyclic spectra
    """
    return rfft(ps, axis=1, workers=workers, norm="ortho")


def cc2cs(cs, workers=2, axis=0):
    """
    complex-valued periodic correlation to cyclic correlation
    complex-to-complex forward FFT transforms lag to radio frequency
    """
    return fft(cs, axis=axis, workers=workers, norm="ortho")

def cs2cc(cc, workers=2, axis=0):
    """
    complex-valued periodic correlation to cyclic correlation
    complex-to-complex backward/inverse FFT transforms radio frequency to lag
    """
    return ifft(cc, axis=axis, workers=workers, norm="ortho")

def cc2pc(cc, workers=2, axis=1):
    """
    complex-valued periodic correlation to cyclic correlation
    complex-to-complex backward/inverse FFT transforms cycle frequency to pulse phase
    """
    return ifft(cc, axis=axis, workers=workers, norm="ortho")

def pc2cc(pc, workers=2, axis=1):
    """
    complex-valued periodic correlation to cyclic correlation
    complex-to-complex forward FFT transforms pulse phase to cycle frequency
    """
    return fft(pc, axis=axis, workers=workers, norm="ortho")


def time2freq(ht, workers=1, axis=0):
    return fft(ht, axis=axis, workers=workers, norm="ortho")


def freq2time(hf, workers=1, axis=0):
    return ifft(hf, axis=axis, workers=workers, norm="ortho")


def phase2harm(pp, workers=1):
    return rfft(pp, workers=workers, norm="ortho")


def match_two_filters(hf1, hf2):
    z = (hf1 * np.conj(hf2)).sum()
    z2 = (hf2 * np.conj(hf2)).sum()  # = np.abs(hf2)**2.sum()
    z /= np.abs(z)
    z *= np.sqrt(1.0 * hf1.shape[0] / np.real(z2))
    return hf2 * z


def pad_wavefield(h_time_freq, new_shape):
    old_shape = h_time_freq.shape
    h_time_delay = freq2time(h_time_freq, axis=1)
    h_doppler_delay = time2freq(h_time_delay, axis=0)

    padded_spectrum = np.zeros(new_shape, dtype=np.complex128)

    first_half_time = old_shape[0] // 2
    last_half_time = old_shape[0] - first_half_time

    padded_spectrum[:first_half_time, : old_shape[1]] = h_doppler_delay[:first_half_time, :]
    padded_spectrum[-last_half_time:, : old_shape[1]] = h_doppler_delay[-last_half_time:, :]

    h_time_delay = freq2time(padded_spectrum, axis=0)
    return time2freq(h_time_delay, axis=1)


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


def normalize_profile(ph):
    """
    Normalize harmonic profile such that first harmonic has magnitude 1
    """
    return ph / np.abs(ph[1])


def normalize_pp(pp):
    """
    Normalize a profile but keep it in phase rather than harmonics
    """
    ph = phase2harm(pp)
    ph = normalize_profile(ph)
    ph[0] = 0
    return harm2phase(ph)


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


def cyclic_padding(cs, bw, ref_freq):
    nharm = cs.shape[1]
    nchan = cs.shape[0]
    for ih in range(nharm):
        imin, imax = chan_limits_cs(ih, nchan, bw, ref_freq)
        cs[:imin, ih] = 0
        cs[imax:, ih] = 0
    return cs


def chan_limits_cs(iharm, nchan, bw, ref_freq):
    chanbw_Hz = bw * 1e6 / nchan  # width of FFT bins in radio frequency Hz
    shift_Hz = iharm * ref_freq / 2
    ichan = round(shift_Hz / chanbw_Hz)
    if ichan > nchan / 2:
        ichan = int(nchan / 2)
    return (ichan, nchan - ichan)  # min,max


def create_shear_phasors(nchan, nharm, bw_MHz, freq_Hz):
    """Construct the two-dimensional array of phasors used to shift the spectral response function

    Specifically, returns exp(i pi \tau_j \alpha_k)

    Parameters
    ----------
    nchan : number of frequency channels in spectra to be shifted
    nharm : number of shifts to perform
    bw_MHz : bandwidth in MHz
    freq_Hz : modulation frequency in Hz

    Returns
    ------
    .. math::
        exp(i pi \tau_j \alpha_k)

    where i=sqrt(-1), j = row index, and k = column index
    """

    # tau[j] in seconds
    tau = np.fft.fftfreq(nchan) * (nchan * 1e-6 / bw_MHz)
    # tau = np.linspace(0.0, 1.0 - 1.0 / nchan, nchan) * (nchan * 1e-6 / bw_MHz)
    # alpha[k] in Hz
    alpha = freq_Hz * np.arange(nharm)

    # shift by 2 pi alpha / 2
    return np.exp(1j * np.pi * np.outer(tau, alpha))


def shear_spectra(spectrum, phasors):
    """Shifts the spectrum for each column of phasors

    Parameters
    ----------
    spectra : one-dimensional spectrum to be shifted
    phasors : the phase gradients for each shift to be applied

    """

    tmp = ifft(spectrum)
    nharm = phasors.shape[1]

    # copy the spectrum for each shift for each harmonic
    spectra = np.repeat(tmp[:, np.newaxis], nharm, axis=1)
    return fft(spectra * np.conj(phasors), axis=0), fft(spectra * phasors, axis=0)


def make_model_cs(hf, s0, bw, ref_freq, phasors, padding=True):
    nchan = hf.shape[0]
    s0.shape[0]
    # profile2cs
    cs = np.repeat(
        s0[np.newaxis, :], nchan, axis=0
    )  # fill the cs model with the harmonic profile for each freq chan

    hfplus, hfminus = shear_spectra(hf, phasors)

    cs = cs * hfplus * np.conj(hfminus)

    if padding:
        cs = cyclic_padding(cs, bw, ref_freq)

    return cs, hfplus, hfminus


def fscrunch_cs(cs, bw, ref_freq, padding):
    cstmp = cs[:]
    if padding:
        cstmp = cyclic_padding(cstmp, bw, ref_freq)
    #    rm = np.abs(cs-cstmp).sum()
    #    print "fscrunch saved:",rm
    return cstmp.sum(0)


def get_params(ht, rindex):
    nlag = ht.shape[0]
    params = np.zeros((2 * nlag - 1,), dtype="float")
    if rindex > 0:
        params[: 2 * (rindex)] = ht[:rindex].view("float")
    params[2 * rindex] = ht[rindex].real
    if rindex < nlag - 1:
        params[2 * rindex + 1 :] = ht[rindex + 1 :].view("float")
    return params


def get_ht(params, rindex):
    nlag = int((params.shape[0] + 1) / 2)
    ht = np.zeros((nlag,), dtype=np.complex128)
    ht[:rindex] = params[: 2 * rindex].view(np.complex128)
    ht[rindex] = params[2 * rindex]
    ht[rindex + 1 :] = params[2 * rindex + 1 :].view(np.complex128)
    return ht


def cyclic_merit_lag(x, CS):
    """
    The objective function. Computes mean squared merit and gradient

    Format is compatible with scipy.optimize
    """
    print("rindex", CS.rindex)
    ht = get_ht(x, CS.rindex)
    merit, grad, nonzero = complex_cyclic_merit_lag(ht, CS, CS.s0, CS.cs, 1.0)
    # the objval list keeps track of how the convergence is going
    CS.objval.append(merit)

    # multiply by 2 when going from Wertinger to real/imag derivatives
    grad = get_params(2.0 * grad, CS.rindex)
    return merit, grad


def complex_cyclic_merit_lag(ht, CS, s0, cs_data, gain):
    hf = time2freq(ht)
    cs_model, hfplus, hfminus = make_model_cs(
        hf, s0, CS.bw, CS.ref_freq, CS.shear_phasors, CS.pad_cyclic_spectra
    )
    cs_model *= gain

    if CS.maxharm is not None:
        cs_model[:, CS.maxharm + 1 :] = 0.0

    extract = cs_model[:, CS.omit_dc :]
    nonzero = np.count_nonzero(extract)

    # WDvS13 Equation 19 and eqn:merit_function of appendix
    merit = (np.abs(extract - cs_data[:, CS.omit_dc :]) ** 2).sum()

    # residual, R = model - data
    diff = cs_model - cs_data

    if CS.dump_residual:
        print(
            f"complex_cyclic_merit_lag power in diff={np.sum(np.abs(diff)**2)} model={np.sum(np.abs(cs_model)**2)} data={np.sum(np.abs(cs_data)**2)}"
        )
        filename = f"complex_cyclic_merit_lag_residual_{CS.rindex:03d}.pkl"
        with open(filename, "wb") as fh:
            pickle.dump(diff, fh)

    phasors = CS.shear_phasors

    # make nchan / nlag copies of the intrinsic profile
    cs0 = np.repeat(s0[np.newaxis, :], CS.nlag, axis=0)

    cc1 = cs2cc(diff * hfminus)
    grad2 = cc1 * phasors * np.conj(cs0)  # WDvS Equation 37
    grad_sum1 = grad2[:, CS.omit_dc :].sum(1)  # sum over all harmonics

    cc1 = cs2cc(np.conj(diff) * hfplus)
    grad2 = cc1 * np.conj(phasors) * cs0
    grad_sum2 = grad2[:, CS.omit_dc :].sum(1)  # sum over all harmonics

    test_conjugacy = False
    if test_conjugacy:
        test = grad_sum1 - np.conj(grad_sum2)
        test_power = (np.abs(test) ** 2).sum()
        sum1_power = (np.abs(grad_sum1) ** 2).sum()
        sum2_power = (np.abs(grad_sum2) ** 2).sum()
        print(
            f"complex_cyclic_merit_lag relative power in test={test_power / np.sqrt(sum1_power * sum2_power)}"
        )

    grad = grad_sum1 + grad_sum2

    if CS.iprint:
        print("merit= %.7e  grad= %.7e" % (merit, (np.abs(grad) ** 2).sum()))

    return merit, grad, nonzero


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


def circular(x):
    x[:] = np.fmod(x, 2.0 * np.pi)


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


def spectral_shift(alpha, hf):
    """
    Return the input mupltiplied by a phase and phase gradient

    Args:
    alpha: A 1D array of two values: phase, and slope
    hf: the complex-valued spectra that is transformed

    Returns:
    The transformed input
    """

    phase = alpha[0]
    slope = alpha[1]
    nus = np.fft.fftfreq(hf.size)
    return hf * np.exp(1j * (phase + slope * nus)), nus


def spectral_distance(alpha, hf_ref, hf):
    """
    Calculates the magnitude of the difference between the two spectra in hf_ref and hf
    after mupltiplying the second one by a phase and phase gradient

    Args:
    alpha: A 1D array of two values: phase, and slope
    hf_ref: the complex-valued spectra used as a reference
    hf: the complex-valued spectra that is aligned to hf_ref by minimizing distance

    Returns:
    The distance and its gradient with respect to the 3 parameters in alpha
    """

    alpha[0]
    alpha[1]
    # print(f"{phase=} {slope=}")

    hfprime, nus = spectral_shift(alpha, hf)
    delta = hf_ref - hfprime

    diff = np.sum(np.abs(delta) ** 2)

    del_phase = 1j * hfprime
    del_slope = 1j * nus * hfprime

    ddiff_dphs = -2 * np.sum(np.real(np.conj(delta) * del_phase))
    ddiff_dslo = -2 * np.sum(np.real(np.conj(delta) * del_slope))

    # print(f"{diff=} del_phi={ddiff_dphs} del_eps={ddiff_dslo}")
    return diff, [ddiff_dphs, ddiff_dslo]


def minimize_difference(hf_ref, hf):
    Nchan = hf_ref.size
    initial_guess = np.zeros(2)

    align_power = True
    if align_power:
        ht = np.abs(ifft(hf)) ** 2
        ht_ref = np.abs(ifft(hf_ref)) ** 2
        ccf_power = np.correlate(ht, ht_ref, "same")
        imax = np.argmax(ccf_power)
        ph_max = 0
    else:
        ccf = fft(np.conj(hf) * hf_ref)
        ccf_power = np.abs(ccf) ** 2
        imax = np.argmax(ccf_power)
        ph_max = np.angle(ccf[imax])

    # print(f"{imax=} {ph_max=} {Nchan=}")

    initial_guess[0] = ph_max

    if imax < Nchan // 2:
        slope = imax
    else:
        slope = imax - Nchan

    initial_guess[1] = slope * 2.0 * np.pi

    options = {"maxiter": 1000, "gtol": 1e-9}

    result = minimize(
        spectral_distance,
        initial_guess,
        args=(hf_ref, hf),
        method="BFGS",
        jac=True,
        options=options,
    )

    alpha = result.x
    hf[:], nus = spectral_shift(alpha, hf)

    # cross-correlation at lag 0
    ccf0 = np.sum(np.conj(hf) * hf_ref)
    # total spectral power in reference spectrum
    tsp0 = np.sum(np.abs(hf_ref) ** 2)
    tsp = np.sum(np.abs(hf) ** 2)
    return ccf0 / np.sqrt(tsp0 * tsp)


def align_to_neighbour(h_time_freq):
    nt, nf = h_time_freq.shape
    hf0 = h_time_freq[0]
    for it in range(1, nt):
        hf = h_time_freq[it]
        minimize_difference(hf0, hf)
        hf0 = hf
        h_time_freq[it, :] = hf


def shifted(input_array, fraction_of_bin, axis=0):
    # Get the shape of the input array
    shape = input_array.shape

    # Create an array of sample frequencies
    frequency = np.fft.fftfreq(shape[axis])

    # Calculate the shift in radians based on the fraction of a bin
    phase_shift = 2 * np.pi * fraction_of_bin

    if axis == 0:
        return input_array * np.exp(1j * phase_shift * frequency[:, np.newaxis])
    elif axis == 1:
        return input_array * np.exp(1j * phase_shift * frequency[np.newaxis, :])
    else:
        raise ValueError("Invalid axis value. Must be 0 or 1.")


def loadCyclicSolver(statefile):
    """
    Load previously saved Cyclic Solver class
    """
    with open(statefile, "rb") as fh:
        cys = pickle.load(fh)
    return cys


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
