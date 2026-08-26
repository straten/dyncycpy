"""
pycyc.io - PSRFITS/pickle I/O for CyclicSolver.

These are methods, not pure functions: loading and saving inherently mean
mutating (or serializing) a CyclicSolver instance's state, so unlike
pycyc.objective/pycyc.profile there is no natural "pure function of explicit
parameters" form here. _IOMixin is inherited by CyclicSolver (pycyc.solver)
so CS.load(...)/.unload_solution(...)/etc. keep working exactly as before;
this file exists purely to keep pycyc/solver.py from also having to hold
every PSRFITS/pickle detail.
"""

__all__ = ["_IOMixin", "loadCyclicSolver"]

import pickle

import numpy as np

try:
    import psrchive
except Exception:
    pass  # pycyc.solver already prints a warning; methods here will raise if psrchive is actually needed

from plotting import plot_Doppler_vs_delay

from .io_utils import writeArray, writeProfile
from .transforms import freq2time, phase2harm, time2freq


class _IOMixin:
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

        print("load_initial_guess loaded:")
        print(f"\t start_time={start_time.printdays(13)} end_time={end_time.printdays(13)}")
        print(f"\t {ntime=} delta-T={dT} -- {nchan=} {bw=}")
        data = np.reshape(data, (ntime, nchan))

        h_time_delay = freq2time(data, axis=1)
        h_doppler_delay = time2freq(h_time_delay, axis=0)

        if self.enforce_real_at_origin:
            h_doppler_delay[0, 0] = np.real(h_doppler_delay[0, 0])

        plot_Doppler_vs_delay(h_doppler_delay, dT, bw, "input_wavefield.png")

        if self.zap_initial_guess and self.zap_edges is not None and self.zap_edges > 0:
            # nsubint, nchan = data.shape
            # print(f"load_initial_guess before zapping {self.zap_edges}: {nsubint=} {nchan=}")
            zap_count = int(self.zap_edges * nchan)
            data = data[:, zap_count:-zap_count]
            # nsubint, nchan = data.shape
            # print(f"load_initial_guess after zapping: {nsubint=} {nchan=}")
            bw = bw * float(zap_count) / float(nchan)

            h_time_delay = freq2time(data, axis=1)
            h_doppler_delay = time2freq(h_time_delay, axis=0)
            plot_Doppler_vs_delay(h_doppler_delay, dT, bw, "input_wavefield_after_zap_edges.png")

        self.initial_h_time_freq = data

        self.load_initial_profile(filename)
        ar = None

    def load_initial_profile(self, filename):
        """
        Load initial guess for intrinsic profile from file
        """

        ar = psrchive.Archive_load(filename)
        assert ar.get_nsubint() >= 1
        tmp = ar.get_Profile(0, 0, 0).get_amps()
        self.pp_intrinsic = np.copy(tmp)
        tmp = phase2harm(self.pp_intrinsic)
        self.intrinsic_ph = np.zeros((1, np.size(tmp)), dtype=np.complex128)
        self.intrinsic_ph[0, :] = tmp[None, :]
        ar = None

    def load(self, filename):
        """
        Load periodic spectrum from psrchive compatible file (.ar or .fits)
        """

        print(f"loading {filename}")
        self.filenames.append(filename)
        # self.filename (singular) is read all over CyclicSolver -- plot
        # titles/directory names, saveResults' default fbase, saveState's
        # default filename -- but was never actually assigned anywhere.
        self.filename = filename
        ar = psrchive.Archive_load(filename)
        if self.pscrunch:
            ar.pscrunch()

        if not self.remove_baseline:
            ar.remove_baseline()

        data = ar.get_data()  # we load all data here, so this should probably change in the long run
        if self.zap_edges is not None and self.zap_edges > 0:
            zap_count = int(self.zap_edges * data.shape[2])
            # print(f"{zap_count=}")
            # nsubint, npol, nchan, nbin = data.shape
            # print(f"before zapping {self.zap_edges}: {nsubint=} {npol=} {nchan=} {nbin=}")
            data = data[:, :, zap_count:-zap_count, :]
            bwfact = 1.0 - self.zap_edges * 2
            # nsubint, npol, nchan, nbin = data.shape
            # print(f"after zapping {self.zap_edges}: {nsubint=} {npol=} {nchan=} {nbin=}")

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
            # print(f"load nphase={self.nphase} nlag={self.nlag} nharm={self.nharm} inc_Nyquist={self.include_Nyquist}")
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

            if gap < 0:
                print(f"last_offset={last_offset} next_offset={next_offset} gap={gap}")
                raise ValueError("new file starts before previous one ended (sort files by time)")

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

            # print(f"load expanding to {new_nsubint} sub-integrations {self.nsubint=} {nsubint=} {missing_subints=}")
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

    def unload_solution(self, filename):
        arch = psrchive.Archive_new_Archive("PSRFITS")

        # use the first file and all of its metadata to create a new archive
        copy = psrchive.Archive_load(self.filenames[0])
        copy.fscrunch()
        arch.copy(copy)

        # unload the wavefield as a dynamic response
        ext = arch.add_dynamic_response()

        ext.set_nchan(self.nchan)
        ext.set_ntime(self.nsubint)
        ext.set_npol(1)

        ext.resize_data()

        h_time_freq = time2freq(self.h_time_delay, axis=1)
        ext.set_data(h_time_freq.flatten())

        start_time = self.reference_epoch
        end_time = start_time + self.mean_time_offset * self.nsubint

        print(f"unload_solution start_time={start_time.printdays(13)} end_time={end_time.printdays(13)}")
        ext.set_minimum_epoch(start_time)
        ext.set_maximum_epoch(end_time)

        ext.set_centre_frequency(self.rf)
        ext.set_bandwidth(self.bw)

        # unload the intrinsic profile

        print("resizing archive sub-integrations")

        nsub = nchan = 1
        arch.resize(nsub, self.npol, nchan, self.nbin)

        print("setting profile data")
        for subint in arch:
            for ipol in range(self.npol):
                for ichan in range(nchan):
                    prof = subint.get_Profile(ipol, ichan)
                    prof.get_amps()[:] = self.pp_intrinsic

        print("unload_solution writing to", filename)
        arch.unload(filename)

    def saveResults(self, fbase=None):
        if fbase is None:
            fbase = self.filename
        writeProfile(fbase + ".pp_intrinsic.txt", self.pp_intrinsic)
        writeProfile(fbase + ".pp_scattered.txt", self.pp_scattered)
        writeArray(fbase + ".hfs.txt", self.optimized_filters)
        writeArray(fbase + ".dynspec.txt", self.dynamic_spectrum)

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

        fh = open(filename, "wb")
        pickle.dump(self, fh, protocol=-1)
        fh.close()

        self.statefile = orig_statefile
        print("Saved state in:", filename)


def loadCyclicSolver(statefile):
    """
    Load previously saved Cyclic Solver class
    """
    with open(statefile, "rb") as fh:
        cys = pickle.load(fh)
    return cys
