#!/usr/bin/env python
# coding: utf-8

'''Generates average impulse responses that can be used as giant pulse detection filters,
   as described as "the first cycle" in Section 4 of Mahajan & van Kerkwijk (2023)'''

import numpy as np
import psrchive

import argparse
import time
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.fft import fftshift, fft, ifft
from scipy.optimize import minimize
from scipy.linalg import svd
from scipy import signal

mpl.rcParams["image.aspect"] = "auto"

# voltage data loaded for the current interval
store = None
store_index = 0

# giant pulse voltage waveforms for each interval
result = None
result_index = 0
result_sum = None

produce_plots = False
lags = None

def align (x, y, idx):

    global lags
    if lags is None:
        lags = signal.correlation_lags(x.size, y.size, mode="full")

    correlation = signal.correlate(x, y, mode="full")
    cabs = np.abs(correlation)
    imax = np.argmax(cabs)
    maxval = np.max(correlation)
    lag = lags[imax]
    y[:] = np.roll(y, lag)
    ph = maxval / np.abs(maxval)
    y *= np.conj(ph)
    print(f"lag[{idx}] {lag=} {maxval=} {ph=}")

def add_filter_to_result (method) -> None:

    global result
    global result_index
    global result_sum

    global produce_plots

    if store is None:
        error("no stored data")

    if method == "max":
        print(f"taking the maximum of {store_index} spectra")
    else:
        print(f"computing the average of {store_index} spectra")

    subset = store[:,:store_index,:]
    nsamp = store.shape[2]

    subset = np.reshape(subset, (2*store_index,nsamp))
    if result is None:
        result = np.zeros((1, nsamp), dtype=np.complex128)
    else:
        result.resize((result_index+1, nsamp))

    sum_power = np.sum(np.abs(subset)**2,axis=1)
    sum_max_idx = np.argmax(sum_power)

    max_power = np.max(np.abs(subset)**2,axis=1)
    max_max_idx = np.argmax(max_power)

    print(f"max sum={sum_max_idx} max={max_max_idx}")

    max_idx = sum_max_idx

    x = subset[max_idx]

    if method != "max":
        if result_sum is None:
            correlate_with = x
        else:
            correlate_with = result_sum

        print(f"aligning and adding to pulse with maximum power at idx={max_idx}")

        for idx in range(subset.shape[0]):
            if result_sum is None and idx == max_idx:
                continue

            y = subset[idx]
            align (x, y, idx)
            x += y

        if result_sum is None:
            result_sum = np.copy(x)
        else:
            result_sum += x

    else:
        print(f"directly using pulse with maximum power at idx={max_idx}")

    # U, s, Vh = svd (subset)
    # print(f"singular values {s}")
    result[result_index,:] = x[:]

    if produce_plots:
        toplot = np.abs(x)**2
        toplot = np.mean(toplot.reshape(-1, 4), axis=1)
        plt.plot(toplot)

        filename = f"svd_{result_index:03}.png"
        plt.savefig(filename)
        plt.close()

    result_index += 1


first_datestr = None
first_second = 0
current_time = 0
file_index = 0

def get_current_offset (filename) -> float:

    global first_datestr
    global first_second
    global current_time 
    global file_index 

    dot = filename.find('.')
    datestr = filename[:dot]
    print(f"loading filename={filename} datestr={datestr}")
    the_time = time.strptime(datestr, "%Y-%m-%d-%H:%M:%S")
    the_second = time.mktime(the_time)

    if file_index == 0:
      first_second = the_second
      first_datestr = datestr

    remainder = filename[dot+1:]
    dot = remainder.find('.')
    frac_second = '0.' + remainder[:dot]

    time_offset = (the_second-first_second) + float(frac_second)

    if file_index == 0:
      current_time = time_offset

    current_offset = time_offset - current_time

    print(f"time={time_offset} offset={current_offset}")
    return current_offset


def create_filters (files, interval, template, method, plot) -> None:

    global store 
    global store_index
    global file_index
    global current_time
    global produce_plots

    produce_plots = plot

    for file in files:

        current_offset = get_current_offset(file)

        if current_offset > interval:
            add_filter_to_result (method)
            current_time += interval
            store_index = 0

        f = open(file, "rb")
        f.seek(4096) # skip the 4k DADA header

        data = np.fromfile(f, dtype=np.complex64)
        # print(f"{data.size} complex<float> loaded")

        data = np.reshape (data, (2, data.size//2), order='F')
        # print(f"new shape={data.shape}")

        if store is None:
            store = np.zeros((2, len(files), data.size//2), dtype=np.complex128)

        store[:,store_index,:] = data
        file_index += 1
        store_index += 1

        if produce_plots and file_index == 1:
            fig, axs = plt.subplots(1,2)

            for ipol in range(2):
                toplot = np.real(data[ipol])
                axs[ipol].plot(toplot)
                axs[ipol].set(ylabel="Re[v(t)]", xlabel="Time (sample index)")
            
            filename = "first_pulse.png"
            plt.savefig(filename)
            plt.close()

    if store_index > 0:
        print(f"adding last interval to result current_offset/interval={current_offset}/{interval}")
        add_filter_to_result(method)
    

def main() -> None:

    """Create average giant pulses that can be used as filters."""

    global result

    # do arg parsing here
    p = argparse.ArgumentParser()
    p.add_argument(
        "-interval",
        type=float,
        default=20,
        help="the integration interval in seconds",
    )
    p.add_argument(
        "-template",
        type=str,
        required=True,
        help="the archive file used as a template",
    )
    p.add_argument(
        "-method",
        type=str,
        default="max",
        help="the method used to derive the impulse response",
    )
    p.add_argument(
        "-plot",
        type=bool,
        default=False,
        help="plot each giant",
    )
    args, files = p.parse_known_args()

    vargs = vars(args)
    create_filters(files, **vargs)

    print(f"{result.shape=}")

    # convert h(t,tau) to H(t,nu)
    result = fft(result,axis=1,norm="ortho")

    # output matched filter frequency response functions
    result = np.conj(result)

    ntime, nchan = result.shape

    # normalize each time interval by the standard deviation (over nu)
    result /= np.sqrt(np.sum(np.abs(result)**2, axis=1) / nchan)[:, np.newaxis]

    with open("giant_pulse_filters.pkl", "wb") as fh:
        pickle.dump(result, fh)

    arch = psrchive.Archive_load(args.template)
    arch.resize(0)

    result = np.reshape(result,ntime*nchan)

    ext = arch.add_dynamic_response()
    ext.set_nchan(nchan)
    ext.set_ntime(ntime)
    ext.set_npol(1)
    ext.resize_data()
    ext.set_data(result)

    start_time = psrchive.MJD()
    start_time.from_datestr(first_datestr)

    end_time = start_time + args.interval * ntime

    print(f"start_time={start_time.printdays(13)} end_time={end_time.printdays(13)}")
    ext.set_minimum_epoch (start_time)
    ext.set_maximum_epoch (end_time)

    cfreq = arch.get_centre_frequency()
    bw = arch.get_bandwidth()

    ext.set_minimum_frequency (cfreq - 0.5*bw)
    ext.set_maximum_frequency (cfreq + 0.5*bw)

    arch.unload("giant_pulse_filters.fits")

if __name__ == "__main__":
  main()

