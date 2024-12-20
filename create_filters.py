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

# average giant pulse voltage waveforms for each interval
result = None
result_index = 0
result_sum = None

produce_plots = False

def add_filter_to_result () -> None:

    global result
    global result_index
    global result_sum

    global produce_plots

    if store is None:
        error("no stored data")

    print(f"computing the average of {store_index} spectra")

    subset = store[:,:store_index,:]
    nsamp = store.shape[2]

    subset = np.reshape(subset, (2*store_index,nsamp))
    if result is None:
        result = np.zeros((1, nsamp), dtype=np.complex128)
    else:
        result.resize((result_index+1, nsamp))

    power = np.sum(np.abs(subset)**2,axis=1)
    max_idx = np.argmax(power)

    x = subset[max_idx]

    if result_sum is None:
        correlate_with = x
    else:
        correlate_with = result_sum

    for idx in range(subset.shape[0]):
        if result_sum is None and idx == max_idx:
            continue

        y = subset[idx]
        correlation = signal.correlate(x, y, mode="full")
        lags = signal.correlation_lags(x.size, y.size, mode="full")
        maxval = np.max(correlation)

        lag = lags[np.argmax(correlation)]
        y = np.roll(y, lag)
        ph = maxval / np.abs(maxval)
        y *= np.conj(ph)
        x += y
        print(f"lag[{idx}]={lag}={maxval}")

    if result_sum is None:
        result_sum = np.copy(x)
    else:
        result_sum += x

    # U, s, Vh = svd (subset)
    # print(f"singular values {s}")
    result[result_index,:] = x[:]

    if produce_plots:
        fig, axs = plt.subplots(1,2)
        toplot = np.copy(np.real(result[result_index,:]))
        axs[0].plot(toplot)
        axs[0].set(ylabel="Re[v(t)]", xlabel="Time (sample index)")
        toplot = np.copy(np.imag(result[result_index,:]))
        axs[1].plot(toplot)
        axs[1].set(ylabel="Im[v(t)]", xlabel="Time (sample index)")

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


def create_filters (files, interval, template, plot) -> None:

    global store 
    global store_index
    global file_index
    global current_time
    global produce_plots

    produce_plots = plot

    for file in files:

        current_offset = get_current_offset(file)

        if current_offset > interval:
            add_filter_to_result ()
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
        add_filter_to_result()
    

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
        "-plot",
        type=bool,
        default=False,
        help="plot each giant",
    )
    args, files = p.parse_known_args()

    vargs = vars(args)
    create_filters(files, **vargs)

    print(f"{result.shape=}")

    with open("giant_pulse_filters.pkl", "wb") as fh:
        pickle.dump(result, fh)

    arch = psrchive.Archive_load(args.template)
    arch.resize(0)

    ntime = result.shape[0]
    nchan = result.shape[1]
    result = np.reshape(result,ntime*nchan)

    ext = arch.add_dynamic_response()
    ext.set_nchan(nchan)
    ext.set_ntime(ntime)
    ext.set_npol(1)
    ext.resize_data()
    ext.set_data(result)

    start_time = psrchive.MJD(first_datestr)
    end_time = start_time + args.interval * ntime
    ext.set_minimum_epoch (start_time)
    ext.set_maximum_epoch (start_time)

    cfreq = arch.get_centre_frequency()
    bw = arch.get_bandwidth()

    ext.set_minimum_frequency (cfreq - 0.5*bw)
    ext.set_maximum_frequency (cfreq + 0.5*bw)

    arch.unload("giant_pulse_filters.fits")

if __name__ == "__main__":
  main()

