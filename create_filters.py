#!/usr/bin/env python
# coding: utf-8

'''Generates average impulse responses that can be used as giant pulse detection filters,
   as described as "the first cycle" in Section 4 of Mahajan & van Kerkwijk (2023)'''

import argparse
import time
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fftshift, fft, ifft
from scipy.optimize import minimize
from scipy.linalg import svd

mpl.rcParams["image.aspect"] = "auto"

# voltage data loaded for the current interval
store = None
store_index = 0

# result of performing SVD on voltage data for the current interval
# produced during add_filter_to_result
result = None
result_index = 0

produce_plots = False

def add_filter_to_result () -> None:

  global result
  global result_index

  if store is None:
      error("no stored data")

  print(f"performing SVD on {store_index} spectra")

  subset = store[:,:store_index,:]
  nsamp = store.shape[2]

  subset = np.reshape(subset, (2*store_index,nsamp))
  if result is None:
      result = np.zeros((1, nsamp), dtype=np.complex128)
  else:
      result.resize((result_index+1, nsamp))

  U, s, Vh = svd (subset)
  print(f"singular values {s}")
  result[result_index,:] = Vh[0,:]

  if produce_plots:
      fig, axs = plt.subplots(1,2)
      toplot = np.copy(np.real(result[result_index,:]))
      axs[0].plot(toplot)
      axs[0].set(ylabel="Re[v(t)]", xlabel="Time (sample index)")
      toplot = np.copy(np.imag(result[result_index,:]))
      axs[1].plot(toplot)
      axs[1].set(ylabel="Im[v(t)]", xlabel="Time (sample index)")

first_second = 0
current_time = 0
file_index = 0

def get_current_offset (filename) -> float:

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

    remainder = filename[dot+1:]
    dot = remainder.find('.')
    frac_second = '0.' + remainder[:dot]

    time_offset = (the_second-first_second) + float(frac_second)

    if file_index == 0:
      current_time = time_offset

    current_offset = time_offset - current_time

    print(f"time={time_offset} offset={current_offset}")
    return current_offset


def create_filters (files, interval) -> None:

  global result_index
  global store 
  global store_index
  global file_index
  global current_time

  for file in files:

    current_offset = get_current_offset(file)

    if current_offset > interval:
        add_filter_to_result ()
        current_time += interval
        store_index = 0
        result_index += 1

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
    """Plot the cyclic spectrum in four different ways."""
    import argparse

    # do arg parsing here
    p = argparse.ArgumentParser()
    p.add_argument(
        "--interval",
        type=float,
        default=20,
        help="the integration interval in seconds",
    )

    args, files = p.parse_known_args()

    vargs = vars(args)
    create_filters(files, **vargs)

    with open("giant_pulse_filters.pkl", "wb") as fh:
        pickle.dump(result, fh)

if __name__ == "__main__":
  main()

