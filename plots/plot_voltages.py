import argparse
import matplotlib.pyplot as plt
import numpy as np

def load_data(file):

    f = open(file, "rb")
    f.seek(4096) # skip the 4k DADA header

    data = np.fromfile(f, dtype=np.complex64)
    data = np.reshape (data, (2, data.size//2), order='F')
    return data

def plot_voltage(data):

    fig, axs = plt.subplots(2,2, figsize=(16, 9))

    for ipol in range(2):
        toplot = np.real(data[ipol])
        axs[0,ipol].plot(toplot)
        toplot = np.imag(data[ipol])
        axs[1,ipol].plot(toplot)

        if ipol == 0:
          axs[0,ipol].set(ylabel="Re[v(t)]")
          axs[1,ipol].set(ylabel="Im[v(t)]")

        axs[0,ipol].set_title(f'Polarization {ipol}')
        axs[1,ipol].set(xlabel="Time (sample index)") 

    filename = "voltage_data.png"
    plt.savefig(filename)
    plt.close()

def plot_intensity(data):

    intensity = np.sum(np.abs(data)**2, axis=0)

    fig, axs = plt.subplots(figsize=(16, 9))

    toplot = np.real(intensity)
    axs.plot(toplot)
    axs.set(ylabel="I(t)")
    axs.set_title('Total Intensity')
    axs.set(xlabel="Time (sample index)")

    filename = "intensity_data.png"
    plt.savefig(filename)
    plt.close()

def main() -> None:

    """Plot voltage waveforms stored in DADA files."""

    global result

    # do arg parsing here
    p = argparse.ArgumentParser()
    p.add_argument(
        "-intensity",
        type=bool,
        default=False,
        help="plot total intensity",
    )

    args, files = p.parse_known_args()

    data = load_data(files[0])

    if args.intensity:
      plot_intensity(data)
    else:
      plot_voltage(data)

if __name__ == "__main__":
  main()
