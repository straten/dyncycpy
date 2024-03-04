import numpy as np
from scipy.fft import fft, fftshift, ifft, irfft, rfft
import matplotlib.pyplot as plt
import psrchive
import math
import sys

SMALL_SIZE = 16
MEDIUM_SIZE = 22
BIGGER_SIZE = 32

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

inputArgs = sys.argv
filename = inputArgs[1]
ar = psrchive.Archive_load(filename)
ar.remove_baseline()
data = ar.get_data()

print(f"size={data.shape}")

X=data[0,0,0,:]
Y=data[0,1,0,:]
I=X+Y

nbin=I.size

fftI=fft(I)
fftI=fftI[1:]
dbI = np.log10(np.abs(fftI))*10

maxval = np.max(dbI)
dbI -= maxval

maxharm=150

fig, axs = plt.subplots(3)

phase = np.linspace(0,1,nbin)
axs[0].plot(phase,I, 'b')
axs[0].set(ylabel="Intensity", xlabel="Phase (turns)")

axs[1].plot(dbI[:maxharm], 'b')
axs[1].set(ylabel="Power [dB]", xlabel="Harmonic Number")

minval = dbI[80]
spectrum = np.random.normal(minval,1,100)
f0=0.641928222127829 # kHz
freq = np.linspace(0,10*f0,100)

spectrum[range(10,99,10)] = dbI[range(1,70,8)]

axs[2].plot(freq, spectrum, 'b')
axs[2].set(ylabel="Power [dB]", xlabel="Frequency (kHz)")

plt.show()

