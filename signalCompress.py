import matplotlib.pyplot as plt
import math as m
import pandas as pd
import numpy as np
import numpy.ma as ma
from DFTfuncs import fftPrimesFaster

df = pd.read_csv("input.csv")
if df.isnull().any(axis=None):
    raise Exception("your signal is funky (NaNs or x and y aren't the same length)")
elif len(df) < 5:
    raise Exception(
        "You call that a signal? That signal is shorter than Yoda in a limbo competition"
    )
if len(df) % 2:
    df = df[:-1]

dftObj = fftPrimesFaster(df.x)

# create a DFT object that's the first half the original one, times 2. Goes from 2-sided to 1-sided transform
dftObjHalf = dftObj[: len(df.x) // 2] * 2
dftObjHalf[0] = dftObjHalf[0] / 2
dftObjHalf[abs(dftObjHalf) < 1e-10] = 0  # noise smoothing

duration = df.t.iloc[-1] - df.t.iloc[0]  # signal duration
numSamples = len(df.x)  # number of samples

mags = ma.abs(dftObjHalf / numSamples)  # amplitude of each singal
phases = ma.angle(dftObjHalf)  # phase shift
# present frequencies (hz)
freqs = (numSamples / duration) * ma.arange(0, numSamples // 2) / numSamples


numBest = int(
    input(
        "How many waves should we keep (must be an integer less than half of the total number of elements)? "
    )
)
while numBest >= numSamples // 2:
    numBest = int(input("Too many waves. Try again: "))

mags[0] = ma.masked  # mask the 0 frequency
# mask the magnitudes of the biggest
for idx in range(numBest):
    mags[np.argmax(mags)] = ma.masked
ditchList = ~mags.mask
mags.mask = freqs.mask = phases.mask = ditchList

## SMOOTHED REPRODUCTION
tSmooth = np.linspace(df.t.iloc[0], df.t.iloc[-1], num=500, endpoint=True)  # smoothed X
xCompSmooth = np.empty(500)  # empty smoothed y (compressed and optimal), to fill
xSmooth = np.empty(500)
for idx, xCord in enumerate(tSmooth):  # fill in the rows
    cosine = np.cos(m.tau * freqs.data * (xCord - df.t.iloc[0]) + phases.data)
    xCompSmooth[idx] = ma.dot(mags, cosine)
    xSmooth[idx] = ma.dot(mags.data, cosine)

## SIZE SAVINGS
originalSize = numSamples * 2  # every x and y
dftSize = (numSamples // 2) * 3  # n/2 mags, n/2 phases, and n/2 freqs
newDftSize = 3 * (numBest + 1)  # numbest plus the zero entry. mags, phases, and freqs

print("Original dataset size\t\t", originalSize, "\tfloats")
print("DFT size\t\t\t", dftSize, "\tfloats")
print("Compressed DFT size\t\t", newDftSize, "\tfloats")
print("Your savings:\t\t\t", round(100 * (1 - newDftSize / originalSize), 5), "\t%")

## PLOTS
plt.style.use(["dark_background"])
plt.rcParams.update(
    {
        "axes.grid": True,
        "grid.color": [0.25, 0.25, 0.25],
        "lines.linewidth": 0.5,
        "lines.markersize": 3,
    }
)

plt.figure("Reconstruction")
plt.plot(df.t, df.x, "-y", lw=2)
plt.plot(tSmooth, xSmooth, "--r", lw=1)
plt.plot(tSmooth, xCompSmooth, "-c", lw=0.5)
plt.legend(["Original data", "Optimal Transform", "Compressed Transform"])
plt.title("Amplitude vs Time")
plt.xlabel("Time (t)")
plt.ylabel("Amplitude (x)")

fig, axs = plt.subplots(2, 1, sharex=True)
fig.suptitle("One-Sided Fourier Transform")
axs[0].plot(freqs.data, mags.data, "-or")
axs[0].plot(freqs, mags, "oc", ms=5)
axs[0].set_ylabel("Amplitude (X)")
axs[1].plot(freqs.data, phases.data, "-or")
axs[1].plot(freqs, phases, "oc", ms=5)
axs[1].set_ylabel("Phase Angle (rad)")
axs[1].set_xlabel("Frequency (hz)")
fig.tight_layout()
plt.show()
