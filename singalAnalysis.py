import matplotlib.pyplot as plt
import math as m
import pandas as pd
import numpy as np
from DFTfuncs import dft1d

df = pd.read_csv("input.csv")
if df.isnull().any(axis=None):
    raise Exception("your signal is funky (NaNs or x and y aren't the same length)")
elif len(df) < 5:
    raise Exception(
        "You call that a signal? That signal is shorter than Yoda in a limbo competition"
    )
if len(df) % 2:
    df = df[:-1]  # cut last element if odd

dftObj = dft1d(df.Y)
dftObj[abs(dftObj) < 1e-6] = 0  # noise smoothing

duration = df.X.iloc[-1] - df.X.iloc[0]  # signal duration
numSamples = len(df.Y)  # number of samples
sampleFreq = 1 / (df.X.iloc[1] - df.X.iloc[0])

## 2 sided
mags2Sided = np.abs(dftObj / numSamples)  # amplitude of each singal
phases2Sided = np.angle(dftObj)  # phase shift
freqs2Sided = np.arange(-numSamples // 2, numSamples // 2) / numSamples * sampleFreq

# create a DFT object that's the first half the original one, times 2. Goes from 2-sided to 1-sided transform
# dftObjHalf = dftObj[: len(df.Y) // 2] * 2
# dftObjHalf[0] = dftObjHalf[0] / 2

## 1 sided
mags1Sided = 2 * mags2Sided[: numSamples // 2]  # amplitude of each singal
mags1Sided[0] = mags1Sided[0] / 2
phases1Sided = phases2Sided[: numSamples // 2]  # phase shift
freqs1Sided = np.arange(numSamples // 2) / duration
# frequency (hz)

xSmooth = np.linspace(df.X.iloc[0], df.X.iloc[-1], num=500, endpoint=True)  # smoothed X
ySmooth = np.empty(500)  # empty smoothed y, to fill
for idx, xCord in enumerate(xSmooth):  # fill in the rows
    ySmooth[idx] = np.sum(
        mags1Sided * np.cos(m.tau * freqs1Sided * (xCord - df.X.iloc[0]) + phases1Sided)
    )

plt.style.use(["dark_background"])
plt.rcParams.update(
    {
        "axes.grid": True,
        "grid.color": [0.25, 0.25, 0.25],
        "lines.linewidth": 1,
        "lines.markersize": 3,
    }
)

plt.figure("Reconstruction")
plt.plot(xSmooth, ySmooth, "-r")
plt.plot(df.X, df.Y, "oy",mfc='none', mew=0.5)
plt.legend(["Transform", "Original data"])
plt.title("Amplitude vs Time")
plt.xlabel("Time")
plt.ylabel("Amplitude")

fig, axs = plt.subplots(2, 1, sharex=True)
fig.suptitle("One-Sided Fourier Transform")
axs[0].plot(freqs1Sided, mags1Sided, "-oc")
axs[0].set_ylabel("Amplitude")
axs[1].plot(freqs1Sided, phases1Sided, "-oc")
axs[1].set_ylabel("Phase Angle (rad)")
axs[1].set_xlabel("Frequency (hz)")
fig.tight_layout()

fig, axs = plt.subplots(2, 1, sharex=True)
fig.suptitle("Two-Sided Fourier Transform")
axs[0].plot(freqs2Sided, mags2Sided, "-oc")
axs[0].set_ylabel("Amplitude")
axs[1].plot(freqs2Sided, phases2Sided, "-oc")
axs[1].set_ylabel("Phase Angle (rad)")
axs[1].set_xlabel("Frequency (hz)")
fig.tight_layout()
plt.show()