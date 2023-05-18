import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from DFTfuncs import fftPrimesFaster

df = pd.read_csv("input.csv")
if df.isnull().any(axis=None):
    raise Exception("your signal is funky (NaNs or x and y aren't the same length)")
elif len(df) < 5:
    raise Exception(
        "You call that a signal? That signal is shorter than Yoda in a limbo competition"
    )

if len(df) % 2:
    df = df[:-1]  # cut last element if odd

dftObj = fftPrimesFaster(df.x)
dftObj[abs(dftObj) < 1e-6] = 0  # noise smoothing

duration = df.t.iloc[-1] - df.t.iloc[0]  # signal duration
numSamples = len(df.t)  # number of samples
sampleFreq = 1 / (df.t.iloc[1] - df.t.iloc[0])

## 2 sided
mags2Sided = np.abs(dftObj / numSamples)  # amplitude of each singal
phases2Sided = np.angle(dftObj)  # phase shift
freqs2Sided = np.arange(-numSamples // 2, numSamples // 2) / numSamples * sampleFreq

## 1 sided
mags1Sided = 2 * mags2Sided[: numSamples // 2]  # amplitude of each singal
mags1Sided[0] = mags1Sided[0] / 2
phases1Sided = phases2Sided[: numSamples // 2]  # phase shift
freqs1Sided = np.arange(numSamples // 2) / duration
# frequency (hz)

tSmooth = np.linspace(df.t.iloc[0], df.t.iloc[-1], num=500, endpoint=True)  # smoothed X
xSmooth = np.empty(500)  # empty smoothed y, to fill
for idx, tCord in enumerate(tSmooth):  # fill in the rows
    xSmooth[idx] = np.sum(
        mags1Sided
        * np.cos(2 * np.pi * freqs1Sided * (tCord - df.t.iloc[0]) + phases1Sided)
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
plt.plot(df.t, df.x, "-y", lw=2)
plt.plot(tSmooth, xSmooth, "-r")
plt.legend(["Original data", "Transform"])
plt.title("Amplitude vs Time")
plt.xlabel("Time ($t$)")
plt.ylabel("Amplitude ($x$)")

fig, axs = plt.subplots(2, 1, sharex=True)
fig.suptitle("One-Sided Fourier Transform")
axs[0].plot(freqs1Sided, mags1Sided, "-c")
axs[0].set_ylabel("Amplitude ($|X|$)")
axs[1].plot(freqs1Sided, phases1Sided, "-c")
axs[1].set_ylabel("Phase Angle ($rad$)")
axs[1].set_xlabel("Frequency ($s^{-1}$)")
fig.tight_layout()

fig, axs = plt.subplots(2, 1, sharex=True)
fig.suptitle("Two-Sided Fourier Transform")
axs[0].plot(freqs2Sided, mags2Sided, "-c")
axs[0].set_ylabel("Amplitude ($|X|$)")
axs[1].plot(freqs2Sided, phases2Sided, "-c")
axs[1].set_ylabel("Phase Angle ($rad$)")
axs[1].set_xlabel("Frequency ($s^{-1}$)")
fig.tight_layout()
plt.show()
