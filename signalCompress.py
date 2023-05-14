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
    df = df[:-1]

dftObj = dft1d(df.Y)

# create a DFT object that's the first half the original one, times 2. Goes from 2-sided to 1-sided transform
dftObjHalf = dftObj[: len(df.Y) // 2] * 2
dftObjHalf[0] = dftObjHalf[0] / 2

duration = df.X.iloc[-1] - df.X.iloc[0]  # signal duration
numSamples = len(df.Y)  # number of samples

magnitudes = np.abs(dftObjHalf / numSamples)  # amplitude of each singal

numBest = int(input("How many waves should we keep (must be an integer less than half of the total number of elements)? "))
while numBest >= numSamples//2:
    numBest = int(input("Too many waves. Try again: "))

# initialize array of how many waves to keep. Always keep the constant term.
idxKeep = np.zeros(numSamples//2)
idxKeep[0] = 1

#set idxKeep to 1 wherever we find a max, and set that max to 0 so it's not found again
for idx in range(numBest):
    idxKeep[np.argmax(magnitudes[1:])+1]=1  # keep the found signal
    magnitudes[np.argmax(magnitudes[1:])+1]=0   # and make sure it's not found again

dftObjHalf[idxKeep==0] = 0  # set all the non-found ones to zero

magnitudes = np.abs(dftObjHalf / numSamples)  # re-establish
phaseOffsets = np.angle(dftObjHalf)  # phase shift
# present frequencies (hz)
frequencies = (numSamples / duration) * np.arange(0, numSamples // 2) / numSamples

## SMOOTHED REPRODUCTION

xSmooth = np.linspace(df.X.iloc[0], df.X.iloc[-1], num=500, endpoint=True)  # smoothed X
ySmooth = np.empty(500) # empty smoothed y, to fill
for idx, xCord in enumerate(xSmooth):  # fill in the rows
    ySmooth[idx] = np.sum(
        magnitudes * np.cos(m.tau * frequencies * (xCord-df.X.iloc[0]) - phaseOffsets)
    )


## PLOTS
plt.rcParams["axes.grid"] = True
plt.rcParams["lines.linewidth"] = 1
plt.rcParams["lines.markersize"] = 3
plt.style.use(["dark_background"])

plt.figure("Reconstruction")
plt.plot(df.X, df.Y, "og")
plt.plot(xSmooth, ySmooth.transpose()[:][:], "-r")
plt.legend(["Original data", "Smoothed Transform"])
plt.title("Amplitude vs Time")
plt.xlabel("time")
plt.ylabel("amplitude")

fig, axs = plt.subplots(2, 1, sharex=True)
fig.suptitle("Fourier Transform")
axs[0].plot(frequencies, magnitudes, "--oc")
axs[0].set_ylabel("Amplitude")
axs[1].plot(frequencies, phaseOffsets, "--oc")
axs[1].set_ylabel("Phase Angle (rad)")
axs[1].set_xlabel("Frequency (hz)")
plt.show()