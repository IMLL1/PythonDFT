import pandas as pd
import numpy as np
from mpl_interactions import panhandler, zoom_factory
import matplotlib.pyplot as plt
import scipy.fftpack as sfft
from scipy.signal import blackman

# %matplotlib widget

df = pd.read_csv("input.csv")
x = np.asarray(df.x)
t = np.asarray(df.t)

numSamples = len(t)  # number of samples

dftObj = sfft.fftshift(sfft.fft(x))

threshold = input("Noise threshold. Any non-valid input will be treated as zero: ")
try:
    threshold = float(threshold)
    threshold = max(threshold, 0)
except ValueError:
    threshold = 0

reconstruct = input(
    "Enter Y, yes, or 1 to reconstruct the signal (and anything else to not reconstruct): "
).upper().strip() in ["1", "Y", "YES"]

dftObj[abs(dftObj) < threshold] = 0  # noise smoothing

duration = t[-1] - t[0]  # signal duration
sampleFreq = 1 / (t[1] - t[0])

## 2 sided
mags2S = np.abs(dftObj) / numSamples  # amplitude of each singal
phases2S = np.angle(dftObj) / (2 * np.pi)  # phase shift in rotations
freqs2S = sfft.fftshift(sfft.fftfreq(numSamples, t[1] - t[0]))

## 1 sided
mags1S = 2 * mags2S[numSamples // 2 :]  # amplitude of each singal
mags1S[0] = mags1S[0] / 2
phases1S = phases2S[numSamples // 2 :]  # phase shift
freqs1S = freqs2S[numSamples // 2 :]
# frequency (hz)

plt.style.use(["dark_background"])
plt.rcParams.update(
    {
        "axes.grid": True,
        "grid.color": [0.25, 0.25, 0.25],
        "lines.linewidth": 0.75,
        "lines.markersize": 3,
    }
)

figSig, axSig = plt.subplots()
if reconstruct:
    tSmooth = np.linspace(t[0], t[-1], num=2 * numSamples, endpoint=True)  # smoothed t
    xSmooth = np.sum(
        mags1S
        * np.cos(2 * np.pi * freqs1S * (tSmooth[:, None] - t[0]) + phases1S * np.pi),
        1,
    )
    plt.plot(t, x, "-y", lw=2)
    plt.plot(tSmooth, xSmooth, "-r")
    plt.legend(["Original data", "Transform"])
else:
    plt.plot(t, x, "-c")
figSig.suptitle("Amplitude vs Time")
axSig.set_xlabel("Time ($t$)")
axSig.set_ylabel("Amplitude ($x$)")

fig1S, axs1S = plt.subplots(2, 1, sharex=True)
fig1S.suptitle("One-Sided Fourier Transform")
axs1S[0].plot(freqs1S, mags1S, "-c")
axs1S[0].set_ylabel("Amplitude ($|X|$)")
axs1S[1].plot(freqs1S, phases1S, "-c")
axs1S[1].set_ylabel("Phase Angle ($rotations$)")
axs1S[1].set_xlabel("Frequency ($s^{-1}$)")
fig1S.tight_layout()

fig2S, axs2S = plt.subplots(2, 1, sharex=True)
fig2S.suptitle("Two-Sided Fourier Transform")
axs2S[0].plot(freqs2S, mags2S, "-c")
axs2S[0].set_ylabel("Amplitude ($|X|$)")
axs2S[1].plot(freqs2S, phases2S, "-c")
axs2S[1].set_ylabel("Phase Angle ($rotations$)")
axs2S[1].set_xlabel("Frequency ($s^{-1}$)")
fig2S.tight_layout()

ph = []
zf = []
for fig in [fig1S, fig2S, figSig]:
    ph.append(panhandler(fig))
    for ax in fig.axes:
        zf.append(zoom_factory(ax))
plt.show()
