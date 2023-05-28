DFTfuncs.py is the DFT functions. There's a sft (slow fourier transform) function, as well as two FFT functions; one slightly faster (a factor of about two) than the other. This file is required for both python scripts
input.csv contains the data. It is also required, and can be edited as neede

signalAnalysis presents some fourier transform information about a singal (specifically, the signal in input.txt)
signalCompress trims a signal down to only its most dominant frequencies. The number of frequencies can be selected by the user.

imgFFT2 runs a DFT on an image, and displays the result in both log scale and absolute scale, with the option to save it
