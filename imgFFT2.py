import numpy as np
from PIL import Image
import scipy.fftpack as sfft


def fft(data):
    # fft = sfft.fftshift(sfft.fft2(data, axes=(0, 1)), axes=(0, 1))
    fft = sfft.fftshift(sfft.fft2(data, axes=(0, 1)), axes=(0, 1))
    return fft  # scaled between 0 and 1


def normLog(data):
    # out = data / 50
    out = 255 * np.log(1 + abs(data)) / np.log(1 + data.max())
    return out


def norm(data):
    out = data - data.min()
    out *= 255 / out.max()
    return out


while True:
    fileName = input("file name (with extension): ")
    try:
        imgInput = Image.open(fileName).convert("RGB")
        break
    except FileNotFoundError:
        pass

fftArray = abs(fft(np.array(imgInput)))
# each "layer" is the same as image size, with num channels = num layers

fftArray = norm(fftArray)

imgLog = Image.new("RGB", fftArray.shape[1::-1])  # imgOut the same size as the FFT
pixelsLog = abs(normLog(250 * fftArray)).reshape(-1, 3).astype(int)
# array where each element is a 3 item list
imgLog.putdata([tuple(pixel) for pixel in pixelsLog])
imgLog.show()

imgLin = Image.new("RGB", fftArray.shape[1::-1])
pixelsLin = abs(fftArray.reshape(-1, 3)).astype(int)
imgLin.putdata([tuple(pixel) for pixel in pixelsLin])
imgLin.show()

saveFile = input("Save file? (Y, YES, or 1): ").upper().strip()
if saveFile in ["Y", "YES", "1"]:
    imgLin.save("ZZoutputLinear.png")
    imgLog.save("ZZoutputLog.png")
