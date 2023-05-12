import matplotlib.pyplot as plt
import numpy as np
import math as m
import csv

readFile = list(csv.reader(open('input.csv', 'r'))) # read the file
readFile = readFile[1:] # get rid of the header
x = [float(entry[0]) for entry in readFile] # read x
y = [float(entry[1]) for entry in readFile] # read y

if(len(x)!=len(y)):
    raise Exception("you call that a signal?")
elif(len(x)<5):
    raise Exception("x and y length not the same")

duration = x[-1]-x[0]       # signal duration
numSamples = len(x)         # number of samples
nList = range(1,numSamples) # frequency list

sinAmp = [sum([y[X] * m.sin(2*m.pi*n*x[X] / duration) for X in range(numSamples-1)]) for n in nList]   # list of riemann sums for coefficient of sine. Note: not scaled by duration or number of samples
cosAmp = [sum([y[X] * m.cos(2*m.pi*n*x[X] / duration) for X in range(numSamples-1)]) for n in nList]   # same thing but for cosine

phase = [m.atan2(cosAmp[n-1],sinAmp[n-1]) for n in nList]                          # phase shift
ampTot = [m.sqrt(sinAmp[n-1] ** 2 + cosAmp[n-1] **2)/(numSamples-1) for n in nList] # scaled amplitude

y2List = [[ampTot[n-1]*m.sin(phase[n-1]+2*m.pi*n*x/duration) for n in nList] for x in x]    # 2d list. One dimension is x, the other direction is frequency
y2 = [sum(y2List[X]) for X in range(numSamples)]                                            # reconstructed y

plt.plot(x,y, 'ok')
plt.plot(x[:-1],y2[:-1], ':r')
plt.legend(["Original data","Reconstruction"])
plt.title("Amplitude vs Time")
plt.xlabel("time")
plt.ylabel("amplitude")
plt.figure()

plt.plot(nList,ampTot)
plt.title("Amplitude-Frequency Domain")
plt.xlabel("Frequency (hz)")
plt.ylabel("Amplitude")
plt.figure()

plt.plot(nList,phase)
plt.title("Phase-Frequency Domain")
plt.xlabel("Frequency (hz)")
plt.ylabel("Phase Angle (rad)")
plt.show()