import matplotlib.pyplot as plt
import numpy as np
import math as m

datafile = open('input.xlsx', 'r', encoding='utf-8')
xOriginal = np.linspace(0,1, num=9,endpoint=True)
yOriginal = np.sin(2*m.pi*(xOriginal+0.25))

if(len(xOriginal)!=len(yOriginal)):
    raise Exception("you call that a signal?")
elif(len(xOriginal)<5):
    raise Exception("x and y length not the same")
else:
    duration = xOriginal[-1]-xOriginal[0]
    fundFreq = 1/duration   # lowest possible frequency
    
    x=np.delete(xOriginal, -1)  # We're getting rid of the last because we don't need it
    y=np.delete(yOriginal, -1)

listLen = len(x)        # number of samples
freq = np.linspace(0,(listLen-1)*fundFreq, listLen)
print("FREQ:", freq)

sinAmp = [sum([y[X] * m.sin(2*m.pi*f*x[X]) for X in range(listLen)]) for f in freq]
cosAmp = [sum([y[X] * m.cos(2*m.pi*f*x[X]) for X in range(listLen)]) for f in freq]

ampTot = [m.sqrt(sinAmp[n] ** 2 + cosAmp[n] **2)*2*duration/listLen for n in range(listLen)]
phase = [m.atan2(cosAmp[n], sinAmp[n]) for n in range(listLen)]

y2 = [[ampTot[n]*m.sin(2*m.pi*freq[n]*(x+phase[n])) for n in range(listLen)] for x in x]
#y2T = [[y2[b][a] for b in range(listLen+1)] for a in range(listLen)]
y2Sum = [sum(y2[n]) for n in range(listLen)]

#print("AMP:", sinAmp)
#print("PHASE:", phase)

plt.plot(x,y)
plt.plot(x,y2Sum)
plt.show()