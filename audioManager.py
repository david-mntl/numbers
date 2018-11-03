
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.cm as cmap
import matplotlib.pyplot as plt
from matplotlib.backend_bases import NavigationToolbar2
from playsound import playsound

home = NavigationToolbar2.home

txtNumbers = ['cero','uno','dos','tres','cuatro','cinco','seis','siete','ocho','nueve','diez','once','doce','trece','catorce','quince']
global currentNumber
currentNumber = 0

'''
*   Returns a string that identifies the file that is required
*   from the user.
*   @param pNumber: int, that identifies the number from 0 to 15.
*   @param pUserID: int, that identifies the user from 1 to 30.
*   @param pUserContext: int, that idenfies morning, afternoon, night from 1 to 3.
*   @return string: filename of the require file
'''
def getFileName(pNumber, pUserID, pUserContext):
    path = "data/" + txtNumbers[pNumber]    #Adds the data path and number
    path += "_" + str(pUserID)              #Adds the user id to the filename
    path += "_" + str(pUserContext)         #Adds the user context to the filename
    path += ".wav"                          #Adds the file extension
    return path

def nextNumber():
    global currentNumber
    if(currentNumber > 15):
        currentNumber = 0
    currentNumber+=1

def prevNumber():
    global currentNumber
    if(currentNumber < 1):
        currentNumber = 15
    currentNumber-=1

def nextUser(self, *args, **kwargs):
    nextNumber()
    plotSTFT(getFileName(currentNumber,26,1),txtNumbers[currentNumber])

    home(self, *args, **kwargs)

def prevUser(self, *args, **kwargs):
    prevNumber()
    plotSTFT(getFileName(currentNumber,26,1),txtNumbers[currentNumber])

    home(self, *args, **kwargs)

NavigationToolbar2.forward = nextUser
NavigationToolbar2.back = prevUser

#--------------------------------------------------------------------------------------

def plotSTFT(pFilename,pNumberText):
    sample_rate, samples = wav.read(pFilename)
    playsound(pFilename)

    print(pNumberText)

    f, t, Zxx = signal.stft(samples, fs=sample_rate)
    plt.pcolormesh(t, f, np.abs(Zxx), cmap='Reds')
    plt.show()

plotSTFT(getFileName(currentNumber,26,1),txtNumbers[currentNumber])