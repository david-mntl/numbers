
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.cm as cmap
import matplotlib.pyplot as plt
from matplotlib.backend_bases import NavigationToolbar2
from playsound import playsound
from pylab import*
import math



home = NavigationToolbar2.home

txtNumbers = ['cero','uno','dos','tres','cuatro','cinco','seis','siete','ocho','nueve','diez','once','doce','trece','catorce','quince']
global currentNumber
global currentID
global audioAverage
audioAverage = 0

currentNumber = 0
currentID = 26

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
    plotSTFT(getFileName(currentNumber,currentID,1),txtNumbers[currentNumber])

    home(self, *args, **kwargs)

def prevUser(self, *args, **kwargs):
    prevNumber()
    plotSTFT(getFileName(currentNumber,currentID,1),txtNumbers[currentNumber])

    home(self, *args, **kwargs)

NavigationToolbar2.forward = nextUser
NavigationToolbar2.back = prevUser



'''
* Returns average energy of all recorded
'''
def normalizeSignalEnergy():
    audio = 0 # energy sum of each audio
    count=0 # count numbers
    for num in range(0, 16): # numero 
        for idx in range(0, 31): # student id 
            for contx in range(1, 4): # context
                try:
                    sample_rateTemp,samplesTemp = wav.read(getFileName(num,idx,contx))
                    # Ecuation of energy, means take the average of each sample 
                    rms_val = sqrt(mean(samplesTemp**2))
                    if (math.isnan(rms_val) != True):
                        audio = audio + rms_val
                    count =count +1       
                except :
                   print("Error al abrir",getFileName(num,idx,contx))
                   # if you get here it means an error happende, maybe you should warn the user
                   # but doing pass will silently ignore it
                   pass
    
    
    print("Audios leidos")
    print(count) 
    audioAverageTemp=audio/count #
    print("Energia promedio")
    print(audioAverageTemp)
    return audioAverageTemp


#--------------------------------------------------------------------------------------

def plotSTFT(pFilename,pNumberText):
    sample_rate, samples = wav.read(pFilename)
    print("Numero de muestra")
    print(pNumberText)
    print("original")
    playsound(pFilename)
    
    # Test audio normalize
    wav.write("test.wav", sample_rate, samples /audioAverage)
    print("normalizado")
    playsound("test.wav")
    
    # plot of frencuence vs time
    f, t, Zxx = signal.stft(samples, fs=sample_rate)
    plt.pcolormesh(t, f, np.abs(Zxx), cmap='Reds')
    plt.show()

audioAverage = normalizeSignalEnergy() # Energia promedio de los audios
plotSTFT(getFileName(currentNumber,currentID,1),txtNumbers[currentNumber])
