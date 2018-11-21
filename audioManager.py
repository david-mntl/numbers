
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
frequencyAverage = 0
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

'''
* Returns average energy of all recorded
'''
def normalizeFrequency():

    # plot of frencuence vs time
    
    
    frequency = 0 # frequency sum of each audio
    count=0 # count numbers
    for num in range(0, 16): # numero 
        for idx in range(0, 31): # student id 
            for contx in range(1, 4): # context
                try:
                    sample_rateTemp,samplesTemp = wav.read(getFileName(num,idx,contx))
                    #Implementa fourier para pasar a un dominio tiempo frecuencia de cada ejemplo 
                    f, t, Zxx = signal.stft(samplesTemp, fs=sample_rateTemp)
                    # Frequency, means take the average of each sample                     
                    frequency = frequency + max(f)
                    count =count +1       
                except :
                   print("Error al abrir",getFileName(num,idx,contx))
                   # if you get here it means an error happende, maybe you should warn the user
                   # but doing pass will silently ignore it
                   pass
    
    
    print("Audios leidos")
    print(count) 
    frequencyAverageTemp=frequency/count #
    print("Frecuencia Pico mas alto promedio")
    print(frequencyAverageTemp)
    return frequencyAverageTemp


#--------------------------------------------------------------------------------------

def plotSTFT(pFilename,pNumberText):
    sample_rate, samples = wav.read(pFilename)
    print("Numero de muestra")
    print(pNumberText)
    print("original")
    playsound(pFilename)
    # Test audio normalize
    wav.write("test.wav", sample_rate, (samples /audioAverage))
    print("normalizado")
    playsound("test.wav")

  
    # plot of frencuence vs time
    f, t, Zxx = signal.stft(samples, fs=sample_rate)

    #Imprime las frecuencias normalizadas
    print("Frecuencias normalizadas",f/frequencyAverage)
    
    plt.pcolormesh(t, f, np.abs(Zxx), cmap='Reds')
    plt.show()
    
print("Normalizando Energia")
audioAverage = normalizeSignalEnergy() # Average Energy of the records
print("Normalizando Frecuencia")
frequencyAverage = normalizeFrequency()# Average Frequency of the records
plotSTFT(getFileName(currentNumber,currentID,1),txtNumbers[currentNumber])
