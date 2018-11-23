import numpy
import scipy.io.wavfile
from scipy.fftpack import dct
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from matplotlib.backend_bases import NavigationToolbar2

home = NavigationToolbar2.home
txtNumbers = ['cero','uno','dos','tres','cuatro','cinco','seis','siete','ocho','nueve','diez','once','doce','trece','catorce','quince']
listaNumframes=[]



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
    Function to extract the file's characteristics.
    This process (MFCC) is the same as the steps in "4 Extracción de características".
    So, it'll be use this technique.
    This function has been modified from:
    https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
'''

def preproceso(num, idx,contx, num_frames, tipoAlgoritmo):
    if(tipoAlgoritmo):
        frecuencia, signal = scipy.io.wavfile.read("predict.wav")
    else:
        frecuencia, signal = scipy.io.wavfile.read(getFileName(num,idx,contx))

    #Etapa de pre-enfasis
    filtroPreEnfasis = 0.97
    signalPosFiltroPreEnfasis = numpy.append(signal[0], signal[1:] - filtroPreEnfasis * signal[:-1])

    #Etapa de framing
    tamMuestras = 0.025
    traslapeMuestras = 0.01
    largoFrame, pasoFrame = tamMuestras * frecuencia, traslapeMuestras * frecuencia 
    largoFrame = int(round(largoFrame))
    pasoFrame = int(round(pasoFrame))
    pad_signal_length = num_frames * pasoFrame + largoFrame
    zeros = numpy.zeros(abs((pad_signal_length - len(signalPosFiltroPreEnfasis))))
    pad_signal = numpy.append(signalPosFiltroPreEnfasis, zeros)
    indices = numpy.tile(numpy.arange(0, largoFrame), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * pasoFrame, pasoFrame), (largoFrame, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]

    #Etapa de Window
    frames *= numpy.hamming(largoFrame)

    #Etapa de transformada de Fourier
    NFFT = 512
    FFTresult = numpy.absolute(numpy.fft.rfft(frames, NFFT))
    energia = ((1.0 / NFFT) * ((FFTresult) ** 2))

    #Filtros
    nfilt = 40
    low_freq_mel = 0
    # Fórmula Hz -> Mel
    high_freq_mel = (2595 * numpy.log10(1 + (frecuencia / 2) / 700))
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    # Fórmula Mel -> Hz
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin = numpy.floor((NFFT + 1) * hz_points / frecuencia)

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = numpy.dot(energia, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)
    filter_banks = 20 * numpy.log10(filter_banks)

    #Etapa de MFCCs
    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] #Se dejan los coeficientes 2 al 13
    cep_lifter=22
    (nframes, ncoeff) = mfcc.shape
    n = numpy.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
    mfcc *= lift

    #Etapa de normalización final 
    filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
    mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)
    return mfcc


'''
    Count the file's frames and append them to a list. With this list we can get the longest "file".
    This function has been modified from:
    https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
'''
def obtenerMayorNumFrame(num, idx,contx):

    frecuencia, signal = scipy.io.wavfile.read(getFileName(num,idx,contx))
    #Etapa de pre-enfasis
    filtroPreEnfasis = 0.97
    signalPosFiltroPreEnfasis = numpy.append(signal[0], signal[1:] - filtroPreEnfasis * signal[:-1])

    #Etapa de framing
    
    tamMuestras = 0.025
    traslapeMuestras = 0.01

    largoFrame, pasoFrame = tamMuestras * frecuencia, traslapeMuestras * frecuencia 
    largoFrame = int(round(largoFrame))
    pasoFrame = int(round(pasoFrame))
    num_frames = int(numpy.ceil(float(numpy.abs(len(signalPosFiltroPreEnfasis) - largoFrame)) / pasoFrame))
    listaNumframes.extend([num_frames])
                    
    

        



