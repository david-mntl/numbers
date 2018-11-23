import numpy
import scipy.io.wavfile
from scipy.fftpack import dct
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from matplotlib.backend_bases import NavigationToolbar2


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


listaNumframes=[]

'''
    .
    This function has been modified from:
    https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
'''

def preproceso(num, idx,contx, num_frames, tipoAlgoritmo):
    if(tipoAlgoritmo):
        frecuencia, signal = scipy.io.wavfile.read("predict.wav")
    else:
        frecuencia, signal = scipy.io.wavfile.read(getFileName(num,idx,contx))


    pre_emphasis = 0.97
    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    frame_size = 0.025
    frame_stride = 0.01

    frame_length, frame_step = frame_size * frecuencia, frame_stride * frecuencia  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    
    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros(abs((pad_signal_length - signal_length)))
    pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]

    #print("%s - %s - %s = %s: " % (num, idx,contx,frames))
    frames *= numpy.hamming(frame_length)
    # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **



    NFFT = 512


    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum


    nfilt = 40

    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (frecuencia / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((NFFT + 1) * hz_points / frecuencia)

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB



    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13


    cep_lifter=22
    (nframes, ncoeff) = mfcc.shape
    n = numpy.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
    mfcc *= lift  #*

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

    pre_emphasis = 0.97
    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    frame_size = 0.025
    frame_stride = 0.01

    frame_length, frame_step = frame_size * frecuencia, frame_stride * frecuencia  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
    listaNumframes.extend([num_frames])
                    
    

        



