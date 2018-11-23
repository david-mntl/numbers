import tkinter
import tkinter as tk
import tkinter.messagebox
import pyaudio
import wave
import os
import misc

import scipy.io.wavfile as wav
import scipy.signal as signal

import matplotlib as mpl
import matplotlib.pyplot
import numpy as np
import matplotlib.backends.tkagg as tkagg
import classifier
import keras 
from matplotlib.backends.backend_agg import FigureCanvasAgg
from keras.models import load_model

class RecAUD:

    def __init__(self, chunk=3024, frmat=pyaudio.paInt16, channels=1, rate=11025, py=pyaudio.PyAudio()):

        # Start Tkinter and set Title
        self.main = tkinter.Tk()
        self.collections = []
        self.main.geometry("920x600+500+150")
        self.main.title("Numbers prediction")
        self.CHUNK = chunk
        self.FORMAT = frmat
        self.CHANNELS = channels
        self.RATE = rate
        self.p = py
        self.frames = []
        self.st = 0
        self.stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)


        # Set Frames
        self.buttons = tkinter.Frame(self.main, padx=120, pady=20)

        self.labAudio = tkinter.Label(self.main,text="Audio prediction",font="Arial 14")
        self.labAudio.place(x=5,y=5)

        

        self.butRecord = tkinter.Button(self.main,width=11,height=1,command=lambda: self.toggleRecord(),text="Start recording",bg=misc.green,fg=misc.black)
        self.butRecord.place(x=10,y=35)

        self.butPredict = tkinter.Button(self.main,width=7,height=1,command=lambda: self.doGraphics(),text="Graph audio",bg=misc.hardBlue,fg=misc.black)
        self.butPredict.place(x=145,y=35)

        self.butPredict = tkinter.Button(self.main,width=7,height=1,command=lambda: self.doPrediction(),text="Predict",bg=misc.hardBlue,fg=misc.black)
        self.butPredict.place(x=243,y=35)

        #-----
        self.predGroup = tkinter.LabelFrame(self.main, text="Prediction", width=200,height=200)
        self.predGroup.place(x=700,y=75)

        self.labTxtPrediction = tkinter.Label(self.predGroup,text="The audio contains a:",font="Arial 14")
        self.labTxtPrediction.place(x=5,y=5)

        self.labTxtPrediction = tkinter.Label(self.predGroup,text="00",font="Arial 72")
        self.labTxtPrediction.place(x=45,y=45)

        #-----
        self.graphGroup = tkinter.LabelFrame(self.main, text="Audio graphics", width=650,height=510)
        self.graphGroup.place(x=5,y=75)

        self.canvas = tkinter.Canvas(self.graphGroup, width=640, height=480)
        self.canvas.place(x=1,y=1)

        #self.canvas = tkinter.Canvas(window, width=300, height=300)

        tkinter.mainloop()

    def toggleRecord(self):
        if( self.st == 0):
            self.butRecord.config(bg=misc.red,text="Stop recording")
            self.st = 1
            self.frames = []
            stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
            while self.st == 1:
                data = stream.read(self.CHUNK)
                self.frames.append(data)
                print("* recording")
                self.main.update()

            stream.close()

            wf = wave.open('predict.wav', 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
        else:
            self.butRecord.config(bg=misc.green,text="Start recording")
            self.st = 0

    def stopRecord(self):
        self.st = 0

    def getNumberString(self,pNumber):
        if(pNumber < 10):
            return "0" + str(pNumber)
        else:
            return str(pNumber)

    def doPrediction(self):
        #intPrediction = 15 #TODO GET THE PREDICTION FROM DAVID'S METHOD
        audio = classifier.audiofile_to_input_vector("predict.wav",13,9)
        inputAudio = audio.reshape(1,audio.shape[0],audio.shape[1])
        inputAudio = keras.preprocessing.sequence.pad_sequences(inputAudio, maxlen=200)
        model = load_model("model.h5")
        intPrediction = model.predict_classes(inputAudio, verbose = 0)
        print (intPrediction)
        txtPrediction = self.getNumberString(intPrediction[0])
        self.labTxtPrediction.config(text=txtPrediction)


    def doGraphics(self):
        self.canvas.delete("all")
        sample_rate, samples = wav.read("predict.wav")
        
        f, t, Zxx = signal.stft(samples, fs=sample_rate)
        matplotlib.pyplot.pcolormesh(t, f, np.abs(Zxx), cmap='Reds')
        matplotlib.pyplot.savefig("imgs/audio.png")

        pAudioImage = misc.load_img("audio.png")

        
        self.canvas.create_image(0, 0, image=pAudioImage, anchor='nw')
        self.canvas.image = pAudioImage

# Create an object of the ProgramGUI class to begin the program.
guiAUD = RecAUD()