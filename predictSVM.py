
import tkinter
import tkinter as tk
import tkinter.messagebox
from tkinter import filedialog
import pyaudio
import wave
import os
import misc

import scipy.io.wavfile as wav
import scipy.signal as signal
import preprocesamiento as prep

import matplotlib as mpl
import matplotlib.pyplot
import numpy as np
import matplotlib.backends.tkagg as tkagg
import classifSVM
import keras 
from matplotlib.backends.backend_agg import FigureCanvasAgg
from keras.models import load_model

from sklearn.externals import joblib
from pydub import AudioSegment
from playsound import playsound

class RecAUD:

    def __init__(self, chunk=3024, frmat=pyaudio.paInt16, channels=1, rate=11025, py=pyaudio.PyAudio()):

        # Start Tkinter and set Title
        self.main = tkinter.Tk()
        self.collections = []
        self.main.geometry("920x625+500+150")
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

        self.butGraph = tkinter.Button(self.main,width=7,height=1,command=lambda: self.doGraphics(),text="Graph audio",bg=misc.hardBlue,fg=misc.black)
        self.butGraph.place(x=145,y=35)

        self.butPredict = tkinter.Button(self.main,width=7,height=1,command=lambda: self.doPrediction(),text="Predict",bg=misc.hardBlue,fg=misc.black)
        self.butPredict.place(x=243,y=35)
        
        self.butClear = tkinter.Button(self.main,width=7,height=1,command=lambda: self.cleanCanvas(),text="Clear",bg=misc.hardBlue,fg=misc.black)
        self.butClear.place(x=340,y=35)

        self.audioPathEntry = tkinter.Entry(self.main,bg="#FFFFFF", width=55)
        self.audioPathEntry.bind("<Key>", lambda e: "break")
        self.audioPathEntry.place(x=10,y=75)

        img_search = misc.load_img("dir_search.png")
        b_dir=tkinter.Button(self.main,justify = tkinter.LEFT,image=img_search,command=lambda: self.askAudioFilePath())
        b_dir.photo = img_search
        b_dir.place(x=520,y=70)
        
        self.butClear = tkinter.Button(self.main,width=7,height=1,command=lambda: self.loadNewAudio(),text="Load",bg=misc.hardBlue,fg=misc.black)
        self.butClear.place(x=560,y=70)
        

        #-----
        self.predGroup = tkinter.LabelFrame(self.main, text="Prediction", width=200,height=200)
        self.predGroup.place(x=700,y=110)

        self.labTxtPrediction = tkinter.Label(self.predGroup,text="The audio contains a:",font="Arial 14")
        self.labTxtPrediction.place(x=5,y=5)

        self.labTxtPrediction = tkinter.Label(self.predGroup,text="00",font="Arial 72")
        self.labTxtPrediction.place(x=45,y=45)

        #-----
        self.predGroup = tkinter.LabelFrame(self.main, text="Audio adjust", width=200,height=200)
        self.predGroup.place(x=700,y=325)

        self.butCrop = tkinter.Button(self.predGroup,width=7,height=1,command=lambda: self.cropAudio(),text="Crop",bg=misc.hardBlue,fg=misc.black)
        self.butCrop.place(x=5,y=5)

        self.butCrop = tkinter.Button(self.predGroup,width=7,height=1,command=lambda: self.playAudio(),text="Play",bg=misc.hardBlue,fg=misc.black)
        self.butCrop.place(x=100,y=5)

        self.initLabel = tkinter.Label(self.predGroup,text="Init time:")
        self.initLabel.place(x=5,y=45)

        self.initEntry = tkinter.Entry(self.predGroup,bg="#FFFFFF")
        self.initEntry.place(x=5,y=65)

        self.endLabel = tkinter.Label(self.predGroup,text="End time:")
        self.endLabel.place(x=5,y=95)

        self.endEntry = tkinter.Entry(self.predGroup,bg="#FFFFFF")
        self.endEntry.place(x=5,y=115)

        #-----
        self.graphGroup = tkinter.LabelFrame(self.main, text="Audio graphics", width=650,height=510)
        self.graphGroup.place(x=5,y=110)

        #self.canvas = tkinter.Canvas(window, width=300, height=300)

        tkinter.mainloop()

    def loadNewAudio(self):
        newAudio = AudioSegment.from_wav(str(self.audioPathEntry.get()))
        newAudio.export('predict.wav', format="wav") #Exports to a wav file in the current path.

    def askAudioFilePath(self):
        pPath = filedialog.askopenfilename(title = "Select file",filetypes = (("wav files","*.wav"),))
        self.audioPathEntry.delete('0', 'end')
        self.audioPathEntry.insert(0,pPath)

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
        for num in range(0, 16): # numero
            for idx in range(0, 31): # student id
                for contx in range(1, 4): # context
                    try:
                        prep.obtenerMayorNumFrame(num, idx,contx)
                    except :
                       pass #Cant find the file

        num_frames=max(prep.listaNumframes)
        audio = prep.preproceso(0,0,0,num_frames,1) 
        audio = classifSVM.reshape_Audio(audio)
        audio = audio.reshape(1,len(audio))                
        model = joblib.load('model.joblib') 
        intPrediction = model.predict(audio)
        print (intPrediction)
        txtPrediction = self.getNumberString(intPrediction[0])
        self.labTxtPrediction.config(text=txtPrediction)

    def playAudio(self):
        playsound("predict.wav")

    def cropAudio(self):
        t1 = int(self.initEntry.get())/10
        t2 = int(self.endEntry.get())/10
        newAudio = AudioSegment.from_wav("predict.wav")
        newAudio = newAudio[t1:t2]
        newAudio.export('predict.wav', format="wav") #Exports to a wav file in the current path.

    def cleanCanvas(self):
        self.canvas.delete("gp")

    def doGraphics(self):
        
        sample_rate, samples = wav.read("predict.wav")
        
        f, t, Zxx = signal.stft(samples, fs=sample_rate)
        #matplotlib.pyplot.pcolormesh(t, f, np.abs(Zxx), cmap='Reds')
        matplotlib.pyplot.clf()
        matplotlib.pyplot.plot(samples)
        matplotlib.pyplot.savefig("imgs/audio.png")

        pAudioImage = misc.load_img("audio.png")


        self.canvas = tkinter.Canvas(self.graphGroup, width=640, height=480)
        self.canvas.place(x=1,y=1)

        self.image = pAudioImage
        self.cleanCanvas()
        self.canvas.create_image(0, 0, image=self.image, anchor='nw',tags=('gp'))

# Create an object of the ProgramGUI class to begin the program.
guiAUD = RecAUD()
