from sklearn.metrics import confusion_matrix
import scipy.io.wavfile as wav
import scipy.signal as signal
from python_speech_features import mfcc
from playsound import playsound
from keras.utils import np_utils
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM
import math,time,glob,os,sys,keras
import numpy as np
from numpy import concatenate, mean
import preprocesamiento as prep
from sklearn import svm




def loadData():    

    y_target= []
    x_data = []

    for num in range(0, 16): # numero                      
        for idx in range(0, 31): # student id             
            for contx in range(1, 4): # context                
                try:
                    prep.obtenerMayorNumFrame(num, idx,contx)

                except :
                   print("Error al abrir",prep.getFileName(num,idx,contx))                   
                   # if you get here it means an error happende, maybe you should warn the user
                   # but doing pass will silently ignore it
                   pass
    num_frames=max(prep.listaNumframes)
    for num in range(0, 16): # numero                      
        for idx in range(0, 31): # student id             
            for contx in range(1, 4): # context                
                try:
                    features = prep.preproceso(num, idx,contx, num_frames,0)                    
                    y_target.append(num)                      
                    x_data.append(features)

                except :
                   print("Error al abrir",prep.getFileName(num,idx,contx))                   
                   # if you get here it means an error happende, maybe you should warn the user
                   # but doing pass will silently ignore it
                   pass 
    
    return [x_data, y_target]


# reshape audio array to vector
def reshape_Audio(audio):
    temp = np.reshape(audio,len(audio)*len(audio[0]))
    return np.array(temp)

def SVM():
    
    Data = loadData()
    x_data = np.array(Data[0])
    y_target = np.array(Data[1])  

#############################################################################
    audio_reshape =  np.zeros((len(x_data), len(x_data[0])*len(x_data[0][0]))) 

    for i in range(0,len(x_data)):
        temp =x_data[i]
        audio_reshape[i] = reshape_Audio(temp)

    audio_reshape = np.array(audio_reshape)
############################################################################

    #Start time
    time_start=time.time()

   #Define model architecture

    clf=svm.SVC(kernel=str("linear"), C=5, gamma= 0.05 )

    svc=clf.fit(audio_reshape,y_target) # train    
    label_predict=svc.predict(audio_reshape) # predict


########################################################################### 
    #Compute ConfussionMatrix 
    print(confusion_matrix(y_target, label_predict))

    # Total Time
    time_end=time.time() 
    print('Time to classify: %0.2f.' % ((time_end-time_start)/60))


    #To save the SVM Model
    from sklearn.externals import joblib
    joblib.dump(svc, 'model.joblib') 
    
    #Evaluating the model
    # accuracy
    accuracy=mean((label_predict==y_target)*1)
    #score = model.evaluate(x_data, y_target, verbose=0)  
    print("Evaluate results:")
   # print('Test loss:', score[0])
    print('Test accuracy:', accuracy)





    

#SVM()
