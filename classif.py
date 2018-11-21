from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.cm as cmap
import matplotlib.pyplot as plt
from playsound import playsound
from pylab import*
import math
from python_speech_features import mfcc

import math
import time
import glob, os
import sys
import audioManager

def loadData():    

    y_target= []
    mfcc_feats = None

    for num in range(0, 16): # numero                      
        for idx in range(0, 31): # student id             
            for contx in range(1, 4): # context                
                try:
                    sample_rateTemp,samplesTemp = wav.read(audioManager.getFileName(num,idx,contx))                    
                    MFCC_ = np.mean(mfcc(samplesTemp.astype(np.float64), sample_rateTemp).T, axis=0 )                    
                    y_target.append(num)                    
                    if (mfcc_feats) is None:
                        mfcc_feats = MFCC_
                    else:
                        mfcc_feats = np.concatenate((mfcc_feats, MFCC_), axis=0)
 
                except :
                   print("Error al abrir",audioManager.getFileName(num,idx,contx))                   
                   # if you get here it means an error happende, maybe you should warn the user
                   # but doing pass will silently ignore it
                   pass               
    return [mfcc_feats, y_target]


def SVM():
    Data = loadData()        
    x_data = np.array(Data[0])
    y_target = np.array(Data[1])
    print ("here")

    # Preprocess class labels
    x_data = np.array(x_data)   
    y_target = np.array(y_target)               

    x_data = x_data.astype('float64')
    y_target = y_target.astype('float64')

#############################################################################
    print(y_target.shape)
    x_data = x_data.reshape(x_data.shape[0], 16)   #<<<------------------- here...        

    y_target = np_utils.to_categorical(y_target)  
    print(y_target.shape)
############################################################################
    # Start time
    time_start=time.time()

    # Scikit-learn SVM Classifier
    c=1.0 # SVM regularization parameter
    clf=svm.SVC( kernel='rbf', C=c , gamma= 1.0 )

    model=clf.fit(x_data,y_target) # train    <<<---------------------------

    #To save the SVM Model
    #joblib.dump(model, 'model.joblib')     

    # Total Time
    time_end=time.time() 
    print('Time to classify: %0.2f.' % ((time_end-time_start)/60))



    label_predict = predict(model , x_data)    
    #print confusion matrix
    print(confusion_matrix(x_data, label_predict))

    # accuracy
    accuracy=mean((label_predict==x_data)*1)
    print('Accuracy: %0.4f.' % accuracy)

    ##return model



def predict(model , test):
    return model.predict(test) # predict




SVM()