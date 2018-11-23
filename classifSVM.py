import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt
import math,time,glob,os,sys,keras
import numpy as np
import preprocesamiento as prep
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from numpy import concatenate, mean
from python_speech_features import mfcc
from playsound import playsound
from keras.utils import np_utils


'''
*   @brief: returns the set of data normalized
*   returns Data Normalized
'''
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



'''
*   @brief: reshape audio array to vector
*   returns audio sample arranged
'''
def reshape_Audio(audio):
    temp = np.reshape(audio,len(audio)*len(audio[0]))
    return np.array(temp)

'''
*   @brief: SVM classifier, trains the model and compute confusion matrix properly
*   returns 
'''
def SVM():
    
    Data = loadData() # Load data normalized
    x_data = np.array(Data[0])
    y_target = np.array(Data[1])  

#############################################################################
#               Re-arrange data to fit into the classifier                  #
#############################################################################
    audio_reshape =  np.zeros((len(x_data), len(x_data[0])*len(x_data[0][0]))) 

    for i in range(0,len(x_data)):
        temp =x_data[i]
        audio_reshape[i] = reshape_Audio(temp)

    audio_reshape = np.array(audio_reshape)
############################################################################

    #Start time
    time_start=time.time()

    #Split data into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(audio_reshape, y_target)

    #Define model architecture
    clf=svm.SVC(kernel=str("linear"), C=5, gamma= 0.05 )

    svc=clf.fit(x_train,y_train) # train    
    label_predict=svc.predict(x_test) # predict
########################################################################### 
    
    #Compute ConfussionMatrix 
    confMatrix =confusion_matrix(y_test, label_predict)
    print(confMatrix)

    # Total Time
    time_end=time.time() 
    print('Time to classify: %0.2f.' % ((time_end-time_start)/60))

    #To save the SVM Model
    from sklearn.externals import joblib
    joblib.dump(svc, 'model.joblib') 
    
    #Evaluating the model
    # accuracy
    accuracy=mean((label_predict==y_test)*1)
    #score = model.evaluate(x_data, y_target, verbose=0)  
    print("Evaluate results:")
   # print('Test loss:', score[0])
    print('Test accuracy:', accuracy)
    plotConfMatrix(confMatrix)




#############################################################################
#                         Plot Confusion Matrix                             #
#############################################################################
def plotConfMatrix(confMatrix):
    fig, ax = plt.subplots()
    ax.imshow(confMatrix)

    data = confMatrix[0]
    labels = confMatrix[1]
    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(data)):
            ax.text(j, i, confMatrix[i, j],
                        ha="center", va="center", color="w")

    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    plt.show()


#SVM()
