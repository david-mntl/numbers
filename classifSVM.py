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
import math,time,glob,os,sys,audioManager,keras
import numpy as np
from numpy import concatenate, mean
import preprocesamiento as prep
from sklearn import svm




def loadData():    

    y_target= []
    x_data = []
    dafaultNumcep = 13
    numContext = 9

    for num in range(0, 16): # numero                      
        for idx in range(0, 31): # student id             
            for contx in range(1, 4): # context                
                try:
                    prep.obtenerMayorNumFrame(num, idx,contx)

                except :
                   print("Error al abrir",audioManager.getFileName(num,idx,contx))                   
                   # if you get here it means an error happende, maybe you should warn the user
                   # but doing pass will silently ignore it
                   pass
    num_frames=max(prep.listNF)
    for num in range(0, 16): # numero                      
        for idx in range(0, 31): # student id             
            for contx in range(1, 4): # context                
                try:
                    features = prep.test(num, idx,contx, num_frames,0)                    
                    y_target.append(num)                      
                    x_data.append(features)

                except :
                   print("Error al abrir",audioManager.getFileName(num,idx,contx))                   
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
    print(len(Data[0]))
    x_data = np.array(Data[0])
    y_target = np.array(Data[1])  
    #y_target = y_target[:3000]
#############################################################################
   
    #y_target = np_utils.to_categorical(y_target,16)  
    #x_data = keras.preprocessing.sequence.pad_sequences(x_data, maxlen=200)
    print(x_data.shape)
    print(x_data[0].shape)
    print(y_target.shape)
    print(y_target[0].shape)

    print("len ",len(x_data))
    print("len ",len(x_data[0]))
    print("len ",len(x_data[0][0]))
    
    audio_reshape =  np.zeros((len(x_data), len(x_data[0])*len(x_data[0][0]))) #len(x_data[0])*len(x_data[0][0])))

    for i in range(0,len(x_data)):
        temp =x_data[i]
        audio_reshape[i] = reshape_Audio(temp)

    audio_reshape = np.array(audio_reshape)
    #img_reshape = img_reshape.astype('float32')
    #print(img_reshape.shape)
    #print(img_reshape[0].shape)

    
############################################################################

    #Start time
    time_start=time.time()

   #Define model architecture

    clf=svm.SVC(kernel=str("linear"), C=1 , gamma= 0.5 )
    #print(X_train.shape)
    #print(y_train.shape)
    #print(X_test.shape)
    svc=clf.fit(audio_reshape,y_target) # train
    
    label_predict=svc.predict(audio_reshape) # predict


###########################################################################    
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



    #Compute ConfussionMatrix
   # testPredict(model, x_data ,y_target)
    



def testPredict(model , x_test, y_test):

    x_test = x_test
    y_test = y_test
    
    audios = x_test
    labels = y_test
      
    a = []
    b = []
    #y = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    print(x_test.shape[0])
    print("This may take a while.")
    for x in range(0, audios.shape[0]-1):
        b.insert(x,labels[x])        
        test = audios[x].reshape(1,audios[x].shape[0],audios[x].shape[1])
        test = keras.preprocessing.sequence.pad_sequences(test, maxlen=200)        
        
        a.insert(x,model.predict_classes(test ,verbose=0))

    a = np.asarray(a)
    b = np.asarray(b)
    print ("a" , a)
    print ("B" , b)
    print(confusion_matrix(b, a))




#######################################################################################
def audiofile_to_input_vector(audio_filename, numcep, numcontext):
    '''
    Turn an audio file into feature representation.
    This function has been modified from Mozilla DeepSpeech:
    https://github.com/mozilla/DeepSpeech/blob/master/util/audio.py
    # This Source Code Form is subject to the terms of the Mozilla Public
    # License, v. 2.0. If a copy of the MPL was not distributed with this
    # file, You can obtain one at http://mozilla.org/MPL/2.0/.
    '''

    # Load wav files
    fs, audio = wav.read(audio_filename)

    # Get mfcc coefficients
    orig_inputs = mfcc(audio, samplerate=fs, numcep=numcep)

    # We only keep every second feature (BiRNN stride = 2)
    orig_inputs = orig_inputs[::2]

    # For each time slice of the training set, we need to copy the context this makes
    # the numcep dimensions vector into a numcep + 2*numcep*numcontext dimensions
    # because of:
    #  - numcep dimensions for the current mfcc feature set
    #  - numcontext*numcep dimensions for each of the past and future (x2) mfcc feature set
    # => so numcep + 2*numcontext*numcep
    train_inputs = np.array([], np.float32)
    train_inputs.resize((orig_inputs.shape[0], numcep + 2 * numcep * numcontext))

    # Prepare pre-fix post fix context
    empty_mfcc = np.array([])
    empty_mfcc.resize((numcep))

    # Prepare train_inputs with past and future contexts
    time_slices = range(train_inputs.shape[0])
    context_past_min = time_slices[0] + numcontext
    context_future_max = time_slices[-1] - numcontext
    for time_slice in time_slices:
        # Reminder: array[start:stop:step]
        # slices from indice |start| up to |stop| (not included), every |step|

        # Add empty context data of the correct size to the start and end
        # of the MFCC feature matrix

        # Pick up to numcontext time slices in the past, and complete with empty
        # mfcc features
        need_empty_past = max(0, (context_past_min - time_slice))
        empty_source_past = list(empty_mfcc for empty_slots in range(need_empty_past))
        data_source_past = orig_inputs[max(0, time_slice - numcontext):time_slice]
        assert(len(empty_source_past) + len(data_source_past) == numcontext)

        # Pick up to numcontext time slices in the future, and complete with empty
        # mfcc features
        need_empty_future = max(0, (time_slice - context_future_max))
        empty_source_future = list(empty_mfcc for empty_slots in range(need_empty_future))
        data_source_future = orig_inputs[time_slice + 1:time_slice + numcontext + 1]
        assert(len(empty_source_future) + len(data_source_future) == numcontext)

        if need_empty_past: 
            past = np.concatenate((empty_source_past, data_source_past))
        else:
            past = data_source_past

        if need_empty_future:
            future = np.concatenate((data_source_future, empty_source_future))
        else:
            future = data_source_future

        past = np.reshape(past, numcontext * numcep)
        now = orig_inputs[time_slice]
        future = np.reshape(future, numcontext * numcep)

        train_inputs[time_slice] = np.concatenate((past, now, future))
        assert(len(train_inputs[time_slice]) == numcep + 2 * numcep * numcontext)

    # Scale/standardize the inputs
    # This can be done more efficiently in the TensorFlow graph
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
    return train_inputs




#SVM()
