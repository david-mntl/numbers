''' #Start time
    time_start=time.time()

   #Define model architecture
    model = Sequential()
    model.add(LSTM(16, input_shape=(200,247)))
    model.add(Dropout(0.25))
    #model.add(Flatten())
    model.add(Dense(1, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='softmax'))
###########################################################################    
        #Compile model
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    #Fit model on training data
    model.fit(x_data, y_target,
            batch_size=32, epochs=200, verbose=1)

    # Total Time
    time_end=time.time() 
    print('Time to classify: %0.2f.' % ((time_end-time_start)/60))

    model.save(str("model.h5"))  # creates a HDF5 file 'my_model.h5'
    '''