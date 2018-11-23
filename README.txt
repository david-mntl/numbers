 
To properly run the program it is needed to install all the dependencies in the file named "dependencies.txt"

To train de the model:

$python3 initTraining.py

To predict and init the evaluation method implemented:
$python3 predictSVM.py

How to use it:

- When running predictSVM click on the "StartRecording" button to record an audio. When you are finished recording click on stop recording.
- Click on the "Graph audio" button to see the signal plotted
- Click on the "predict" button to predict the recorded audio with the already trained model. It will display the prediction in the box on the top right of the screen
- To crop the audio put the numbers on the axis x on the corresponding init and end Entrys and press the Crop button
- Click the "Graph audio" again to verify the crop has been done succesfully
- Click the "Predict Button" to verify the input

- NOTE: If you want to run the program with an audio file recorded with another program (for example audacity), just click the search icon and click the "load button". It will replace
  the file predict.wav and the you can run the program as mentioned above.