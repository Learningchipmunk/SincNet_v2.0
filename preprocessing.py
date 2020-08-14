"""
Audio Preprocessing 
 Author: Jean-Charles LAYOUN 
 August 2020

Description: 
 This code prepares audio files for sound recognition experiments. 
 It normalizes  audio files then removes start and end parts that are inferior to a threshold of the energy (by default it is 25%). It also pads audio files that are below the required length.
 
How to run it:
 python preprocessing.py $TRAIN_FOLDER $TEST_FOLDER $OUTPUT_TRAIN_FOLDER $OUTPUT_TEST_FOLDER $wlen $sr
    ex: python preprocessing.py /data2/dcase2018/task2/FSDKaggle2018.audio_train/ /data2/dcase2018/task2/FSDKaggle2018.audio_test/ Data/test_train/ Data/test_test/ 1000 32000

NOTE: This script is case sensitive.
"""


import librosa
import librosa.display
import soundfile
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import os
from os import listdir
from os.path import isfile, join


## <!>-------------------------- Functions useful for preprocessing --------------------------<!> ##
def Normalization(signal):
    """Function that takes a monophonic audio signal as input and normalizes the amplitude.

    Args:
        signal (list or np.array): audio file as 1D list.

    Returns:
        signal (list or np.array): audio file normalized as 1D list.
    """
    return signal/np.max(np.abs(signal))   


def plotImg(audio, samplingRate, ylabel = "Amplitude", name = "Audio"):
    librosa.display.waveplot(audio, sr=samplingRate)
    plt.ylabel(ylabel)
    plt.title('Amplitude envelope of a waveform')
    plt.savefig(name)
    plt.clf()




def EnergyWindowMean(audio, L=300, stride=150, padding = False, removingZeros = False, debug_mode = False):
    """Function that computes the energy window of the input. 

    Args:
        audio (list or np.array): audio file as 1D list.
        L (int, optional): Size of the window. Defaults to 300.
        stride (int, optional): Hop length of the window. Defaults to 150.
        padding (bool, optional): Indicates if the user wishes to remove zeros from the computed Energy Window. Defaults to False.
        removingZeros (bool, optional): Indicates if the user wishes to remove zeros from the result of the energy computation, must be True if so. Defaults to False.
        debug_mode (bool, optional): Indicates if the user wishes to print information during the computation, must be True if so. Defaults to False.

    Returns:
        EnergyWindow (np.array): Energy window of the input.
    """
    N = len(audio)
    
    ## Just for display purposes, removes
    if(debug_mode):print(N)
    
    ## Adds padding if it is requested:
    if(padding):
        
        if((N - L)%stride != 0):
            for i in range (0, stride - (N - L)%stride):
                audio = np.append(audio, 0)
                
        ## Computes the new length after padding 
        N = len(audio)
        
        if(debug_mode):print(N-L, N)
    
    
    ## <!>--------- Computes the enrgy Here ---------<!> ##
    
    # Initializes the energy array
    #np.array([sum([el*el for el in audio[i:i+L]])/L  for i in (0, (N - L), stride)])
    Energy = np.zeros(int((N-L)/stride) + 1)
    
    ## Very important +1 is needed for i to be equal to (N - L)!
    for i in range(0, (N - L + 1), stride):
        E = 0
        for j in range (i, i + L):
            E += audio[j]*audio[j]
            
        E /= L
        
        Energy[int(i/stride)] = E

    ## <!>------------------- Done -------------------<!> ##

    if(removingZeros):
        ## Removing zeros from energy:
        for i in range (len(Energy)-1, -1, -1):
            if(Energy[i] > 0):
                Energy = Energy[:i+1]
                break
    
    if(debug_mode):
        print(len(Energy))
        
        string_pad = "with padding" if padding else "without padding"
        
        print("Expected length of the array " + string_pad + " : " + str(int((N-L)/stride) + 1))
    
    return Energy



def TrimEnergy(audio,
               threshold=5,
               delay = 0,
               window_length = 200,
               samplingRate=32000,
               L=600, stride=300,
               padding = True,
               random_padding_zeros = False,
               repeating_signal = False,
               printInterval = False):
    """Trims Audio based on threshold and Energy mean.

    Args:
        audio (list or np.array): audio file as 1D list.
        threshold (int, optional): Is the threshold of the trim in percentage (%), meaning that it cuts the signal if the energy is below the specified values. Defaults to 5%.
        delay (int, optional): Is the time delay of the trim, basically shifts the trim to the left (values). Defaults to 0 values.
        window_length (int, optional): The size of the desired window in (values). Defaults to 200.
        samplingRate (int, optional): The desired samplig rate in (Hz). Defaults to 32000Hz.
        L (int, optional): Size of the window. Defaults to 300.
        stride (int, optional): Hop length of the window. Defaults to 150.
        padding (bool, optional): Indicates if the user wishes to remove zeros from the computed Energy Window. Defaults to True.
        random_padding_zeros (bool, optional): Indicates if the user wishes to pad by adding a random number of zeros to the left and the remaining amount to the right. Defaults to False.
        repeating_signal (bool, optional): Indicates if the user wishes to pad repeating the signal until desired length is reached. Defaults to False.
        printInterval (bool, optional): Indicates if the user wishes to print information during the computation, must be True if so. Defaults to False.

    Returns:
        audio (np.array): Trimmed audio file as 1D list.
    """
    # Is the size of the window of SincNet:
    minAudioSize = samplingRate *window_length /1000
    
    # We store the sieze of the audio file:
    size = len(audio)
    
    ## Computing the average energy of the signal:
    Energy_Mean = sum([el*el for el in audio])/size
    
    ## Computing the energy threshold:
    Energy_Threshold = Energy_Mean * threshold/100
    
    ## The Energy window: (It is the average energy of a window)
    Energy = EnergyWindowMean(audio, L, stride, padding = padding, debug_mode = printInterval)
    
    #print(Energy_Mean, Energy_Threshold, Energy[:100])
    
    
    ## Pads with 0s if audio is not long enough:
    if(size < minAudioSize):
        if(repeating_signal):
            
            ## We continue to add values till the signal is at required window size, resulting into a repeated signal:
            i=0
            while(size < minAudioSize):
                audio = np.append(audio, audio[i])
                
                ## Updates the size, is basically a size+=1 but for correctness is written like this.
                size = len(audio)
                i+=1
                
            return audio
        
        elif(random_padding_zeros):
            
            number_of_zeros_added = minAudioSize - size
            nbre_zeros_gauche     = int(np.random.randint(0, number_of_zeros_added))
            nbre_zeros_droite     = int(number_of_zeros_added - nbre_zeros_gauche)
            
            #print("The number of zeros to the left is {}".format(nbre_zeros_gauche))
            
            return np.pad(np.append(np.zeros(nbre_zeros_gauche), audio), (0, nbre_zeros_droite), 'constant')
        
        else:
            ## New version that pads to the right and the left equally:
            nbre_zeros_gauche = int((minAudioSize - size)/2 + (minAudioSize - size)%2)
            nbre_zeros_droite = int((minAudioSize - size)/2)

            return np.pad(np.append(np.zeros(nbre_zeros_gauche), audio), (0, nbre_zeros_droite), 'constant')
        
            ## Pads everything to the right:
            #return np.pad(audio, (0, abs(int(minAudioSize - size))), 'constant')
        
    elif(size == minAudioSize):
        return audio
    
    
    ## <!>----------------------- Trimming begins here -----------------------<!> ##
    begin, end = (0, size-1)
    
    ## Time complexity is O(size/stride), way better than TrimTheFat
    for i in range(0, size, stride):
        ChangedPostition = False
        
        if(Energy[int(-i/stride) -1] <= Energy_Threshold):
            end = size - i
            ChangedPostition= True
            
        
        # Left pos is an anchor, once it finds the right pos it doesn't move anymore, right can if we change the last if.
        if(Energy[int(begin/stride)] <= Energy_Threshold):
            begin = i
            ChangedPostition= True
        
        
        # Because of the stride, sometimes we remove more then we would like to...
        if(end-begin <= minAudioSize):
            ## We add reamining points from right to left
            remaining_points_to_add = minAudioSize - (end - begin)#remaining_points_to_add in theory is < 2 * stride !
            
            ## We compute the number of points to add:
            nbre_points_left  = int((remaining_points_to_add)/2 + remaining_points_to_add%2)
            nbre_points_right = int((remaining_points_to_add)/2)
            
            ## We add to the right the number of points we couldn't add to the left:
            nbre_points_right += nbre_points_left - (begin - max(begin - nbre_points_left, 0))
            
            ## Updating iterators:
            begin = max(begin - nbre_points_left, 0)
            end   += nbre_points_right
        
            break
        
        ## If the positions of the trim do not change, then it means that we finished the trimming! 
        ## We can change that if we wish to 
        if(not ChangedPostition):
            break
    
    ## For visualization only:
    if(printInterval):
        print(begin, end, size)
        print("Resulting size is : {}".format(end-begin))
    
    if(delay != 0):
        begin = max(begin-delay, 0)
        
    ## For visualization only:
    if(printInterval):
        print("Last interval print, after the Trim: [{0}, {1}] of original size {2}.".format(begin, end, size))
        if(end-begin < minAudioSize):
            print("Houston - we have a problem.")
    
    ## We take one more point in case !
    return audio[begin:end+1]
## <!>-------------------------- End of functions useful for preprocessing --------------------------<!> ##


## Dunction used for Preprocessing:
def preprocess(targetSamplingRate,
               AudioFiles,
               path_to_load,
               path_to_save,
               window_length,
               threshold_percentage = 0,
               L=600,# User should not worry about stride and Energy size.
               stride = 300,
               threshold = 0,
               delay = 0,# By default it is 0 because if user does not specify a delay we should not add one.
               TrimWithEnergy = True,
               padding = True,
               random_padding_zeros = False,
               repeating_signal = False,
               path_to_save_audio = None,
               writing_audio = False):
    """Function that executes the preprocessing.

    Args:
        targetSamplingRate (float): The desired samplig rate in (Hz).
        AudioFiles (list): A list of the names of the audio files.
        path_to_load (str): String that indicates the path of the audio files we need to load and preprocess.
        path_to_save (str): String that indicates the path owhere we need to save the audio files.
        window_length (int): The size of the desired window in (values). 
        threshold_percentage (int, optional): Is the threshold of the trim in percentage (%), meaning that it cuts the signal if the energy is below the specified values. Defaults to 5%.
        L (int, optional): Size of the window. Defaults to 600.
        stride (int, optional): Hop length of the window. Defaults to 300.
        padding (bool, optional): Indicates if the user wishes to remove zeros from the computed Energy Window. Defaults to True.
        random_padding_zeros (bool, optional): Indicates if the user wishes to pad by adding a random number of zeros to the left and the remaining amount to the right. Defaults to False.
        repeating_signal (bool, optional): Indicates if the user wishes to pad repeating the signal until desired length is reached. Defaults to False.
        path_to_save_audio (str, optional): If the user wishes to save the preporcessed audio as wav files, he needs to indicate the path where the algorithm is going to save them. Defaults to None.
        writing_audio (bool, optional): Indicates if  the user wishes to save the preporcessed audio as wav files, if so it needs to bet set to True. Defaults to False.
    """
    ## Removes user error:
    if(TrimWithEnergy and threshold_percentage==0):
        print("Please set a value for the threshold_percentage var.")
        return -1
    elif(not TrimWithEnergy and threshold==0):
        print("Please set a value for the threshold var.")
        return -1
    
   # Folders creation if the folder path does not exist:
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    print("Done!")
    
    
    for i, filename in enumerate(AudioFiles):

        train_audio, _ = librosa.load(path_to_load + "/" + filename,  sr=targetSamplingRate)

        ## Converts from stereo or more to mono:
        if(len(train_audio.shape) != 1):
            print("Warning, SincNet only support one channel (here, channels = {%i})" % (len(train_audio.shape)))
            print("We took only the first channel of the audio.")
            train_audio = train_audio[0,:]


        


        ## Does not display info on audio files:
        printInterval = False

        ## We use the preprocessing algorithms use above:
        preprocessedTrainAudio = TrimEnergy(Normalization(train_audio),
                                            threshold = threshold_percentage,
                                            delay=delay,
                                            window_length = window_length,
                                            samplingRate = targetSamplingRate,
                                            L=L,
                                            stride=stride,
                                            padding = padding,
                                            random_padding_zeros = random_padding_zeros,
                                            repeating_signal = repeating_signal,
                                            printInterval = printInterval)


        ## Plotting first 10 audios in a temporary directory:
        if(i<10):
            imgs_path = "temporary_images/"
            if not os.path.exists(imgs_path):
                print("Saving temporary images of audio files in the directory `{}`".format(imgs_path))
                os.mkdir(imgs_path)

            name = imgs_path + "audio_" + filename.split(".")[0]
            plotImg(preprocessedTrainAudio, targetSamplingRate, name=name)

        ## Saving audio files if requested:        
        if(writing_audio):
            soundfile.write(path_to_save_audio + "/" + filename, data=preprocessedTrainAudio, samplerate=targetSamplingRate)
        
        ## Stores the audio in a tensor
        temp_t = torch.tensor(preprocessedTrainAudio).float()
        
        ## Solves new name for tensor
        new_name = filename.split(".")[0] + ".pt"
        
        ## Saves the tensor to path
        torch.save(temp_t, path_to_save + new_name)
    
        # The ausio size is desired to be of size at least targetSamplingRate * window_length/ 1000
        if(len(preprocessedTrainAudio) < targetSamplingRate *window_length /1000):
            print('audio sample {0} is too small ! His length is equal to {1}'.format(filename, len(preprocessedTrainAudio)))
            break



def main():
    ## -- Handling arguments: -- ##
    if(len(sys.argv) != 7):
        print("Error: preprocessing.py was used incorrectly. Please use as follows: `python preprocessing.py $TRAIN_FOLDER $TEST_FOLDER $OUTPUT_TRAIN_FOLDER $OUTPUT_TEST_FOLDER $sr $wlen`")
        exit()

    ## Adds an `/` at the end of each arg if it is missing:
    for i in range (1, 5):
        if sys.argv[i][-1]!= "/":
            sys.argv[i] += "/"

    # Stores the path of the original audio files:
    in_folder_train=sys.argv[1]
    in_folder_test=sys.argv[2]

    # Stores the path of the output audio files:
    out_folder_train=sys.argv[3]
    out_folder_test=sys.argv[4]

    # Stores the data relevant to the preprocessing:
    window_length = int(sys.argv[5])#ms
    targetSamplingRate = int(sys.argv[6])#Hz
    ## -- End of handling arguments -- ##

    print("Reading audio files... \t\t\t\t\t\t\t\t", end="")
    # Gets the names of the .wav files in the folder
    trainAudioFiles = [f for f in listdir(in_folder_train) if isfile(join(in_folder_train, f))]
    testAudioFiles = [f for f in listdir(in_folder_test) if isfile(join(in_folder_test, f))]
    print("Done!")

    ## Defining parameters for Preprocessing:
    # Parameters used to read audio files:
    targetSamplingRate#Hz
    # Train
    in_folder_train
    trainAudioFiles
    # Test:
    in_folder_test
    testAudioFiles

    # Parameter used to store ouput tensor files:
    out_folder_train
    out_folder_test

    # Parameters of the trim:
    window_length#ms
    threshold_percentage = 25#%
    L = 600#values
    stride = 300#values
    delay = 320 #values
    padding = True
    random_padding_zeros = True
    repeating_signal = False

    # Parameter used to store audio files:
    writingDirAudioTrain_3 = "Data/Audio_Files/NeedToCreateDir"
    writing_audio = False


    print("Beginning preprocessing of train audio files with fs={0}Hz and wlen={1}... \t".format(targetSamplingRate, window_length), end="")
    ## Runinng the preprocessing script on Train data:
    preprocess(targetSamplingRate,
            trainAudioFiles,
            in_folder_train,
            out_folder_train,
            window_length,
            threshold_percentage = threshold_percentage,
            L=L,
            stride = stride,
            delay = delay,
            TrimWithEnergy = True,
            padding = padding,
            random_padding_zeros = random_padding_zeros,
            repeating_signal = repeating_signal,
            path_to_save_audio = writingDirAudioTrain_3,
            writing_audio = writing_audio)
    print("Preprocessing of train audio is done!")
    print("Wrote in the " + out_folder_train + " directory!\n")

    print("Beginning preprocessing of test audio files with fs={0}Hz and wlen={1}... \t".format(targetSamplingRate, window_length), end="")
    ## Runinng the preprocessing script on Test data with same paramters as Train data:
    preprocess(targetSamplingRate,
            testAudioFiles,
            in_folder_test,
            out_folder_test,
            window_length,
            threshold_percentage = threshold_percentage,
            L=L,
            stride = stride,
            delay = delay,
            TrimWithEnergy = True,
            padding = padding,
            random_padding_zeros = random_padding_zeros,
            repeating_signal = repeating_signal,
            path_to_save_audio = writingDirAudioTrain_3,
            writing_audio = writing_audio)
    print("Preprocessing of test audio is done!")
    print("Wrote in the " + out_folder_train + " directory!\n")

    ## Removing temporary images
    shutil.rmtree("temporary_images/")
    print("Removed all audio images.")



if __name__ == "__main__":
    # execute only if run as a script
    main()