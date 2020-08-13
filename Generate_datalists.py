"""
Generating Data Lists
 Author: Jean-Charles LAYOUN 
 August 2020

Description: 
 This code generates from .txt files the lists of tensors used for eitheir training, validation or testing.   
 
How to run it:
 python Generate_datalists.py $TXT_PATH $DATA_CATEGORY
    ex: python Generate_datalists.py data_lists/fold1_train.txt Train

NOTE:
    The $CSV_PATH must be relative to the location of this script.
    $DATA_CATEGORY must contain strictly one of these words: ["train", "test", "valid"]. It is not case sensitive, TrAiNing will be recognized.
"""

import os
import sys
import pandas as pd
import numpy as np

def DataCategory(Inp):
    Inp = Inp.lower()

    if "train" in Inp:
        return "Training"
    elif "test" in Inp:
        return "Test"
    elif "valid" in Inp:
        return "Validation"
    else:
        print("Error: $DATA_CATEGORY must contain strictly one of these words: [`train`, `test`, `valid`]. It is not case sensitive, `TrAiNing` will be recognized.")

def main():
    ## -- Setting up output folder: -- ##
    output_folder = "data_lists/"
    if not os.path.exists(output_folder):
        print("creating data list folder at `{}`".format(output_folder))
        os.mkdir(output_folder)

    ## -- Handling arguments: -- ##
    if(len(sys.argv) != 3):
        print("Error: Generate_datalists.py was used incorrectly. Please use as follows: `python Generate_datalists.py $TXT_PATH $DATA_CATEGORY`")
        exit()

    # Stores the path of the txt file and the data category of the input:
    txt_path = sys.argv[1]
    data_cat = DataCategory(sys.argv[2])
    ## -- End of handling arguments -- ##

    ## Loads txt:
    print("Warning: this script reads files with headers and data format similar to `fold1_train.txt`! If your txts have different format please modify this script.")
    # Announces to the user that it is reading the txt file:
    print("Reading the txt file... \t\t\t", end="")
    data_frame = pd.read_csv(txt_path, delimiter="\t", names = ["File Name", "Label", "Manually Verified"])
    # Announces the end to the user:
    print("Done!")

    ## -- Creates the list of data -- ##
    # Announces to the user that it is creating the data list:
    print("Creating the data list... \t\t\t", end="")

    # Initializes the list:
    data_list = []

    for index, row in data_frame.iterrows():
        fn = row['File Name']
        
        # If filename contains /, we extract the name:
        if "/" in fn:
            fn = fn.split('/')[-1]

        # Remove audio extension and replaces it with tensor extension
        data_list.append(fn.split(".")[0] + ".pt")

    print("Done!")
    ## -- End of creation block -- ##

    # Saves the dict:
    saved_file_name = "Tensor_{}_list.npy".format(data_cat)
    if(os.path.isfile(output_folder + saved_file_name)):
        response = str(input("Data lists were already provided would you like to overwrite existing data list ([y]/n)?"))
        if(response == "y"):
            np.save(output_folder + saved_file_name, data_list)
            print("Saved the list `{0}` in the directory `{1}`, initial file was overwritten.".format(saved_file_name, output_folder))
        else:
            print("Data list was not saved.")
    else:
        # Announces the end to the user and the file name:
        np.save(output_folder + saved_file_name, data_list)
        print("Saved the list `{0}` in the directory `{1}`!".format(saved_file_name, output_folder))




if __name__ == "__main__":
    # execute only if run as a script
    main()