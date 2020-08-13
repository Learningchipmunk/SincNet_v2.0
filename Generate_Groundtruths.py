"""
Generating Ground Truths
 Author: Jean-Charles LAYOUN 
 August 2020

Description: 
 This code generates from .csv files dictionaries of ground truths.
 
How to run it:
 python Generate_Groundtruths.py $CSV_PATH $DATA_CATEGORY
    ex: python Generate_Groundtruths.py data_lists/train.csv Train

Note:
    The $CSV_PATH must be relative to the location of this script.
    $DATA_CATEGORY must contain strictly one of these words: ["train", "test"]. It is not case sensitive, TrAiNing will be recognized.
"""

import sys
import os
import pandas as pd
import numpy as np


def DataCategory(Inp):
    Inp = Inp.lower()

    if "train" in Inp:
        return "train"
    elif "test" in Inp:
        return "test"
    else:
        print("Error: $DATA_CATEGORY must contain strictly one of these words: [`train`, `test`]. It is not case sensitive, `TrAiNing` will be recognized.")


def main():
    ## -- Setting up output folder: -- ##
    output_folder = "data_lists_test/"
    if not os.path.exists(output_folder):
        print("creating data list folder at `{}`".format(output_folder))
        os.mkdir(output_folder)

    ## -- Handling arguments: -- ##
    if(len(sys.argv) != 3):
        print("Error: Generating_Groundtruths.py was used incorrectly. Please use as follows: `python Generating_Groundtruths.py $CSV_PATH $DATA_CATEGORY`")
        exit()

    # Stores the path of the csv file and the data category:
    csv_path = sys.argv[1]
    data_cat = DataCategory(sys.argv[2])
    ## -- End of handling arguments -- ##


    ## -- Checks if a dictionary that converts labels to numbers exists -- ##
    create_ltn = False

    labels_to_number = {}
    if(os.path.isfile(output_folder + "labelsToNumberDict.npy")):
        labels_to_number = np.load(output_folder + "labelsToNumberDict.npy").item()
    else:
        create_ltn = True
    ## -- End of Checking -- ##

    ## Loads csv:
    data_frame = pd.read_csv(csv_path)

    ## -- Creates the dictionary that converts labels to numbers if needed -- ##
    if create_ltn:
        # Announces to the user that it is creating the dict:
        print("Creating the dictionary that converts labels to numbers... \t\t", end="")

        # Initializing counter:
        countLabels = 0

        for index, row in data_frame.iterrows():
            fileName = row['fname']
            label    = row['label']
            
            # Adds new entry and increments label: 
            if(label not in labels_to_number.keys()):
                labels_to_number[label] = countLabels
                countLabels += 1
        
        # Saves it for future uses:
        np.save(output_folder + "labelsToNumberDict.npy", labels_to_number)

        # Announces the end to the user:
        print("Done!")
    ## -- End of creation -- ##

    ## -- Creates the Dictionary of Ground Truths -- ##
    # Announces to the user that it is creating the dict:
    print("Creating the dictionary of Ground Truths... \t\t\t", end="")

    groundtruths_tensors = {}

    for index, row in data_frame.iterrows():
        fileName = row['fname']
        label    = row['label']
        
        # Storing label by file name:
        groundtruths_tensors[fileName.split(".")[0] + ".pt"] = labels_to_number[label]

    print("Done!")
    ## -- End of creation block -- ##

    # Saves the dict:
    saved_file_name = "DCASE_tensor_{}_labels.npy".format(data_cat)
    if(os.path.isfile(output_folder + saved_file_name)):
        response = str(input("Ground Truth Dict were already provided would you like to overwrite existing data list ([y]/n)?"))
        if(response == "y"):
            np.save(output_folder + saved_file_name, groundtruths_tensors)
            print("Saved the Ground Truth Dict `{0}` in the directory `{1}`, initial file was overwritten.".format(saved_file_name, output_folder))
        else:
            print("Data Ground Truth Dict was not saved.")
    else:
        np.save(output_folder + saved_file_name, groundtruths_tensors)
        # Announces the end to the user and the file name:
        print("Saved the Ground Truth Dict `{0}` in the directory `{1}`!".format(saved_file_name, output_folder))



if __name__ == "__main__":
    # execute only if run as a script
    main()
