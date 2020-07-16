import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch


## Loaded the list of indexes I already created:
dictOfLabels = np.load("data_lists/labelsToNumberDict.npy").item()

## List of labels
index = [0 for i in range (0, len(dictOfLabels))]

## Putting the label in the right order
for label, i in  dictOfLabels.items():
    index[i] = label

    
def confusion_matrix(mat, qty, pred=None, labels=None, index = index, write_results = False, name = "Test", cuda = True): 
    """This function computes the confusion matrix and saves it.

    Args:
        mat (2D matrix): is a 2D confusion matrix.
        qty (list): is 
        pred (torch.Tensor, optional): the predictions of the current batch. Defaults to None.
        labels (torch.Tensor, optional): the groundtruths of the current batch. Defaults to None.
        index (list, optional): a list of label names by integer index values. Defaults to index.
        write_results (bool, optional): If the user wishes to save confusion matrix it needs to be true. Defaults to False.
        name (str, optional): name of the confusion matrix saving file. Defaults to "Test".
        cuda (bool, optional): Indicates if the user is using cuda. Defaults to True.

    Returns:
        (2D matrix, list): returns mat and qty.
    """
    size = len(index)
    
    if(write_results):
        
        ## Dividing by total number:
        for k in range(size):
            if(qty[k]!= 0):
                mat[k] *= 1/qty[k]
                
        ## Converting mat into data frame in order to use seaborn:
        dataframe = pd.DataFrame(mat, index=index)
        sn.set(font_scale=1)

        # Creates the heatmap:
        fig=plt.figure(figsize=(15, 12), dpi= 400, facecolor='w', edgecolor='k')
        svm = sn.heatmap(dataframe, annot=False)

        # Saves the figure than plots it:
        figure = svm.get_figure()
        figure.savefig('Images/Confusion_Matrices/Conf_Mat_' + name +'.png', dpi=400)
        #plt.show()
        
    else:
    
        if(pred is None or labels is None):
            print("Error, inputs can't be None if not in write results mode.")
            return -1

        if(mat is None):
            #print("Initialize confusion matrix")
            mat = np.zeros([size,size])
        
        if(qty is None):
            qty = [0]*size
        
        if cuda and (not pred.is_cuda or not labels.is_cuda):
            pred   = pred.type(torch.cuda.FloatTensor)
            labels = labels.type(torch.cuda.LongTensor)
        
        
        for k in range(labels.size(0)): # ground truth first (row) and then prediction (col)
         # print("Here : " +str(labels[k].item()) + " ; " + str(predicted[k].item()))
            mat[labels[k].item(),pred[k].item()] +=1
            qty[labels[k].item()]                +=1
            
        
            
    return mat, qty