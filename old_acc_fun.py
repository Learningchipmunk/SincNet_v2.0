"""
Old Accuracy function is stored here in order to compare its results with new one.
"""
import tqdm
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from utils import plot_grad_flow, mixup

## Local files imports:
from Confusion_Matrix import confusion_matrix


## <!>---------------------------- Old accuracy function ----------------------------<!>
def accuracy(net, test_loader, criterion, n_classes,
             ## SincNet Params
             Batch_dev, wlen, wshift, 
             ## Confusion_Matrix param:
             matrix_name, compute_matrix = False,
             cuda=True):
    """ This is the function that computes the accuracy of the Network.

    Args:
        net (nn.Module): inherits from nn.Module and is a network.
        test_loader (torch.utils.data.DataLoader): the test data loader.
        criterion (nn.Loss): is the loss function.
        Batch_dev (int): is the number of test tensors that are stored at once in test loop.
        wlen (int): is the size of the input of the network, for SincNet it is considered the window size of the audio.
        wshift (int): is the step in between each window.
        matrix_name (str): is the name of the confusion matrix that will be saved.
        compute_matrix (bool, optional): Indicates if user wants to compute and save confusion matrix. Defaults to False.
        cuda (bool, optional): Indicates if net is on cuda. Defaults to True.

    Returns:
        (float, float, float): (best class error, validation loss, mean error on each window)
    """


    ## Modifs pour SincNet
    net.eval()
    

    loss_sum=0
    err_sum=0
    err_sum_snt=0
    ## End
    
    with torch.no_grad():
        ## Number of batches in test_loader
        snt_te = 0

        # Initializes confusion matrix and quantity to None:
        mat = None
        qty = None

        ## For SincNet, information in testloader is raw and can be of various lengths!
        for data in test_loader:
            ## Increments the number of batches in test_loader
            snt_te += 1

            ## Stores data from test_loader
            audios, labels = data
            audios = audios[0]#We take the first and only tensor...
            
            if cuda:
                audios = audios.type(torch.cuda.FloatTensor)
                labels = labels.type(torch.cuda.LongTensor)
                         
            
            
            ## split signals into chunks of wlen
            beg_samp=0
            end_samp=wlen

            N_fr=int((audios.shape[0]-wlen)/(wshift))
            #print(audios.shape[0], wlen, N_fr, audios.shape)

            ## Var initialization
            sig_arr=torch.zeros([Batch_dev,wlen]).float().cuda().contiguous()
            
            lab= Variable((torch.zeros(N_fr+1) + labels[0]).cuda().contiguous().long())
            pout=Variable(torch.zeros(N_fr+1,n_classes).float().cuda().contiguous())


            count_fr=0
            count_fr_tot=0
            while end_samp<audios.shape[0]:
                sig_arr[count_fr,:]=audios[beg_samp:end_samp]

                ## Shifts the signal every iteration
                beg_samp=beg_samp+wshift
                end_samp=beg_samp+wlen

                count_fr=count_fr+1
                count_fr_tot=count_fr_tot+1

                if count_fr==Batch_dev:
                    inp=Variable(sig_arr)
                    pout[count_fr_tot-Batch_dev:count_fr_tot,:] = net(inp)
                    count_fr=0
                    sig_arr=torch.zeros([Batch_dev,wlen]).float().cuda().contiguous()

            if count_fr>0:
                inp=Variable(sig_arr[0:count_fr])
                pout[count_fr_tot-count_fr:count_fr_tot,:]=net(inp)


            ## Predicts for every chunk of audio the label and counts how many time it got it correctly
            pred=torch.max(pout,dim=1)[1]
            loss = criterion(pout, lab.long())
            err = torch.mean((pred!=lab.long()).float())

            print(pred.size(0))

            ## Updates the confusion matrix:
            if(compute_matrix):
                mat, qty = confusion_matrix(mat, qty, pred=pred, labels=lab, write_results = False, name = "Pas Important", cuda = True)

            ## Updates the error that I use here:
            loss_sum=loss_sum+loss.detach()
            err_sum=err_sum+err.detach()

            ## Sum the probability over the columns, then it stores the value and the position of the max. (Lionel's Method)
            [val,best_class]=torch.max(torch.sum(pout,dim=0), 0)
            err_sum_snt=err_sum_snt+(best_class!=lab[0]).float()

        
        ## mean Error of best class:
        err_tot_dev_snt=err_sum_snt/snt_te
        
        ## mean Loss:
        loss_tot_dev=loss_sum/snt_te
        
        ## mean Error on each window:
        err_tot_dev=err_sum/snt_te
        
        ## Plots and saves the confusion matrix:
        if(compute_matrix):
            mat = confusion_matrix(mat, qty, write_results = True, name = matrix_name, cuda = True)
            
            
    net.train()

    return (err_tot_dev_snt, loss_tot_dev, err_tot_dev)

