
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
from matplotlib.lines import Line2D

## Loads previously trained model:
#Function was modified especially for .py folders:
def LoadPrevModel(CNN_net, DNN1_net, DNN2_net, model_file_path, Models_file_extension, Load, inSameFile = True, at_epoch = 0, evalMode = False):
    """Loads previously trained model.

    Args:
        CNN_net (nn.Module): inherits from nn.Module and is a network.
        DNN1_net (nn.Module): inherits from nn.Module and is a network.
        DNN2_net (nn.Module): inherits from nn.Module and is a network.
        model_file_path (str): Model's file path.
        Models_file_extension (str): File extensions.
        Load (bool): The user has to set it to true if he wishes to load a previous model.
        evalMode (bool, optional): Indicates if the user wishes to evaluat the net, if true sets networks to eval(). Defaults to False.
        inSameFile (bool, optional): Indicates if the models are all saved in the same file. Defaults to True.
        at_epoch (int, optional): The amount of epoch the model was trained on. Defaults to N_epochs.

    Returns:
        (int): The current starting epoch for the loaded model. 
    """
    
    if(Load == False):
        return 0
    
    
   
    if(inSameFile):
        ## Loading the pretrained setup file
        pretrainedSetup = torch.load(model_file_path + Models_file_extension)
        #print(pretrainedSetup['CNN_model_par'])
        
        ## Loading net parameters one by one:
        CNN_net.load_state_dict(pretrainedSetup['CNN_model_par'])
        if(evalMode):CNN_net.eval()

        DNN1_net.load_state_dict(pretrainedSetup['DNN1_model_par'])
        if(evalMode):DNN1_net.eval()

        DNN2_net.load_state_dict(pretrainedSetup['DNN2_model_par'])
        if(evalMode):DNN2_net.eval()
    else:
        ## Loading all the pretrained setup file
        
        pretrainedSetup_CNN  = torch.load(model_file_path + "_CNN" + Models_file_extension)
        pretrainedSetup_DNN1 = torch.load(model_file_path + "_DNN1" + Models_file_extension)
        pretrainedSetup_DNN2 = torch.load(model_file_path + "_DNN2" + Models_file_extension)
        
        ## Loading net parameters one by one:
        CNN_net.load_state_dict(pretrainedSetup_CNN)
        if(evalMode):CNN_net.eval()

        DNN1_net.load_state_dict(pretrainedSetup_DNN1)
        if(evalMode):DNN1_net.eval()

        DNN2_net.load_state_dict(pretrainedSetup_DNN2)
        if(evalMode):DNN2_net.eval()
    
    #print(CNN_net.state_dict()['conv.0.low_hz_'][0])
    
    print("Models from " + model_file_path + " were loaded successfully!")
    
    return CNN_net, DNN1_net, DNN2_net, at_epoch + 1




## Gradient histogram rep, Calot's suggestion:
def plot_grad_flow(named_parameters, plot_both = False):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Args:
        named_parameters (self.model.named_parameters()): Named parameters of the network.
        plot_both (bool, optional): If user desires to plot bar and line. Defaults to False.

    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    """

    ave_grads = []
    max_grads= []
    layers = []

    # n is for name and p is for parameter
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and (p.grad is not None):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    
    ## Plotting also lines like in plot_grad_flow_simple:
    if(plot_both):
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
    
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    
    
## <!>------------------------------------ Mixup Utilities ------------------------------------<!> ##

def onehot(label, n_classes):
    """Returns a one hot encoded version of the labels.

    Args:
        label (Torch.Tensor): A tensor of type long wich describes the labels as integers.
        n_classes (int): The number of classes.

    Returns:
        (Torch.Tensor):  One hot encoded labels.
    """
    return torch.zeros(label.size(0), n_classes).scatter_(
        1, label.view(-1, 1), 1)

def mixup(data, targets, beta_coef, n_classes, sameClasses = False, debug = False):
    """Implements a data augmentation technique called mixup, we basically mix data together to create new data.

    Args:
        data (Torch.Tensor): The data tensor.
        targets (Torch.Tensor): The groundtruths.
        beta_coef (float): The coef of the Beta(beta_coef , beta_coef) distribution.
        n_classes (int): The number of classes.
        sameClasses (bool, optional): If the user chooses to apply mixup only on data with same classes,
                                      he needs to set this attribute's value to True. Defaults to False.
        debug (bool, optional): If the user wishes to see a step by step output of this function, 
                                he needs to set this attribute's value to True. Defaults to False.

    Returns:
        (Torch.Tensor, Torch.Tensor, float): The data and targets tensor mixed up with the mixup percentage.
    """

    if(data.size(0) != targets.size(0)):
        print("The amount of data and labels are not the same !")
        return -1
    
    if(sameClasses):
        indices_by_labels = {}
        
        for i, el in enumerate(targets):
            ## Gets the value in the tensor:
            el = el.item()
            
            if indices_by_labels.get(el) is None:
                indices_by_labels[el] = [i]
            else:
                l = indices_by_labels[el]
                l.append(i)
                indices_by_labels[el] = l
                        
        indices = torch.zeros(targets.size(0), dtype = torch.long)
        
        for key in indices_by_labels.keys():
            
            initial_ids = np.array(indices_by_labels[key])
            
            perm        = np.random.permutation(initial_ids.size)
            
            new_ids     = initial_ids[perm]
            
            indices[torch.from_numpy(initial_ids)] = torch.from_numpy(new_ids)
            
        #print(targets == targets[indices])
        
    else:
        # Creates a random permutation for the data:
        indices = torch.randperm(data.size(0))


    # Creates mixed up data:
    data_mix    = data[indices]
    targets_mix = targets[indices]
    
    # Converts labels into one_hot encoded labels:
    targets     = onehot(targets, n_classes)    
    targets_mix = onehot(targets_mix, n_classes)
    
    # draws the mixup coefficient
    mixup_var = torch.FloatTensor(np.random.beta(beta_coef, beta_coef, data.size(0)))

    # Computes the percentage of data that are affected by the mixup:
    mixup_percentage = np.array([i for i in range (data.size(0))]) != np.array([el.item() for el in indices])
    
    for i, el in enumerate(mixup_var):
        mixup_percentage[i] = (mixup_percentage[i] and (el < 0.9999))

    mixup_percentage = sum(mixup_percentage)/data.size(0)
    
    if debug:
        print("mix_var shape %s" % str(mixup_var.shape))
        print("X shape %s" % str(data.shape))
        print("labels %s" % str(targets))
        print("mixup vars :", mixup_var)
        print("Indices :", indices)
        
    # applies mixup to both input data and one-hot encoded labels
    X = torch.rand(data.shape, dtype = torch.float)
    
    for i in range (data.size(0)):
        X[i] = data[i] * mixup_var[i] + data_mix[i] * (1. - mixup_var[i])
        
    y = torch.rand((data.size(0), n_classes), dtype = torch.float)
    
    for i in range (targets.size(0)):    
        y[i] = (targets[i] * mixup_var[i] + targets_mix[i] * (1. - mixup_var[i]))
    
    if debug:
        print("mix label shape %s" % str(y.shape))
        print("%s" % str(y[0]))
    
    return X, y, mixup_percentage


## <!>------------------------------------ Class Definitions: ------------------------------------<!> ##
## Dataset for train tensors:
class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, path, wlen, fact_amp, wshift=0, eval_mode = False, train = False):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.path = path
        
        ## SincNet's Window length:
        self.wlen     = wlen
        self.wshift   = wshift
        self.fact_amp = fact_amp
        self.train    = train
        
        ## <!>------------- For testloader or for evaluating dataset performance: -------------<!> ##
        ##Initializes the window:
        self.test_mode            = eval_mode
        self.beg_samp             = 0
        self.end_samp             = wlen
        self.valid_sample_number  = 0
        self.current_valid_sample = None 
        self.current_valid_label  = None
        
        ## Storing the number of samples:
        if(self.test_mode):
            self.number_of_samples = 0
            
            ## Go through all the files and gets their lenght:
            for el in self.list_IDs:
                ## Reads tensor with ID el:
                X = torch.load(self.path + el)#removed ".pt", it was already in my IDs
                y = self.labels[el]
                
                ## Number of frames extracted from tensors
                N_fr=int((X.shape[0]-wlen)/(wshift))
                
                ## Counter the number of total extracted frames:
                self.number_of_samples += N_fr
        else:
            self.number_of_samples = len(self.list_IDs)
            



  def __len__(self):
        'Denotes the total number of samples'
        return self.number_of_samples

  def __getitem__(self, index):
        'Generates one sample of data'
        ID = self.list_IDs[index]
        
        # Selecting sample
        if self.test_mode:
            ID = self.list_IDs[self.valid_sample_number]        
        
        
        ## Modifications for SincNet:
        if(self.test_mode):
            if(self.wshift == 0):
                print("Error, for validation set wshift can't be equal to 0!")
            
            ## Reinitializes the dataset if the sample number has exceeded the total amount of samples:
            if(self.valid_sample_number >= self.number_of_samples):
                self.valid_sample_number = 0
            
            ## Checks if we need to load a new sample:
            if(self.current_valid_sample is None or self.end_samp >= self.current_valid_sample.shape[0]):
                self.current_valid_sample = torch.load(self.path + ID)#remove ".pt", it was already in my IDs
                self.current_valid_label  = self.labels[ID]
                
                ## Increments the next sample number
                self.valid_sample_number += 1
                
                ##Initializes the window:
                self.beg_samp = 0
                self.end_samp = wlen
            
            X = self.current_valid_sample[self.beg_samp:self.end_samp]
            y = self.current_valid_label
            
            self.beg_samp += self.wshift
            self.end_samp  = self.beg_samp + self.wlen
            
            return X, y

        else:
            # Load data and get label 
            X = torch.load(self.path + ID)#remove ".pt", it was already in my IDs
            y = self.labels[ID]
            
            if(self.train):

                snt_len = X.shape[0]
                rand_amp = np.random.uniform(1.0-self.fact_amp,1+self.fact_amp)

                snt_beg = 0 if snt_len-self.wlen-1 in (0, -1) else np.random.randint(snt_len-self.wlen-1)
                snt_end = snt_beg + self.wlen

                return X[snt_beg:snt_end]*rand_amp, y
            else:
                return X, y

def NLLL_onehot(input, target, reduction= "mean"):
    
    loss = -torch.sum(input[:,:target.size(1)] * target.float())
    
    if "mean" in reduction:
        return loss / input.size(0)
    else:
        return loss


class NLLL_OneHot(object):
    def __init__(self, reduction= "mean"):
        self.reduction = reduction

    def __call__(self, input, target):
        return NLLL_onehot(input, target, self.reduction)

# Dummy class Created to regroup all optimizers into a single object:
class Optimizers(object):

    def __init__(self, optimizer_CNN, optimizer_DNN1, optimizer_DNN2):
        self.optimizer_CNN  = optimizer_CNN
        self.optimizer_DNN1 = optimizer_DNN1
        self.optimizer_DNN2 = optimizer_DNN2
    
    def step(self):
        self.optimizer_CNN.step()
        self.optimizer_DNN1.step()
        self.optimizer_DNN2.step()
        
    def zero_grad(self):
        self.optimizer_CNN.zero_grad() 
        self.optimizer_DNN1.zero_grad()
        self.optimizer_DNN2.zero_grad()