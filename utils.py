
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
from matplotlib.lines import Line2D

## Local files imports:
from ipython_exit import exit


## Function that raises excepetion and exists Jupyter Notebook's cell:
def test_2D_raise_or_run(is_conv2D, message = "The current Network is not `SincNet2D` therefore the code won't be able to run! \nExited cell safely."):
    """Function that raises excepetion and exists Jupyter Notebook's cell id the network is not `SincNet2D`.

    Args:
        is_conv2D (bool): True indicates that the Network is indeed `SincNet2D`. False indicates otherwise.
        message (str, optional): Exception message that will be shown to the user. Defaults to "The current Network is not `SincNet2D` therefore the code won't be able to run! \nExited cell safely.".

    Raises:
        Exception: Shows to the user the error message.
    """    
    if not is_conv2D:
        # Announces to the user that we are exiting the cell.
        print("Raising exception and exiting cell.")

        # Raises the Exception
        raise Exception(message)
        
        # Exits the cell
        exit()


## Function that reads the .res file:
def readResults(path):
    """This function reads the .res file of a network previously trained.

    Args:
        path (str): Is the path of the directory conatined the .res file.
        filename (str): Name of the .res file to read.

    Returns:
        (list): Returns the values inside the .res file with their associated epoch number.
    """

    ## Opens the .res
    f = open(path, "r")
    
    ## Initializes the list:
    perfs = []

    ## Reads the file line by line:
    lines = f.readlines()

    for el in lines:

        values = []
        elements = el.split(" ")

        ## Processes the rest of the line here:
        elements = elements[1:]
        for i, val in enumerate(elements):
            if("\n" not in val):
                ## After the epoch number there is a comma, we use it as a seperator to get the epoch number:
                if i ==0:
                    values.append(int(val.split(",")[0]))
                ## now each number is preceded by an equal sign, that is why we use it as a seperator:
                else:
                    values.append(float(val.split("=")[1]))

        perfs.append(values)

    f.close()
    return perfs


## Function that returns the right optimizer initialized:
#Added to initialize optimizers according to user's demand:
def InitOptimizer(Optimizer_type, parameters, lr = None, momentum = None):
    """Functions that returns the right optimizer initialized.

    Args:
        Optimizer_type (str): Name of the optimizer the user wishes to use.
        parameters (Module.parameters): The network's parameters we need to optimize.
        lr (float, optional): The optimizer's learning rate. Defaults to None.
        momentum (float, optional): The optimizer's momentum. Defaults to None.

    Returns:
        torch.optim: Returns the optimizer initialized with the parameters.
    """
    Optimizer_type = Optimizer_type.lower()

    if "rmsprop" in Optimizer_type:
        
        if lr is None:
            lr = 0.001
        
        if momentum is None:
            momentum = 0

        return torch.optim.RMSprop(parameters, lr=lr, alpha=0.95, eps=1e-8, momentum=momentum) 
    
    elif "adamax" in Optimizer_type:
        
        if lr == None:
            lr=0.002
        
        return torch.optim.Adamax(parameters, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    
    elif "adam" in Optimizer_type:
        
        if lr == None:
            lr = 0.001
        
        return torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    else:
        print("Error: `optimizer_type` was not used properly. Please use one of these values [`rmsprop`, `adamax`, `adam`]. \nNote that the code is not case sensitive to `optimizer_type`, RmSpRoP will be understood.")
        exit()


def determine_mode(Scheduler_type, is_lower_case=False):
    """Function that determines if the user wishes to use a CyclicLR scheduler.

    Args:
        Scheduler_type (str): A string that indicates the type of scheduler the user wants.
        is_lower_case (bool): True indicates that Scheduler_type is already written in lower case.

    Returns:
        (str): Returns the value in string of the mode the user wished to use.
    """
    if not is_lower_case:
        Scheduler_type = Scheduler_type.lower()


    """
    “triangular”: A basic triangular cycle without amplitude scaling.

    “triangular2”: A basic triangular cycle that scales initial amplitude by half each cycle.

    “exp_range”: A cycle that scales initial amplitude by gamma^cycle_iterations at each cycle iteration.
    """
    if "triangular2" in Scheduler_type:       
        return "triangular2"

    elif "triangular" in Scheduler_type:
        return "triangular"    
    
    elif "exp_range" in Scheduler_type:
        return "exp_range"
    
    else:
        print("Warning: You did not dpecify CyclicLR mode. By default, CyclicLR is initialized with `triangular2`.")
        return "triangular2"



## Function that returns the right scheduler initialized:
#Added to initialize schedulers according to user's demand:
def InitScheduler(Scheduler_type, optimizer, lr = 0.001, scheduler_patience = 2, scheduler_factor = 0.5, step_size_up = 2000, verbose = False):
    """Function that returns the right scheduler initialized.

    Args:
        Scheduler_type (str): Name of the scheduler the user wishes to use.
        optimizer (torch.optim): The network's parameters we need to optimize.
        lr (float, optional): The optimizer's learning rate. Defaults to 0.001.
        scheduler_patience (int, optional): Number of epochs with no improvement after which learning rate will be reduced. Defaults to 2.
        scheduler_factor (float, optional): Is the amount the Learning rate will be reduced if scheduler patience is exceeded. Defaults to 0.5.
        step_size_up (int, optional): Number of training iterations in the increasing half of a cycle. Defaults to 2000.
        verbose (bool, optional): If True, prints a message to stdout for each update. Defaults to False.

    Returns:
        torch.optim: Returns the scheduler initialized with the parameters.
    """    
    Scheduler_type = Scheduler_type.lower()
    is_lower_case  = True

    if "cycliclr" in Scheduler_type:
        
        """ Quoting Cyclical Learning Rates for Training Neural Networks by Leslie N. Smith:
        Alternatively, one can use the rule of thumb that the optimum learning rate is usually within a
        factor of two of the largest one that converges [2] and set base lr to 1/3 or 1/4 of max lr.
        """
        max_lr  = lr
        base_lr = max_lr/4

        ## Determining which mode user wishes to use:
        mode = determine_mode(Scheduler_type, is_lower_case)
        
        return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=step_size_up, step_size_down=None, mode=mode, gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1) 
    
    elif "reducelronplateau" in Scheduler_type:

        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=verbose, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    
    else:
        print("Error: `scheduler_type` was not used properly. Please use one of these values  [`ReduceLROnPlateau`, `CyclicLR_triangular`, `CyclicLR_triangular2`, `CyclicLR_exp_range`]. \nNote that the code is not case sensitive to `scheduler_type`, reduceLRONPlaTeau will be understood.")
        exit()

## Loads previously trained model:
#Function was modified especially for .py folders:
def LoadPrevModel(Main_net, CNN_net, DNN1_net, DNN2_net, model_file_path, Models_file_extension, Load, inSameFile = True, test_acc_period = 5, at_epoch = 0, evalMode = False):
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
        test_acc_period (int, optional): The period, in epoch, of each validation test. Defaults to 5.
        at_epoch (int, optional): The amount of epoch the model was trained on. Defaults to N_epochs.

    Returns:
        (nn.Module(x4), int, float): The current starting epoch for the loaded model and the min_loss achieved by the model. 
    """
    
    if(Load == False):
        return Main_net, CNN_net, DNN1_net, DNN2_net, 0, float("inf")
    
    ## <!>----------------- Reading old results: <!>----------------- ##
    perfs    = readResults(model_file_path + ".res")
    at_epoch = len(perfs)
    min_loss = min([perfs[i][4] for i in range (0, at_epoch)])
    at_epoch*= test_acc_period    
    
    ## <!>------------ Fetching and storing old model: ------------<!> ##
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

    ## Putting all the nets into Main_net:
    Main_net.CNN_net  = CNN_net
    Main_net.DNN1_net = DNN1_net
    Main_net.DNN2_net = DNN2_net
    
    if(evalMode):Main_net.eval()

    
    print("Models from " + model_file_path + " were loaded successfully!")
    print("Model reached a minimum loss of {0} after {1} epochs.".format(min_loss, at_epoch))
    
    return Main_net, CNN_net, DNN1_net, DNN2_net, at_epoch, min_loss




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

## Dataset that loads tensors:
class Dataset(torch.utils.data.Dataset):
    """
    Characterizes a dataset for PyTorch, it loads chunks of data for training and accuracy functions.
    """
    def __init__(self, list_IDs, labels, path, wlen, fact_amp=0, wshift=0, using_mixup=False, beta_coef=0.4, mixup_prop=1, sameClasses = False, train = False, is_fastai = False):
        """Initialization of the DataSet.

        Args:
            list_IDs (list): A list of strings containing the names of the tensors to load.
            labels (dict): A dictionary containing the labels corresponding to each tensor ID. 
                           Usage: self.labels[ID] returns int (ground truth).
            path (string): Directory location of the tensors.
            wlen (int): Size of the window of each audio data returned. (Basically is the size of the tensor returned.)
            fact_amp (float): Each tensor is amplified randomly with a factor of np.random.uniform(1.0-self.fact_amp,1+self.fact_amp).
            wshift (int, optional): Is the Hop size in between valid/test chunks. Defaults to 0.
            using_mixup (bool, optional): Indicates if user wishes to use mixup for training, it must be se to True if so. Defaults to False.
            beta_coef (float, optional): Is the coeficient for the Bet distribution used for mixup. Defaults to 0.4.
            mixup_prop (int, optional): Is the probability that the data gets mixed up. Parameter of a variable following a uniform distribution. 
                                        Defaults to 1.
            sameClasses (bool, optional): Indicates if the user wishes to mixup only data with same corresponding label, it must be se to True if so.
                                          Defaults to False.
            train (bool, optional): Indicates if user wishes to create a training dataset, it must be se to True if so. Defaults to False.
            is_fastai (bool, optional): Indicates if user wishes to use fastai, it must be se to True if so. Defaults to False.
        """
        self.labels = labels
        self.list_IDs = list_IDs
        self.path = path
        
        ## SincNet's Window length:
        self.wlen     = wlen
        self.wshift   = wshift
        self.fact_amp = fact_amp
        self.train    = train
        
        ## Mixup variables:
        self.using_mixup = using_mixup
        self.beta_coef   = beta_coef
        self.mixup_prop  = mixup_prop
        self.sameClasses = sameClasses
        
        ## for fastai:
        self.is_fastai   = is_fastai 
        
        ## Initializes the dictionary of the indicies of each tensor grouped by class:
        self.tensor_by_class_dict = {}
        
        ## <!>------------- For testloader or valid loader: -------------<!> ##
        if(not self.train):
            self.list_IDs_chunks = []
        
        ## Stores the number of classes:
        self.n_classes = 0
            
        ## Go through all the files and gets their lenght:
        for i, el in enumerate(self.list_IDs):
            ## Reads tensor with ID el:
            X = torch.load(self.path + el)#removed ".pt", it was already in my IDs
            y = self.labels[el]
            
            
            if(not self.train):
                ## Number of frames extracted from tensors
                N_fr = int((X.shape[0]-wlen)/(wshift)) + 1

                ## Appends the (tensor idenx in list_ID, chunk number) to the list of IDs and chunks
                self.list_IDs_chunks += [(i, j) for j in range(0, N_fr)]
            
            ## Stores the indices of tensors in tensor list by class in the dict:
            if self.tensor_by_class_dict.get(y) is None:
                ## Adds the new element as a singleton in a list:
                self.tensor_by_class_dict[y] = [i]

                ## Updates the number of classes:
                self.n_classes              += 1
            else:
                l = self.tensor_by_class_dict[y]
                l.append(i)
                self.tensor_by_class_dict[y] = l
                

        
        ## Sets the size of dataset depending on the type:
        if self.train:
            self.number_of_samples = len(self.list_IDs)
        else:
            self.number_of_samples = len(self.list_IDs_chunks)
            

    def onehot(self, label, n_classes):
        """Returns a one hot encoded version of the labels.

        Args:
            label (Torch.Tensor): A tensor of type long wich describes the labels as integers.
            n_classes (int): The number of classes.

        Returns:
            (Torch.Tensor):  One hot encoded labels.
        """
        onehot_label = torch.zeros(n_classes)
        onehot_label[label] = 1
        return onehot_label

    def __len__(self):
        'Denotes the total number of samples'
        return self.number_of_samples
    
    def __getitem__(self, index):
        'Generates one sample of data'
        
        # Selecting sample
        if self.train:
            ID                       = self.list_IDs[index]
        else:
            tensor_idx, chunk_number = self.list_IDs_chunks[index]
            ID                       = self.list_IDs[tensor_idx]
         
        # Load data and its label 
        X = torch.load(self.path + ID)#remove ".pt", it was already in my IDs
        y = self.labels[ID]

        
        ## Modifications for SincNet:
        if(not self.train):
            if(self.wshift == 0):
                print("Error, for validation or test set wshift can't be equal to 0!")
            
            ##Initializes the window accordingly to chunk number:
            beg_samp = chunk_number * self.wshift
            end_samp = beg_samp + self.wlen
            
            if self.is_fastai:
                return X[beg_samp:end_samp], y
            else:
                return X[beg_samp:end_samp], y, tensor_idx


        else:            
            ## Stores the lenght of the signal:
            snt_len = X.shape[0]

            ## Randomly amplifies the signal by rand_amp factor
            rand_amp = np.random.uniform(1.0-self.fact_amp,1+self.fact_amp)

            if(snt_len < self.wlen):
                print("Error, file too small! Size is equal to {0} and should be at least {1}.".format(snt_len, self.wlen))
            ## Chooses a random chunk of the signal:
            snt_beg = 0 if snt_len-self.wlen-1 in (0, -1) else np.random.randint(snt_len-self.wlen-1)
            snt_end = snt_beg + self.wlen
            
            ## Creates the random chunk:
            data = X[snt_beg:snt_end]
            
            ## Boolean that indicates if we used mixup for this sample of data:
            used_mixup = False
            
            ## Rand variable That decides if we mixup or not:
            uniform01 = np.random.uniform()
            
            if self.using_mixup:
                ## Gets the data for mixup:
                if self.sameClasses:
                    data_mix, y2 = self.get_item_randomly(y)
                else:
                    data_mix, y2 = self.get_item_randomly()
                    
                # Converts labels into one_hot encoded labels:
                y  = self.onehot(y,  self.n_classes)    
                y2 = self.onehot(y2, self.n_classes)
            
                ## Gets the mixup coef:
                mixup_var = np.random.beta(self.beta_coef, self.beta_coef)
                
                ## After reading an article, it seems best to take coefs > 0.5
                if mixup_var <= 0.5:
                    mixup_var = 1 - mixup_var
                
                ## Prop
                if uniform01 <= self.mixup_prop:
                    ## Mixing up data:
                    data = data * mixup_var + data_mix * (1. - mixup_var)

                    ## Mixing labels:
                    y    = y    * mixup_var + y2       * (1. - mixup_var)

                    ## States that we used mixup:
                    used_mixup = True
                

            ## Amplifies the signal:
            data *= rand_amp
            
            ## Returns a random amplified chunk of the signal (can be mixed up):
            if(self.using_mixup):
                return data, y , used_mixup
            else :
                return data, y
    

    def get_item_randomly(self, lab = -1):
        if(not self.train):
            print("Warning, user should not use this function for test loaders or valid loaders.")
        
        ## Different 
        if lab != -1 :
            ## Gets the total number of tensors labeled lab:
            N = len(self.tensor_by_class_dict[lab])  
        else:
            ## Gets the total number of tensors:
            N = len(self.list_IDs)
        
        ## Chooses randomly one tensor:
        idx = np.random.randint(0, N)
        
        ## Stores the index of the tensor:
        idx_tensor = self.tensor_by_class_dict[lab][idx] if lab != -1 else idx
        
        ## Stores the ID of the tensor:
        ID = self.list_IDs[idx_tensor]
        
        ## Loads the tensor and its label:
        X = torch.load(self.path + ID)#removed ".pt", it was already in my IDs
        y = self.labels[ID]
            
        ## Stores the length of the signal:
        snt_len = X.shape[0]
        
        ## Chooses a random chunk of the signal:
        snt_beg = 0 if snt_len-self.wlen-1 in (0, -1) else np.random.randint(snt_len-self.wlen-1)
        snt_end = snt_beg + self.wlen

        ## Returns the random chunk of the signal:
        return X[snt_beg:snt_end], y


## Naive dataset for train tensors:
class Dataset2(torch.utils.data.Dataset):
  """
  Characterizes a dataset for PyTorch, it loads chunks of data for training and accuracy.
  """
  def __init__(self, list_IDs, labels, path, wlen, fact_amp, wshift=0, train = False):
        """Initialization of the DataSet.

        Args:
            list_IDs (list): A list of strings containing the names of the tensors to load.
            labels (dict): A dictionary containing the labels corresponding to each tensor ID. 
                           Usage: self.labels[ID] returns int (ground truth).
            path (string): Directory location of the tensors.
            wlen (int): Size of the window of each audio data returned. (Basically is the size of the tensor returned.)
            fact_amp (float): Each tensor is amplified randomly with a factor of np.random.uniform(1.0-self.fact_amp,1+self.fact_amp).
            wshift (int, optional): Is the Hop size in between valid/test chunks. Defaults to 0.
            train (bool, optional): Indicates if user wishes to create a training dataset, it must be se to True if so. Defaults to False.
        """
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
        self.beg_samp             = 0
        self.end_samp             = wlen
        self.valid_sample_number  = 0
        self.current_valid_sample = None 
        self.current_valid_label  = None

        ## Stores the number of samples:
        self.number_of_samples = len(self.list_IDs)


  def __len__(self):
        'Denotes the total number of samples'
        return self.number_of_samples

  def __getitem__(self, index):
        'Generates one sample of data'
        ID = self.list_IDs[index]
        
        ## Modifications for SincNet:
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


# Dummy class Created to regroup all schedulers into a single object:
class Schedulers(object):
    
    def __init__(self, scheduler_CNN, scheduler_DNN1, scheduler_DNN2):
        self.scheduler_CNN  = scheduler_CNN
        self.scheduler_DNN1 = scheduler_DNN1
        self.scheduler_DNN2 = scheduler_DNN2
        
    def step(self, metric = None):
        if metric is not None:
            self.scheduler_CNN.step(metric)
            self.scheduler_DNN1.step(metric)
            self.scheduler_DNN2.step(metric)
        else:
            self.scheduler_CNN.step()
            self.scheduler_DNN1.step()
            self.scheduler_DNN2.step()
    
# Dummy class Created for options object to replace OptionParser's object
class Options(object):

    def __init__(self, cfg):
        '''Defines the cfg file'''
        self.configPath = cfg