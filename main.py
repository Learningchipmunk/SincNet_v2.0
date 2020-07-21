"""
main.py
====================================
This is where the training occurs

How to run it:
python main.py --configPath=cfg/SincNet_DCASE_Preprocessing_WithEnergy_Window_800.cfg --cuda=0
"""

import os
import librosa 
import librosa.display
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import tqdm

## Local files imports:
from Models import MLP, flip
from Models import SincNet as CNN 
from read_conf_files import read_conf, str_to_bool
from utils import Optimizers, Dataset, plot_grad_flow, NLLL_OneHot, LoadPrevModel
from training_and_acc_fun import train, accuracy


## <!>---------------------------- Reading the config file ----------------------------<!> ##
# Reading cfg file
options=read_conf()

## Architecture of the file:
#[data]
tr_lst=options.tr_lst
te_lst=options.te_lst
pt_file=options.pt_file
class_dict_file=options.lab_dict
data_folder=options.data_folder
output_folder=options.output_folder

#[windowing]
fs=int(options.fs)
cw_len=int(options.cw_len)
cw_shift=int(options.cw_shift)

#[cnn]
cnn_N_filt=list(map(int, options.cnn_N_filt.split(',')))
cnn_len_filt=list(map(int, options.cnn_len_filt.split(',')))
cnn_max_pool_len=list(map(int, options.cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp=str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp=str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm=list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
cnn_use_batchnorm=list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
cnn_act=list(map(str, options.cnn_act.split(',')))
cnn_drop=list(map(float, options.cnn_drop.split(',')))


#[dnn]
fc_lay=list(map(int, options.fc_lay.split(',')))
fc_drop=list(map(float, options.fc_drop.split(',')))
fc_use_laynorm_inp=str_to_bool(options.fc_use_laynorm_inp)
fc_use_batchnorm_inp=str_to_bool(options.fc_use_batchnorm_inp)
fc_use_batchnorm=list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
fc_use_laynorm=list(map(str_to_bool, options.fc_use_laynorm.split(',')))
fc_act=list(map(str, options.fc_act.split(',')))

#[class]
class_lay=list(map(int, options.class_lay.split(',')))
class_drop=list(map(float, options.class_drop.split(',')))
class_use_laynorm_inp=str_to_bool(options.class_use_laynorm_inp)
class_use_batchnorm_inp=str_to_bool(options.class_use_batchnorm_inp)
class_use_batchnorm=list(map(str_to_bool, options.class_use_batchnorm.split(',')))
class_use_laynorm=list(map(str_to_bool, options.class_use_laynorm.split(',')))
class_act=list(map(str, options.class_act.split(',')))


#[optimization]
lr=float(options.lr)
batch_size=int(options.batch_size)
Batch_dev=int(options.Batch_dev)
patience=int(options.patience)
N_epochs=int(options.N_epochs)
N_batches=int(options.N_batches)
N_eval_epoch=int(options.N_eval_epoch)
train_acc_period=int(options.train_acc_period)
use_mixup=str_to_bool(options.use_mixup)
beta_coef=float(options.beta_coef)
mixup_batch_prop=float(options.mixup_batch_prop)
## same_classes has default value False:
same_classes=False
if(options.same_classes!=None):
    same_classes=str_to_bool(options.same_classes)
else:
    ## If we are using mixup without specifying which type, it alerts the user.
    if(use_mixup):
        print("Warning: you are using mixup but you did not mention which type in config file. \n"+
              "By default it will be set to False. You are advised to add a same_class attribute to your cfg file and set it to True or False.")    
seed=int(options.seed)


## The location of all the wav files are stored here:
# training list
tensors_lst_tr = np.load(tr_lst)
snt_tr=len(tensors_lst_tr)

# test list
tensors_lst_te = np.load(te_lst)
snt_te=len(tensors_lst_te)


# Folder creation
try:
    os.stat(output_folder)
except:
    os.mkdir(output_folder) 
    
    
# setting seed
torch.manual_seed(seed)
np.random.seed(seed)

# loss functions
cost = nn.NLLLoss()
cost_onehot = NLLL_OneHot()
  
# Converting context and shift in samples
wlen=int(fs*cw_len/1000.00)
wshift=int(fs*cw_shift/1000.00)

## Setting cuda Device
print("Selecting Cuda device... \t\t", end="")
Desired_cuda_device_number = int(options.cuda)

if torch.cuda.is_available(): # we'll use cuda
    device = "cuda:"+str(Desired_cuda_device_number)
    torch.cuda.set_device(device)
    if(torch.cuda.current_device() == Desired_cuda_device_number and torch.cuda.is_available()):
        print("Cuda device {} was selected successfully!".format(Desired_cuda_device_number))
    else:
        print("Cuda was not selected successfully...")
else:
    print("Cuda device(s) is(are) not available.")


## <!>------------------- Initializing the Networks with .cfg options -------------------<!> ##

print("Initializing the Networks... \t\t", end="")
# Feature extractor CNN
CNN_arch = {'input_dim': wlen,
          'fs': fs,
          'cnn_N_filt': cnn_N_filt,
          'cnn_len_filt': cnn_len_filt,
          'cnn_max_pool_len':cnn_max_pool_len,
          'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
          'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
          'cnn_use_laynorm':cnn_use_laynorm,
          'cnn_use_batchnorm':cnn_use_batchnorm,
          'cnn_act': cnn_act,
          'cnn_drop':cnn_drop,          
          }

## Initializes SincNet:
CNN_net=CNN(CNN_arch)
CNN_net.cuda()

# Loading label dictionary
lab_dict=np.load(class_dict_file).item()


## First DNN, follows the config from the section [dnn] in .cfg file
DNN1_arch = {'input_dim': CNN_net.out_dim,
          'fc_lay': fc_lay,
          'fc_drop': fc_drop, 
          'fc_use_batchnorm': fc_use_batchnorm,
          'fc_use_laynorm': fc_use_laynorm,
          'fc_use_laynorm_inp': fc_use_laynorm_inp,
          'fc_use_batchnorm_inp':fc_use_batchnorm_inp,
          'fc_act': fc_act,
          }

DNN1_net=MLP(DNN1_arch)
DNN1_net.cuda()

## Last trainable layer, has softmax as activation function see section [class] in .cfg
DNN2_arch = {'input_dim':fc_lay[-1] ,
          'fc_lay': class_lay,
          'fc_drop': class_drop, 
          'fc_use_batchnorm': class_use_batchnorm,
          'fc_use_laynorm': class_use_laynorm,
          'fc_use_laynorm_inp': class_use_laynorm_inp,
          'fc_use_batchnorm_inp':class_use_batchnorm_inp,
          'fc_act': class_act,
          }


DNN2_net=MLP(DNN2_arch)
DNN2_net.cuda()



print("Initialization done!")

## <!>---------------------------- Initializing optimizers ----------------------------<!>
## Uses by default RMSprop optimization
# Maybe use adam Like Paul ??? Faire des tests (chapitre 8 paragraphe 5) + SGD

print("Initializing optimizers... \t\t", end="")
# Added momentum:
momentum = 0.9 if use_mixup else 0
    
optimizer_CNN  = optim.RMSprop(CNN_net.parameters(), lr=lr,alpha=0.95, eps=1e-8, momentum=momentum) 
optimizer_DNN1 = optim.RMSprop(DNN1_net.parameters(), lr=lr,alpha=0.95, eps=1e-8, momentum=momentum) 
optimizer_DNN2 = optim.RMSprop(DNN2_net.parameters(), lr=lr,alpha=0.95, eps=1e-8, momentum=momentum) 

optimizers = Optimizers(optimizer_CNN, optimizer_DNN1, optimizer_DNN2)
print("Optimizers are ready!")


print("Creating the datasets... \t\t", end="")
## Creating the datasets:
train_dataset = Dataset(tensors_lst_tr, lab_dict, data_folder, wlen, 0.2, wshift = 0, eval_mode= False, train = True)
valid_dataset = Dataset(tensors_lst_te, lab_dict, data_folder, wlen, 0.2, wshift = wshift, eval_mode= False, train = False)
print("Done!")


print("Setting up the data loaders... \t\t", end="")
## Setting up the loaders:
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

## Batchsize can only be 1 for valid_loader because each tensor has a different shape...
valid_loader  = torch.utils.data.DataLoader(dataset=valid_dataset,
                                          batch_size=1,
                                          shuffle=False)
print("Done!")


# python main.py --configPath=cfg/SincNet_DCASE_EnergyPre1000_Window800_withMixup_Drop30.cfg --cuda=0
# python main.py --configPath=cfg/SincNet_DCASE_Rand0Pre_WithEnergy_Window_800.cfg --cuda=1 
# nohup python main.py --configPath=cfg/test.cfg --cuda=1 &
# nohup python main.py --configPath=cfg/test.cfg --cuda=0 &
# nohup python main.py --configPath=cfg/SincNet_DCASE_Rand0Pre_WithEnergy_Window_800_PReLu_Drop30.cfg --cuda=1 & 
# python main.py --configPath=cfg/SincNet_DCASE_Rand0Pre_WithEnergy_Window_800_PReLu.cfg --cuda=0
# python main.py --configPath=cfg/SincNet_DCASE_Rand0Pre_WithEnergy_Window_800_HiddenLay4_PReLu.cfg --cuda=1
## Parameters that needs to change each execution:
Training_model_file   = output_folder.split("/")[-2] if output_folder.split("/")[-1]=="" else output_folder.split("/")[-1]
Models_file_extension = ".pkl" if pt_file == 'none' else pt_file.split(".")[1]
previous_model_path   = output_folder+ '/' + Training_model_file if pt_file == 'none' else pt_file.split(".")[0]
Load_previous_model   = False if pt_file == 'none' else True
previous_epoch        = 0
inTheSameFile         = False
plotGrad              = False
compute_matrix        = False
n_classes             = class_lay[-1]#41 for SincNet
same_classes          = same_classes


## are in cfg:
#beta_coef=0.4
#use_mixup
#N_eval_epoch = 1
#same_classes = True


## Loading previously trained model if needed:
if(Load_previous_model):
    CNN_net, DNN1_net, DNN2_net, previous_epoch = LoadPrevModel(CNN_net, DNN1_net, DNN2_net,
                                                            previous_model_path, 
                                                            Models_file_extension, 
                                                            Load= Load_previous_model, 
                                                            inSameFile = inTheSameFile,
                                                            at_epoch = previous_epoch,
                                                            evalMode = False)


## Training parameters available in the SincNet_TIMIT.cfg file section [optimization]:
# batch_size
# N_epochs     = 1500
# N_batches    = 800
# N_eval_epoch = 8

## Overwriting N_epochs just bcz flemme:
#N_epochs = 58



train(CNN_net, DNN1_net, DNN2_net, optimizers, train_loader, valid_loader, cost, cost_onehot,
          ## SincNet variables:
          wlen,
          wshift,
          n_classes,
          ## File variables:
          output_folder,
          Training_model_file,
          Models_file_extension,
          ## Hyper param:
          n_epoch = N_epochs,
          patience = patience,
          Batch_dev = Batch_dev,#Number of batches for testing set
          train_acc_period = train_acc_period,
          test_acc_period = N_eval_epoch,
          ## For Mixup
          beta_coef = beta_coef,
          mixup_batch_prop = mixup_batch_prop,
          use_mixup = use_mixup,
          same_classes = same_classes,
          ## Loaded model params:
          starting_epoch = previous_epoch,
          ## Tracking gradient
          plotGrad = plotGrad,
          ## If user wishes to save and compute confusion matrix:
          compute_matrix = compute_matrix,
          cuda=True)
