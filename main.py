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
from Models import MLP, flip, MainNet
from Models import SincNet as CNN 
from Models import SincNet2D as CNN2D
from read_conf_files import read_conf, str_to_bool
from utils import Optimizers, Schedulers, Dataset, plot_grad_flow, NLLL_OneHot, LoadPrevModel, InitOptimizer
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

#[cnn2D/1D]
is_conv2D = options.is_conv2D
conv_type = '2D' if is_conv2D else '1D'
print("The file contains the config of a {} convolutional network.".format(conv_type))
if is_conv2D:
    #[cnn2D]
    cnn_N_filt=list(map(int, options.cnn_N_filt.split(',')))
    cnn_len_filt_W=list(map(int, options.cnn_len_filt_W.split(',')))
    cnn_len_filt_H=list(map(int, options.cnn_len_filt_H.split(',')))
    cnn_energy_L=int(options.cnn_energy_L)
    cnn_energy_stride=int(options.cnn_energy_stride)
    cnn_max_pool_len_W=list(map(int, options.cnn_max_pool_len_W.split(',')))
    cnn_max_pool_len_H=list(map(int, options.cnn_max_pool_len_H.split(',')))
else:
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
optimizer_type=str(options.optimizer_type)
lr=float(options.lr)
use_scheduler = str_to_bool(options.use_scheduler)
scheduler_type = str(options.scheduler_type)
scheduler_patience = int(options.scheduler_patience)
scheduler_factor = float(options.scheduler_factor)
batch_size=int(options.batch_size)
Batch_dev=int(options.Batch_dev)
patience=int(options.patience)
N_epochs=int(options.N_epochs)
N_batches=int(options.N_batches)
N_eval_epoch=int(options.N_eval_epoch)
train_acc_period=int(options.train_acc_period)
fact_amp=float(options.fact_amp)
use_mixup=str_to_bool(options.use_mixup)
beta_coef=float(options.beta_coef)
mixup_batch_prop=float(options.mixup_batch_prop)
same_classes=str_to_bool(options.same_classes)
seed=int(options.seed)

#[Misc]
use_SincConv_fast = str_to_bool(options.use_SincConv_fast)


## The location of all the wav files are stored here:
# training list
tensors_lst_tr = np.load(tr_lst)
snt_tr=len(tensors_lst_tr)

# test list
tensors_lst_te = np.load(te_lst)
snt_te=len(tensors_lst_te)


# Folder creation
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
    
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

if(Desired_cuda_device_number == -1):
    device = "cpu"
    torch.cuda.set_device(device)
    print("CPU was selected successfully!")
elif torch.cuda.is_available(): # we'll use cuda
    device = "cuda:"+str(Desired_cuda_device_number)
    torch.cuda.set_device(device)
    if(torch.cuda.current_device() == Desired_cuda_device_number and torch.cuda.is_available()):
        print("Cuda device {} was selected successfully!".format(Desired_cuda_device_number))
    else:
        print("Cuda was not selected successfully...")
else:
    device = "cpu"
    torch.cuda.set_device(device)
    print("Cuda device(s) is(are) not available... \t We selected CPU instead!")


## <!>------------------- Initializing the Networks with .cfg options -------------------<!> ##

print("Initializing the Networks... \t\t", end="")

# Feature extractor CNN
if is_conv2D:
    CNN_arch = {'input_dim': wlen,
            'fs': fs,
            'cnn_N_filt': cnn_N_filt,
            'cnn_len_filt_W': cnn_len_filt_W,
            'cnn_len_filt_H': cnn_len_filt_H,
            'cnn_energy_L': cnn_energy_L,
            'cnn_energy_stride': cnn_energy_stride,
            'cnn_max_pool_len_W': cnn_max_pool_len_W,
            'cnn_max_pool_len_H': cnn_max_pool_len_H,
            'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
            'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
            'cnn_use_laynorm':cnn_use_laynorm,
            'cnn_use_batchnorm':cnn_use_batchnorm,
            'cnn_act': cnn_act,
            'cnn_drop':cnn_drop,
            'use_SincConv_fast': use_SincConv_fast,          
            }
else:
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
            'use_SincConv_fast': use_SincConv_fast,                      
            }

## Initializes SincNet:
CNN_net=CNN2D(CNN_arch) if is_conv2D else CNN(CNN_arch)
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

## Last trainable layer, The classifier. Has logsoftmax as activation function for classification see section [class] in .cfg
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

## Network that regroups all 3 networks:
Main_net = MainNet(CNN_net, DNN1_net, DNN2_net)
Main_net.cuda()


print("Initialization done!")

## <!>---------------------------- Initializing optimizers ----------------------------<!>
## Uses by default RMSprop optimization

print("Initializing optimizers... \t\t", end="")
# Added momentum:
momentum = 0.9 if use_mixup else 0
    
# InitOptimizers
optimizer_CNN  = InitOptimizer(optimizer_type, CNN_net.parameters(), lr=lr, momentum=momentum)#optim.RMSprop(CNN_net.parameters(), lr=lr,alpha=0.95, eps=1e-8, momentum=momentum) 
optimizer_DNN1 = InitOptimizer(optimizer_type, DNN1_net.parameters(), lr=lr, momentum=momentum)#optim.RMSprop(DNN1_net.parameters(), lr=lr,alpha=0.95, eps=1e-8, momentum=momentum) 
optimizer_DNN2 = InitOptimizer(optimizer_type, DNN2_net.parameters(), lr=lr, momentum=momentum)#optim.RMSprop(DNN2_net.parameters(), lr=lr,alpha=0.95, eps=1e-8, momentum=momentum) 

# Puts the in the same wrapper:
optimizers = Optimizers(optimizer_CNN, optimizer_DNN1, optimizer_DNN2)

print("{} optimizers are ready!".format(optimizer_type))


## Initializing all schedulers for optims:
print("Initializing schedulers... \t\t", end="")
scheduler_CNN  = optim.lr_scheduler.ReduceLROnPlateau(optimizer_CNN, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
scheduler_DNN1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_DNN1, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
scheduler_DNN2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_DNN2, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

schedulers = Schedulers(scheduler_CNN, scheduler_DNN1, scheduler_DNN2)
#torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1, verbose=False)
print("{} schedulers are ready!".format(scheduler_type))


print("Creating the datasets... \t\t", end="")
## Creating the datasets:
train_dataset      = Dataset(tensors_lst_tr, lab_dict, data_folder, wlen, fact_amp = fact_amp, wshift = 0, using_mixup=use_mixup, beta_coef=beta_coef, mixup_prop=mixup_batch_prop, sameClasses = same_classes, train = True, is_fastai=False)
valid_dataset      = Dataset(tensors_lst_te, lab_dict, data_folder, wlen, fact_amp = 0, wshift = wshift, train = False, is_fastai=False)
print("Done!")


print("Setting up the data loaders... \t\t", end="")
## Setting up the loaders:
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

## Batchsize can only be 1 for valid_loader because each tensor has a different shape...
valid_loader  = torch.utils.data.DataLoader(dataset=valid_dataset,
                                          batch_size=Batch_dev,
                                          shuffle=False)
print("Done!")


# python main.py --configPath=cfg/SincNet_DCASE_EnergyPre1000_Window800_withMixup_Drop30.cfg --cuda=0
# python main.py --configPath=cfg/SincNet_DCASE_Rand0Pre_WithEnergy_Window_800.cfg --cuda=1 
# nohup python main.py --configPath=cfg/test.cfg --cuda=1 &
# nohup python main.py --configPath=cfg/test.cfg --cuda=0 &
# nohup python main.py --configPath=cfg/SincNet_DCASE_Rand0Pre_WithEnergy_Window_800_PReLu_Drop30.cfg --cuda=1 & 
# python main.py --configPath=cfg/SincNet_DCASE_Rand0Pre_WithEnergy_Window_800_PReLu.cfg --cuda=0
# python main.py --configPath=cfg/SincNet_DCASE_Rand0Pre_WithEnergy_Window_800_HiddenLay4_PReLu.cfg --cuda=1
# python main.py --configPath=cfg/SincNet_DCASE_RedimCNN_Rand0Pre_WithEnergy_Window_800_PReLu_Drop30.cfg --cuda=1 
# python main.py --configPath=cfg/SincNet_DCASE_CNNLay4_Rand0PreEnergyWindow800_Scheduler_PReLu_Drop30.cfg --FileName=CNNlay4_Rand0PreEnergy1000ms_Scheduler_Window800ms_PReLu_Drop30_try2 --cuda=1 
# python main.py --configPath=cfg/SincNet_DCASE_CNNLay4_Rand0Pre_EnergyPre1000_Window800_PReLu_withMixup.cfg --cuda=0
# python main.py --configPath=cfg/SincNet_DCASE_CNNLay6_Rand0Pre_WithEnergy_Window3000_PReLu_Drop30.cfg --FileName=CNNlay6_Rand0PreEnergy1000ms_Scheduler_Window3000ms_PReLu_Drop30_try2 --cuda=0 
# nohup python main.py --configPath=cfg/SincNet_DCASE_CNNLay6_DNN1024_Rand0Pre_WithEnergy_Window3000_PReLu_Drop30.cfg --FileName=CNNlay6_DNN1024_Rand0PreEnergy4000ms_Scheduler_Window3000ms_PReLu_Drop30_try2 --cuda=0 &
# nohup python main.py --configPath=cfg/SincNet_DCASE_CNNLay4_Rand0PreEnergyWindow800_Scheduler_PReLu_Drop30.cfg --FileName=CNNlay4_Rand0PreEnergy1000ms_Schedulerfact0.2_Window800ms_PReLu_Drop30 --cuda=1 &
# nohup python main.py --configPath=cfg/SincNet2D/SincNet2D_CNNLay4_Rand0PreEnergyWindow3000_Scheduler_PReLu_Drop30.cfg --cuda=0 &
# nohup python main.py --configPath=cfg/SincNet2D/SincNet2D_CNNLay5_Rand0PreEnergyWindow800_Scheduler_PReLu_Drop30.cfg --cuda=0 &

# nohup python main.py --configPath=cfg/SincNet_DCASE_CNNLay4_Rand0PreEnergyWindow800_Scheduler_PReLu_Drop30.cfg --FileName=CNNlay4_Rand0PreEnergy1000ms_Scheduler_Window800ms_PReLu_Drop30_FreezeSincNet --cuda=0 > nohup_freezed.out &
# nohup python main.py --configPath=cfg/SincNet_DCASE_CNNLay4_Rand0PreEnergyWindow800_Scheduler_PReLu_Drop30.cfg --FileName=CNNlay4_Rand0PreEnergy1000ms_Scheduler_Window800ms_PReLu_Drop30_normalSincNet --cuda=0 > nohup_normal.out &
# nohup python main.py --configPath=cfg/SincNet_DCASE_CNNLay4_Rand0PreEnergyWindow800_Scheduler_1DConvOnly_PReLu_Drop30.cfg --cuda=0 > nohup_ConOnly.out &
# nohup python main.py --configPath=cfg/SincNet_DCASE_CNNLay4_Rand0PreEnergyWindow800_Scheduler_PReLu_Drop30.cfg --FileName=CNNlay4_Rand0PreEnergy1000ms_Scheduler_Window800ms_PReLu_Drop30_test --cuda=1 > nohup_test.out &

# nohup python main.py --configPath=cfg/SincNet2D/SincNet2D_CNNLay5_Rand0PreEnergyWindow3000_Scheduler_PReLu_Drop30.cfg --cuda=0 &
# nohup python main.py --configPath=cfg/SincNet2D/SincNet2D_CNNLay4_Rand0PreEnergyWindow4000_16kHz_Scheduler_Drop30.cfg --cuda=1 &
# nohup python main.py --configPath=cfg/SincNet2D/SincNet2D_CNNLay4_Rand0PreEnergyWindow5000_16kHz_Scheduler_Drop30.cfg --cuda=1 &
# nohup python main.py --configPath=cfg/SincNet2D/SincNet2D_CNNLay4_Rand0PreEnergyWindow5000_16kHz_DNN512_Scheduler_Drop30.cfg --cuda=1 &
# nohup python main.py --configPath=cfg/SincNet2D/SincNet2D_CNNLay4_Rand0PreEnergyWindow5000_16kHz_DNN256_FirstMaxpoolW1_Scheduler_Drop30.cfg --cuda=1 &
# nohup python main.py --configPath=cfg/SincNet2D/SincNet2D_CNNLay4_Rand0PreEnergyWindow3000_ModifCNN_Scheduler_Drop30.cfg --cuda=1 >nohup_espoir.out &
# nohup python main.py --configPath=cfg/SincNet2D/SincNet2D_CNNLay5_Rand0PreEnergyWindow3000_Scheduler_PReLu_Drop30.cfg --FileName=CNNlay5_Rand0PreEnergy4000ms_Scheduler0.2_Window3000ms_PReLu_Drop30_test2 --cuda=0 &
# nohup python main.py --configPath=cfg/SincNet2D/SincNet2D_CNNLay4_Rand0PreEnergyWindow4400_32kHz_Scheduler_Drop30.cfg --cuda=0 &
# nohup python main.py --configPath=cfg/SincNet2D/SincNet2D_CNNLay4_Rand0PreEnergyWindow4400_32kHz_Scheduler_More2dconvs_Drop30.cfg --cuda=0 &
# nohup python main.py --configPath=cfg/SincNet2D/SincNet2D_CNNLay4_Rand0PreEnergyWindow5000_32kHz_Scheduler_2dconvs64,128,256_Drop30.cfg --cuda=0 &
# nohup python main.py --configPath=cfg/SincNet2D/vggish_model.cfg --cuda=1 &
# nohup python main.py --configPath=cfg/test.cfg --cuda=0 &
## Parameters that needs to change each execution:
model_file_name   = output_folder.split("/")[-2] if output_folder.split("/")[-1]=="" else output_folder.split("/")[-1]
## Loads the file from options.FileName if the parameter is used:
if(options.FileName != 'None'):
    model_file_name = options.FileName
#Training_model_file  += "_try2"
Models_file_extension = ".pkl" if pt_file == 'none' else pt_file.split(".")[1]
previous_model_path   = output_folder+ '/' + model_file_name if pt_file == 'none' else pt_file.split(".")[0]
Load_previous_model   = False if pt_file == 'none' else True
inTheSameFile         = False
plotGrad              = False
compute_matrix        = False
n_classes             = class_lay[-1]#41 for SincNet
same_classes          = same_classes
is_SincNet            = "SincNet" in options.configPath


## are in cfg:
#beta_coef=0.4
#use_mixup
#N_eval_epoch = 1
#same_classes = True

## Freezing Sinc filters:
#print("Parameters of Sinc filters are freezed.")
#Main_net.CNN_net.conv[0].low_hz_.requires_grad  = False
#Main_net.CNN_net.conv[0].band_hz_.requires_grad = False

## Loading previously trained model if needed:
Main_net, CNN_net, DNN1_net, DNN2_net, previous_epoch, min_loss = LoadPrevModel(Main_net, CNN_net, DNN1_net, DNN2_net,
                                                                previous_model_path, 
                                                                Models_file_extension, 
                                                                Load= Load_previous_model, 
                                                                inSameFile = inTheSameFile,
                                                                test_acc_period = N_eval_epoch,
                                                                evalMode = False)


## Training parameters available in the SincNet_TIMIT.cfg file section [optimization]:
# batch_size
# N_epochs     = 1500
# N_batches    = 800
# N_eval_epoch = 8

## Overwriting N_epochs just bcz flemme:
#N_epochs = 58



train(Main_net, optimizers, train_loader, valid_loader, cost, cost_onehot,
          ## Data related variables:
          wlen,
          wshift,
          n_classes,
          ## File variables:
          output_folder,
          model_file_name,
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
          initial_minloss = min_loss,
          ## Tracking gradient
          plotGrad = plotGrad,
          ## If user wishes to use a scheduler:
          use_scheduler = use_scheduler,
          scheduler = schedulers,
          ## If user wishes to save and compute confusion matrix:
          compute_matrix = compute_matrix,
          ## Indicates if the network that is trained is SincNet
          is_SincNet = True,
          cuda=True)
