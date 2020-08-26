"""
Generating Summaries
 Author: Jean-Charles LAYOUN 
 August 2020

Description: 
 This code generates summaries of the models from .cfg files .
 
How to run it:
 python Generate_Summary.py --configPath=$CFG_PATH --cuda=$CUDA_Number
    ex: 
    1D models:
        python Generate_Summary.py --configPath=cfg/SincNet_DCASE_Preprocessing_WithEnergy_Window_800.cfg --cuda=0
        python Generate_Summary.py --configPath=cfg/SincNet_DCASE_CNNLay4_Rand0PreEnergyWindow800_Scheduler_PReLu_Drop30.cfg --cuda=0
        python Generate_Summary.py --configPath=cfg/SincNet_DCASE_CNNLay6_DNN1024_Rand0Pre_WithEnergy_Window3000_PReLu_Drop30.cfg --cuda=0
    
    2D models:
        python Generate_Summary.py --configPath=cfg/SincNet2D/SincNet2D_CNNLay4_Rand0PreEnergyWindow5000_DNN384_32kHz_Scheduler_More2dconvs_Drop30.cfg --cuda=0
        python Generate_Summary.py --configPath=cfg/SincNet2D/SincNet2D_CNNLay4_Rand0PreEnergyWindow5000_32kHz_Scheduler_More2dconvs_Drop30.cfg --cuda=0


Note:
    The $CFG_PATH must be relative to the location of this script.
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import configparser as ConfigParser
from prettytable import PrettyTable

## Local files imports:
from Models import MLP, flip, MainNet
from Models import SincNet as CNN
from Models import SincNet2D as CNN2D
from read_conf_files import str_to_bool
from utils import Dataset, LoadPrevModel
from training_and_acc_fun import accuracy
from read_conf_files import read_conf


def count_parameters(model, CNN_arch, DNN1_arch, DNN2_arch):
    """Function that prints a the summary table of a network. 
        It also counts how many parameters require gradient.

    Args:
        model (nn.Module): A neural network with multiple paramters requiring gradient.
        CNN_arch (dict): A dictionary that stores the Architecture of the CNN.
        DNN1_arch (dict): A dictionary that stores the Architecture of the DNN.
        DNN2_arch (dict): A dictionary that stores the Architecture of the Classifier.

    Returns:
        int: Returns the total number of parameters that requires gradient. 
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
            
        ## Splitting the name into processable data:
        l        = name.split(".")

        Net_name = l[0]
        fun      = l[1]
        num      = int(l[2]) if len(l[2])==1 else -1
        
        

        if('bn' in fun):
            if('CNN' in Net_name):
                if(not CNN_arch['cnn_use_batchnorm'][num]):
                    continue
            elif('DNN1' in Net_name):
                if(not DNN1_arch['fc_use_batchnorm'][num]):
                    continue
            elif('DNN2' in Net_name):
                if(not DNN2_arch['fc_use_batchnorm'][num]):
                    continue
            
        if('ln' in fun):
            if('0' in fun):
                if('CNN' in Net_name):
                    if(not CNN_arch['cnn_use_laynorm_inp']):
                        continue
                elif('DNN1' in Net_name):
                    if(not DNN1_arch['fc_use_laynorm_inp']):
                        continue
                elif('DNN2' in Net_name):
                    if(not DNN2_arch['fc_use_laynorm_inp']):
                        continue
            else:
                if('CNN' in Net_name):
                    if(not CNN_arch['cnn_use_laynorm'][num]):
                        continue
                elif('DNN1' in Net_name):
                    if(not DNN1_arch['fc_use_laynorm'][num]):
                        continue                        
                elif('DNN2' in Net_name):
                    if(not DNN2_arch['fc_use_laynorm'][num]):
                        continue
            
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    


def main():

    ## -- Handling arguments: -- ##
    # Reading cfg file and storing its parameters into options :
    options = read_conf()    
    ## -- End of handling arguments -- ##

    ## <!>---------------------------- Reading the config file ----------------------------<!> ##
    ## Architecture of the file:
    #[data]
    pt_file=options.pt_file
    output_folder_model=options.output_folder

    #[windowing]
    fs=int(options.fs)
    cw_len=int(options.cw_len)
    cw_shift=int(options.cw_shift)

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
    Batch_dev=int(options.Batch_dev)

    #[Misc]
    use_SincConv_fast = str_to_bool(options.use_SincConv_fast)

    # Converting context and shift in samples
    wlen=int(fs*cw_len/1000.00)
    wshift=int(fs*cw_shift/1000.00)
    ## <!>---------------------------- End of cfg information extraction ----------------------------<!> ##

    ## -- Setting cuda Device: -- ##
    print("Selecting device... \t\t", end="")
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
    ## --       Setup Done     -- ##


    ## <!>----------------------- Initializing the Networks with .cfg options -----------------------<!> ##

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

    #fc_lay = [2048]*3

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

    Main_net = MainNet(CNN_net, DNN1_net, DNN2_net)
    Main_net.cuda()

    print("Initialization done!")
    ## <!>----------------------- Initialization done -----------------------<!> ##


    ## -- Setting up the output folder: -- ##
    output_folder = "Model_Summaries/SincNet2D" if is_conv2D else "Model_Summaries/SincNet1D"
    if not os.path.exists(output_folder):
        print("Creating the summary folder at `{}`".format(output_folder))
        os.makedirs(output_folder)
    ## --           Set up Done         -- ##

    ## -- Computing the model summary:  -- ##
    model_file_name   = output_folder_model.split("/")[-2] if output_folder_model.split("/")[-1]=="" else output_folder_model.split("/")[-1]

    # Computing number of params:
    print("")
    count_parameters(Main_net, CNN_arch, DNN1_arch, DNN2_arch)
    print("")

    original_stdout = sys.stdout # Save a reference to the original standard output

    # Saving Summary:
    print("Saving the summary in the folder located at `{}`... \t\t".format(output_folder), end="")

    with open(output_folder + "/" + model_file_name + '.txt', 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        count_parameters(Main_net, CNN_arch, DNN1_arch, DNN2_arch)
        sys.stdout = original_stdout # Reset the standard output to its original value
    
    print("Saved without problems!")
    ## --             Done              -- ##



if __name__ == "__main__":
    # execute only if run as a script
    main()
