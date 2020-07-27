"""
Test_Model.py
====================================
This is script that you can use to test trained nets on test dataset.

How to run it:
python Test_Model.py --configPath=cfg/SincNet_DCASE_Preprocessing_WithEnergy_Window_800_class=462.cfg --FileName=NTF_Energy_Window1000_p7 --cuda=1
python Test_Model.py --configPath=cfg/SincNet_DCASE_Preprocessing_WithEnergy_Window_800.cfg --FileName=NTF_Energy_Window1000_p7_try2 --cuda=1
python Test_Model.py --configPath=../SincNet_DCASE/cfg/SincNet_DCASE_Preprocessing_biggerWindow_800.cfg --FileName=model_raw_32kHz_Preprocessed_BiggerWindow_800 --cuda=1
python Test_Model.py --configPath=cfg/SincNet_DCASE_EnergyPre1000_Window800_withSameClassMixup.cfg --FileName=NTF_Energy_Window1000_p7_withSameClassMixup_beta0.1 --cuda=1 
python Test_Model.py --configPath=cfg/SincNet_DCASE_Rand0Pre_WithEnergy_Window_800.cfg --cuda=1 
python Test_Model.py --configPath=cfg/SincNet_DCASE_Rand0Pre_WithEnergy_Window_800_Drop30.cfg --cuda=1 
python Test_Model.py --configPath=cfg/SincNet_DCASE_Rand0Pre_WithEnergy_Window_800_Drop30.cfg  --FileName=NTF_EnergyPrepRepeat_Window_800--cuda=1 
python Test_Model.py --configPath=cfg/SincNet_DCASE_Rand0Pre_WithEnergy_Window_800_HiddenLay4_PReLu_Drop30.cfg --cuda=0
python Test_Model.py --configPath=cfg/SincNet_DCASE_Rand0Pre_WithEnergy_Window_800_PReLu_Drop30.cfg --cuda=0
python Test_Model.py --configPath=cfg/SincNet_DCASE_Rand0Pre_WithEnergy_Window_800_HiddenLay4_PReLu_Drop30.cfg --cuda=1
python Test_Model.py --configPath=cfg/SincNet_DCASE_Rand0Pre_WithEnergy_Window_800_PReLu.cfg --cuda=0
python Test_Model.py --configPath=cfg/SincNet_DCASE_Rand0Pre_WithEnergy_Window_800_HiddenLay4_PReLu.cfg --cuda=1
python Test_Model.py --configPath=cfg/SincNet_DCASE_RedimCNN_Rand0Pre_WithEnergy_Window_800_PReLu_Drop30.cfg --cuda=0
python Test_Model.py --configPath=cfg/SincNet_DCASE_CNNLay4_Rand0PreEnergyWindow800_Scheduler_PReLu_Drop30.cfg --cuda=1 
python Test_Model.py --configPath=cfg/SincNet_DCASE_CNNLay6_Rand0Pre_WithEnergy_Window3000_PReLu_Drop30.cfg --cuda=0
python Test_Model.py --configPath=cfg/SincNet_DCASE_CNNLay4_Rand0PreEnergyWindow800_Scheduler_PReLu_Drop30.cfg --FileName=CNNlay4_Rand0PreEnergy1000ms_Scheduler_Window800ms_PReLu_Drop30_try3 --cuda=1
"""
import numpy as np
import torch
import torch.nn as nn

## Local files imports:
from Models import MLP, flip, MainNet
from Models import SincNet as CNN 
from read_conf_files import read_conf, str_to_bool
from utils import Dataset, LoadPrevModel #,Dataset2
from training_and_acc_fun import accuracy
#from old_acc_fun import accuracy as old_accuracy

## <!>---------------------------- Reading the config file ----------------------------<!> ##
# Reading cfg file
options=read_conf()

## Architecture of the file:
#[data]
pt_file=options.pt_file
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
Batch_dev=int(options.Batch_dev)


# Converting context and shift in samples
wlen=int(fs*cw_len/1000.00)
wshift=int(fs*cw_shift/1000.00)


# loss functions
cost = nn.NLLLoss()

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

## Network that regroups all 3 networks:
Main_net = MainNet(CNN_net, DNN1_net, DNN2_net)
Main_net.cuda()


print("Initialization done!")


## <!>----------------------- Getting the data relevant to the test dataset -----------------------<!> ##
print("Getting data lists and dicts... \t\t", end="")

testTensorFiles = np.load("data_lists/Tensor_Test_list.npy")
# Stores the Number of files:
snt_te=len(testTensorFiles)

if("800" in options.configPath or "1000" in options.configPath):
    data_folder_test = "Data/Audio_Tensors/Test/Preprocessed_withEnergy_AudioTensors_Window1000ms/"
else:
    data_folder_test = "Data/Audio_Tensors/Test/Preprocessed_withEnergy_AudioTensors_Window4000ms_Random0Padding/"

lab_dict = np.load("data_lists/DCASE_tensor_test_labels.npy").item()
print("Done!")


print("Creating the dataset and dataloader... \t\t", end="")
## For old accuracy:
#test_dataset  = Dataset2(testTensorFiles, lab_dict, data_folder_test, wlen, 0.2, wshift = wshift, train = False)
#test_loader   = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=1,shuffle=False, drop_last=False)
## For new accuracy:
test_dataset  = Dataset(testTensorFiles, lab_dict, data_folder_test, wlen, 0.2, wshift = wshift, train = False)
test_loader   = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=Batch_dev,shuffle=False, drop_last=False)
print("Done!")


# test.cfg
## Parameters that needs to change each execution:
model_file_name   = output_folder.split("/")[-2] if output_folder.split("/")[-1]=="" else output_folder.split("/")[-1]
if(options.FileName != 'None'):
    model_file_name = options.FileName

Models_file_extension = ".pkl" if pt_file == 'none' else pt_file.split(".")[1]
previous_model_path   = output_folder+ '/' + model_file_name if pt_file == 'none' else pt_file.split(".")[0]
Load_previous_model   = True
at_epoch              = 100
inTheSameFile         = False## To change depending on versions
compute_matrix        = True
n_classes             = class_lay[-1]#41 for SincNet
sincnet_version       = 2## To change depending on versions
Save_net_for_kaggle   = False


## Adapts the paths if it is the version 1 of SincNet:
if(sincnet_version == 1):
    previous_model_path = "../SincNet_DCASE/" + previous_model_path



## Loading previously trained model if needed:
#Function was modified especially for .py folders:
Main_net, CNN_net, DNN1_net, DNN2_net, _ = LoadPrevModel(Main_net, CNN_net, DNN1_net, DNN2_net, 
                                                previous_model_path, 
                                                Models_file_extension, 
                                                Load= Load_previous_model, 
                                                inSameFile = inTheSameFile,
                                                at_epoch = at_epoch,
                                                evalMode = True)


best_class_error, cur_loss, window_error = accuracy(Main_net, test_loader, cost, n_classes,
                                                    Batch_dev, wlen, wshift,
                                                    matrix_name = model_file_name+"_test_dataset", compute_matrix = compute_matrix,
                                                    cuda=True)


print("\n")
print("Test set : ")
print('test loss: %.3f'       %(cur_loss))
print('window acc: %.3f'      %(1-window_error))
print('best class acc: %.3f'  %(1-best_class_error))

if(Save_net_for_kaggle):
    print("\n")
    print("Saved model `{0}` in `{1}`.".format(model_file_name, previous_model_path))
    torch.save(Main_net.state_dict(), previous_model_path + "_Main_net" + Models_file_extension)
