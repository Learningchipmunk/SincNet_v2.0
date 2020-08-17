import configparser as ConfigParser
#argparse doc: https://docs.python.org/2/library/argparse.html#module-argparse
import argparse

#import numpy as np

## Here are the config data processing functions used in data_io:

# Converts string to bool:
def str_to_bool(s):
    """Convert Strings like "True" and "False" to booleans.

    Args:
        s (String): A string that should be "True" or "False"

    Raises:
        ValueError: If String has a wrong value

    Returns:
        (bool): Converted bool.
    """
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         print(s)
         raise ValueError 


def read_conf():
    """The purpose of this function is to read the config files saved in .cfg format.

    Returns:
        (object): Returns an object with all the needed information.
    """
    # Executing with --cfg = path
    parser = argparse.ArgumentParser(description='Running model with config file.')
    
    parser.add_argument('--configPath', metavar='-cfg', type=str,
                    help='The path of the configuration file. \n\t ex: cfg/SincNet_DCASE_Preprocessing_WithEnergy_Window_800.cfg')
    parser.add_argument('--FileName', metavar='-fn', type=str, default='None',
                        help='Indicates the name of the saved/loaded file.')
    parser.add_argument('--TestDataPath', metavar='-tdp', type=str, default='None',
                        help='Indicates the path of the saved test tensor files.')
    parser.add_argument('--cuda', metavar='-c', type=int, default=-1,
                        help='Indicates the number of the Cuda Device(Graphic Card) you wish to use! -1 Means CPU.')

    # Reads the arguments the user wrote on the command line:
    options = parser.parse_args()
    

    # Reading the config file with config parser
    Config = ConfigParser.ConfigParser()
    Config.read(options.configPath)

    #[data]
    options.tr_lst=Config.get('data', 'tr_lst')
    options.te_lst=Config.get('data', 'te_lst')
    options.lab_dict=Config.get('data', 'lab_dict')
    options.data_folder=Config.get('data', 'data_folder')
    options.output_folder=Config.get('data', 'output_folder')
    options.pt_file=Config.get('data', 'pt_file')

    #[windowing]
    options.fs=Config.get('windowing', 'fs')
    options.cw_len=Config.get('windowing', 'cw_len')
    options.cw_shift=Config.get('windowing', 'cw_shift')

    if('cnn2D' in Config.sections()):
        #[cnn2D]
        options.is_conv2D = True
        options.cnn_N_filt=Config.get('cnn2D', 'cnn_N_filt')
        options.cnn_len_filt_W=Config.get('cnn2D', 'cnn_len_filt_W')
        options.cnn_len_filt_H=Config.get('cnn2D', 'cnn_len_filt_H')
        options.cnn_energy_L=Config.get('cnn2D', 'cnn_energy_L')
        options.cnn_energy_stride=Config.get('cnn2D', 'cnn_energy_stride')
        options.cnn_max_pool_len_W=Config.get('cnn2D', 'cnn_max_pool_len_W')
        options.cnn_max_pool_len_H=Config.get('cnn2D', 'cnn_max_pool_len_H')
        options.cnn_use_laynorm_inp=Config.get('cnn2D', 'cnn_use_laynorm_inp')
        options.cnn_use_batchnorm_inp=Config.get('cnn2D', 'cnn_use_batchnorm_inp')
        options.cnn_use_laynorm=Config.get('cnn2D', 'cnn_use_laynorm')
        options.cnn_use_batchnorm=Config.get('cnn2D', 'cnn_use_batchnorm')
        options.cnn_act=Config.get('cnn2D', 'cnn_act')
        options.cnn_drop=Config.get('cnn2D', 'cnn_drop')
    else:
        #[cnn]
        options.is_conv2D = False
        options.cnn_N_filt=Config.get('cnn', 'cnn_N_filt')
        options.cnn_len_filt=Config.get('cnn', 'cnn_len_filt')
        options.cnn_max_pool_len=Config.get('cnn', 'cnn_max_pool_len')
        options.cnn_use_laynorm_inp=Config.get('cnn', 'cnn_use_laynorm_inp')
        options.cnn_use_batchnorm_inp=Config.get('cnn', 'cnn_use_batchnorm_inp')
        options.cnn_use_laynorm=Config.get('cnn', 'cnn_use_laynorm')
        options.cnn_use_batchnorm=Config.get('cnn', 'cnn_use_batchnorm')
        options.cnn_act=Config.get('cnn', 'cnn_act')
        options.cnn_drop=Config.get('cnn', 'cnn_drop')


    #[dnn]
    options.fc_lay=Config.get('dnn', 'fc_lay')
    options.fc_drop=Config.get('dnn', 'fc_drop')
    options.fc_use_laynorm_inp=Config.get('dnn', 'fc_use_laynorm_inp')
    options.fc_use_batchnorm_inp=Config.get('dnn', 'fc_use_batchnorm_inp')
    options.fc_use_batchnorm=Config.get('dnn', 'fc_use_batchnorm')
    options.fc_use_laynorm=Config.get('dnn', 'fc_use_laynorm')
    options.fc_act=Config.get('dnn', 'fc_act')

    #[class]
    options.class_lay=Config.get('class', 'class_lay')
    options.class_drop=Config.get('class', 'class_drop')
    options.class_use_laynorm_inp=Config.get('class', 'class_use_laynorm_inp')
    options.class_use_batchnorm_inp=Config.get('class', 'class_use_batchnorm_inp')
    options.class_use_batchnorm=Config.get('class', 'class_use_batchnorm')
    options.class_use_laynorm=Config.get('class', 'class_use_laynorm')
    options.class_act=Config.get('class', 'class_act')


    #[optimization]
    if('optimization' in Config.sections()):
        ## optimizer_type:
        if 'optimizer_type' in Config['optimization']:
            options.optimizer_type=Config.get('optimization', 'optimizer_type')
        else:
            options.optimizer_type='RMSprop'
            print("You did not specify the value of `optimizer_type`, it is set to {}.".format(options.optimizer_type))

        options.lr=Config.get('optimization', 'lr')

        ## use_scheduler:
        if 'use_scheduler' in Config['optimization']:
            options.use_scheduler=Config.get('optimization', 'use_scheduler')
        else:
            print("You did not specify the value of `use_scheduler`, it is set to False.")
            options.use_scheduler='False'

        ## scheduler_type:
        if 'scheduler_type' in Config['optimization']:
            options.scheduler_type=Config.get('optimization', 'scheduler_type')
        else:
            options.scheduler_type='ReduceLROnPlateau'
            print("You did not specify the value of `scheduler_type`, it is set to {}.".format(options.scheduler_type))


        ## scheduler_patience:
        if 'scheduler_patience' in Config['optimization']:
            options.scheduler_patience=Config.get('optimization', 'scheduler_patience')
        else:
            options.scheduler_patience=2
            print("You did not specify the value of `scheduler_patience`, it is set to {}.".format(options.scheduler_patience))

        ## scheduler_factor:
        if 'scheduler_factor' in Config['optimization']:
            options.scheduler_factor=Config.get('optimization', 'scheduler_factor')
        else:
            options.scheduler_factor=0.5
            print("You did not specify the value of `scheduler_factor`, it is set to {}.".format(options.scheduler_factor))


        options.batch_size=Config.get('optimization', 'batch_size')

        ## Batch_dev:
        if 'Batch_dev' in Config['optimization']:
            options.Batch_dev=Config.get('optimization', 'Batch_dev')
        else:
            options.Batch_dev=32
            print("You did not specify the value of `Batch_dev`, it is set to {}.".format(options.Batch_dev))

        ## patience:
        if 'patience' in Config['optimization']:
            options.patience=Config.get('optimization', 'patience')
        else:
            print("You did not specify the value of `patience`, it is set to 7.")
            options.patience=7

        options.N_epochs=Config.get('optimization', 'N_epochs')
        options.N_batches=Config.get('optimization', 'N_batches')
        options.N_eval_epoch=Config.get('optimization', 'N_eval_epoch')
        
        ## train_acc_period:
        if 'train_acc_period' in Config['optimization']:
                options.train_acc_period=Config.get('optimization', 'train_acc_period')
        else:
            print("You did not specify the value of `train_acc_period`, it is set to 5.")
            options.train_acc_period=5
        
        ## fact_amp:        
        if 'fact_amp' in Config['optimization']:
                options.fact_amp=Config.get('optimization', 'fact_amp')
        else:
            options.fact_amp=0.2
            print("You did not specify the value of `fact_amp`, it is set to {}.".format(options.fact_amp))
        
        ## use_mixup:
        if 'use_mixup' in Config['optimization']:
            options.use_mixup=Config.get('optimization', 'use_mixup')
        else:
            print("You did not specify the value of `use_mixup`, it is set to False.")
            options.use_mixup='False'
        
        ## mixup_batch_prop:        
        if 'mixup_batch_prop' in Config['optimization']:
            options.mixup_batch_prop=Config.get('optimization', 'mixup_batch_prop')
        else:
            options.mixup_batch_prop=float(1.0) if options.use_mixup=='True' else float(0.0)
            print("You did not specify the value of `mixup_batch_prop`, it is set to {}%.".format(options.mixup_batch_prop*100))
        
        ## beta_coef:
        if 'beta_coef' in Config['optimization']:
            options.beta_coef=Config.get('optimization', 'beta_coef')
        else:
            print("You did not specify the value of `beta_coef`, it is set to 0.4.")
            options.beta_coef=0.4
        
        ## same_classes:        
        if 'same_classes' in Config['optimization']:
            options.same_classes=Config.get('optimization', 'same_classes')
        else:
            options.same_classes='False'
            print("You did not specify the value of `same_classes`, it is set to {}.".format(options.same_classes))
            if("True" in options.use_mixup):
                print("Warning: you are using mixup but you did not mention which type in config file. \n"+
                    "By default it will be set to False. You are advised to add a same_class attribute to your cfg file and set it to True or False.")    

            
        options.seed=Config.get('optimization', 'seed')
    else:
        print("Error, you did not specify optimization parameters in your cfg. Consequently, the code won't run...")

    #[Misc]
    ## In SincNet, we must always use SincConv_fast, it is the whole point of SincNet. But, just for testing, we added the possibility to deactivate it. 
    ## This is why we do not prompt the user if he does not have a `Misc`section.
    options.use_SincConv_fast='True'
    if('Misc' in Config.sections()):
        ## use_SincConv_fast:        
        if 'use_SincConv_fast' in Config['Misc']:
            options.use_SincConv_fast=Config.get('Misc', 'use_SincConv_fast')
        else:
            print("You did not specify the value of `use_SincConv_fast`, but don't worry, it is set to {}.".format(options.use_SincConv_fast))
        

    return options
