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
                        help='Indicates the name if the saving/loading file.')
    parser.add_argument('--cuda', metavar='-c', type=int,
                        help='Indicates the graphic Card you wish to use!')

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

    #[cnn]
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
    #[optimization]
    if('optimization' in Config.sections()):
        options.lr=Config.get('optimization', 'lr')
        options.batch_size=Config.get('optimization', 'batch_size')
        if 'Batch_dev' in Config['optimization']:
            options.Batch_dev=Config.get('optimization', 'Batch_dev')
        else:
            print("You did not specify the value of `Batch_dev`, it is set to 32.")
            options.Batch_dev=32

        if 'patience' in Config['optimization']:
            options.patience=Config.get('optimization', 'patience')
        else:
            print("You did not specify the value of `patience`, it is set to 7.")
            options.patience=7

        options.N_epochs=Config.get('optimization', 'N_epochs')
        options.N_batches=Config.get('optimization', 'N_batches')
        options.N_eval_epoch=Config.get('optimization', 'N_eval_epoch')
        
        if 'train_acc_period' in Config['optimization']:
                options.train_acc_period=Config.get('optimization', 'train_acc_period')
        else:
            print("You did not specify the value of `train_acc_period`, it is set to 5.")
            options.train_acc_period=5
        
        if 'use_mixup' in Config['optimization']:
            options.use_mixup=Config.get('optimization', 'use_mixup')
        else:
            print("You did not specify the value of `use_mixup`, it is set to False.")
            options.use_mixup='False'

        if 'beta_coef' in Config['optimization']:
            options.beta_coef=Config.get('optimization', 'beta_coef')
        else:
            print("You did not specify the value of `beta_coef`, it is set to 0.4.")
            options.beta_coef=0.4
        
        if 'same_classes' in Config['optimization']:
            options.same_classes=Config.get('optimization', 'same_classes')
        else:
            print("You did not specify the value of `same_classes`, it is set to None.")
            options.same_classes=None
            
        options.seed=Config.get('optimization', 'seed')
    else:
        print("Error, you did not specify optimization parameters in your cfg. Consequently, the code won't run...")

    return options
