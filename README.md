# SincNet v2.0 for DCASE:

This is the Version of SincNet with newer data loader, training and testing functions.

## Conda Environment Setup

You can either use the `create_conda_environment.sh` provided script or follow the step by step procedure below.

### Bash Script Setup

`create_conda_environment.sh` has two options:

- `-n|--name` with this option you can overwrite the default environment names which is `SincNet`
- `-g|--gpu` if this option is set, the environment will contain a pytorch version with gpu support

To create the conda environment `SincNet`, with gpu support, run:

```bash
bash ./create_conda_environment.sh -g
```

### Step by Step Guide

If you do not use the setup script, you can create the environment with the following steps:

- Create the environment

```bash
conda create -n SincNet python=3.6.9 numpy=1.16.1
```

- Activate the created conda environment

```bash
conda activate SincNet
```

- Within conda environment, Install the libraries

```bash
conda install -y jupyter pandas matplotlib seaborn nbformat
conda install -y librosa prettytable jupyterlab pysoundfile tqdm -c conda-forge
```

* Within conda environment, pip Install those libraries too

  ```bash
  pip install nnAudio, nbresuse, torchsummary
  ```

Those packages are compatible with pytorch 1.1.0

- Install pytorch(within conda environment)

  - with gpu support:

  ```bash
  conda install pytorch==1.1.0 torchvision==0.3.0 -c pytorch
  ```

  - or without gpu support:

  ```bash
  conda install pytorch==1.1.0 torchvision==0.3.0 cpuonly -c pytorch
  ```

## How to Setup SincNet for Experimentation:

### Fetch DCASE2018 Task2 Data

You can either fetch the Data from Orange's server (**recommended**) or fetch it from kaggle directly.

#### On Orange's server

The DCASE data already reviewed by Lionel is available at `/data2/dcase2018/task2/FSDKaggle2018.audio_train/` and `/data2/dcase2018/task2/FSDKaggle2018.audio_test/`on the server **yd-4q2twm2**.

You can store this data on your local machine anywhere as long as you save the path, you will need it for the next steps.

#### Kaggle

Follow the [link](https://www.kaggle.com/c/freesound-audio-tagging/data), register and download the data.

###### Remarks

>  Data lists are already available for DCASE data.

> :warning: **If you are not using DCASE audios**: you must generate your own data lists! :warning:

### Preprocessing

The preprocessing is done in the notebook `Pre-processing_audio_files_to_Tensors`. It must be placed at the same level as main.py, in the same directory as Images.

In this notebook, you should replace the values of the variables `dir_audio_train` and `dir_audio_test` with the paths of the Train and Test data that you fetched before hand. 

![Replace values here](Images/Readme.md_Images/Pre_Image1.PNG)

#### Training Set

Afterward, execute everything before **Preprocessing train audio on Energy** in the notebook. Then proceed to change the values of the variables and the folder's  location to your liking.

![Replace values here](Images/Readme.md_Images/Pre_Image2.PNG)

In our tests, we preprocessed with those values:

* **targetSamplingRate** = [16kHz, 32kHz]

* **window_length** = [1000ms, 4000ms, 5000ms]

  > **delay, L, stride, random_padding_zeros, repeating signal,** etc... Were set after multiple test runs and were chosen based on their results. You can change them to your convenience, but keep in mind that you might not have the same results as we did... 

#### Testing Set

Just below the Training Set preprocessing you have the **Preprocessing test audio on Energy** section that does the same to the test dataset!

> :warning: **Remark**: We recommend having the same settings for testing and training data preprocessing. :warning:

### Setup the Configuration File

In SincNet, the configuration files are usually in the cfg directory, they are recognizable by their file format `.cfg`.

- Modify the *[data]* section of `cfg/test.cfg` file according to your paths. In particular, modify the **data_folder** with the location of the preprocessed data that you created following the tutorial above. The other parameters of the config file belong to the following sections:

1. *[windowing]*, that defines how each sentence is split into smaller chunks.
2. *[cnn]* or *[cnn2d]*, that specifies the characteristics of the CNN architecture. **(WARNING: you should not have both in one .cfg file.)**
   1. *[cnn]*, specifies the characteristics of a 1D convolutional layer.
   2. [cnn2d], specifies the characteristics of a 2D convolutional layer after SincConvfast (that is 1D).
3. *[dnn]*, that specifies the characteristics of the fully-connected DNN architecture following the CNN layers.
4. *[class]*, that specify the logsoftmax classification part.
5. *[optimization]*, that reports the main hyperparameters used to train the architecture.

* Once you setup the *config* file, you can attempt to train your model. See [Training and Testing Models.](### Training and Testing Models)

### Training and Testing Models

#### Training a Model

To train a model, you must run the python script `main.py`.

`main.py` has three options:

- `-cfg|--configPath` with this option, you indicate the path of the configuration file you would like to use

* `-fn|--FileName` if this option has a value set, the saved models will be named after this value
* `-c|--cuda` if this option is set, the pytorch code will run with the Cuda device you specified. It defaults to -1, meaning that the CPU is chosen

To train a model with `test.cfg` on device `cuda:0`, execute the following command:

```bash
python main.py --configPath=cfg/test.cfg --cuda=0
```

#### Testing a model

To test a model, you must run the python script `Test_Model.py`.

>  `Test_Model.py` has the same options as `main.py`. ([See above](### Training a Model).)

> In `Test_Model.py`, in the section **Getting the data relevant to the test dataset**, you should modify the paths of `testTensorFiles`,  	`data_folder_test` and `lab_dict` according to the preprocessing you did in the [Testing Set](#### Testing Set) section!

To test your previously trained model `test.cfg` on device `cuda:0`, execute the following command:

```bash
python Test_Model.py --configPath=cfg/test.cfg --cuda=0
```



## Utilities

##### Path of the previously trained models

They are on the **yd-4q2twm2** server @ `/home/nlpt4239/SincNet_DCASE_v2.0/exp/SincNet_DCASE_v2.0`

##### Results of the previously trained models

You can find the most relevant results of previously trained models [here](https://gitlab.tech.orange/golden-ear-for-things/nn-acoustic-feature-extraction/test).


##### Tutors

Lionel Delphin-Poulat [link.](https://fr.linkedin.com/in/lionel-delphin-6a091a162)
Cyril Plapous [link.](https://fr.linkedin.com/in/cyril-plapous-983b04b1)

#####  Author

My name is Jean-Charles Layoun and I am the author of this document, if you want to contact me professionally please follow the [link.](https://fr.linkedin.com/in/jclayoun)

##### Co-Intern

 Paul Calot-Plaetevoet [link.](https://fr.linkedin.com/in/paul-calot-43549814b)