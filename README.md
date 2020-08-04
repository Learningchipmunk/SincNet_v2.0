# SincNet v2.0 for DCASE:

This is the Version of SincNet with newer data loader, training and testing functions.

## Conda Environment Setup

You can either use the `create_conda_environment.sh` provided script or follow the step by step procedure below.

### Running the bash Script to setup the environment

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

### Fetch Data

[TODO]

Here I should write how to fetch the data and where to put them.

### Preprocessing

[TODO]

Here I should write how to use the preprocessing notebook to preprocess kaggle data.

[For jupyter notebook, add pictures !]

#### Training Set



#### Testing Set



### Setup the Configuration File

In SincNet, the configuration files are usually in the cfg directory, they are recognizable by their file format `.cfg`.

- Modify the *[data]* section of *cfg/test.cfg* file according to your paths. In particular, modify the *data_folder* with the location of the preprocessed data that you created following the tutorial above. The other parameters of the config file belong to the following sections:

1. *[windowing]*, that defines how each sentence is split into smaller chunks.
2. *[cnn]*, that specifies the characteristics of the CNN architecture.
3. *[dnn]*, that specifies the characteristics of the fully-connected DNN architecture following the CNN layers.
4. *[class]*, that specify the logsoftmax classification part.
5. *[optimization]*, that reports the main hyperparameters used to train the architecture.

* Once you setup the cfg file, you can attempt to train your model. See [Training and Testing Models.](### Training and Testing Models)

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

> In `Test_Model.py`, in the section **Getting the data relevant to the test dataset**, you should modify the paths of your `testTensorFiles`,  	`data_folder_test` and `lab_dict` according to the preprocessing you did in the [Testing Set](#### Testing Set) section!

To test your previously trained model `test.cfg` on device `cuda:0`, execute the following command:

```bash
python Test_Model.py --configPath=cfg/test.cfg --cuda=0
```



## Utilities

##### Path for the trained models

They are on the yd-4q2twm2 machine @ /home/nlpt4239/SincNet_DCASE_v2.0/exp/SincNet_DCASE_v2.0

##### Results of the trained models

You can find the most relevant results of previous training [here](https://gitlab.tech.orange/golden-ear-for-things/nn-acoustic-feature-extraction/test).


##### Tutors

Lionel Delphin-Poulat [link.](https://fr.linkedin.com/in/lionel-delphin-6a091a162)
Cyril Plapous [link.](https://fr.linkedin.com/in/cyril-plapous-983b04b1)

#####  Author

My name is Jean-Charles Layoun and I am the author of this document, if you want to contact me professionally please follow the [link.](https://fr.linkedin.com/in/jclayoun)

##### Co-Intern

 Paul Calot-Plaetevoet [link.](https://fr.linkedin.com/in/paul-calot-43549814b)