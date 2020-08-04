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
conda install -n $envname -y jupyter pandas matplotlib seaborn nbformat
conda install -n $envname -y librosa prettytable jupyterlab pysoundfile tqdm -c conda-forge
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

### Test installation

When you finish the installation you can test it by executing the following commands:

```bash
conda activate SincNet
python Test_Model.py --configPath=cfg/SincNet_DCASE_CNNLay4_Rand0PreEnergyWindow800_Scheduler_PReLu_Drop30.cfg --FileName=CNNlay4_Rand0PreEnergy1000ms_Scheduler_Window800ms_PReLu_Drop30_normalSincNet --cuda=1
```

## Utilities

##### Path for trained models

They are on the yd-4q2twm2 machine @ /home/nlpt4239/SincNet_DCASE_v2.0/exp/SincNet_DCASE_v2.0


##### Tutors

Lionel Delphin-Poulat [link.](https://fr.linkedin.com/in/lionel-delphin-6a091a162)
Cyril Plapous [link.](https://fr.linkedin.com/in/cyril-plapous-983b04b1)

#####  Author

My name is Jean-Charles Layoun and I am the author of this document, if you want to contact me professionally please follow the [link.](https://fr.linkedin.com/in/jclayoun)

##### Co-Intern

 Paul Calot-Plaetevoet [link.](https://fr.linkedin.com/in/paul-calot-43549814b)