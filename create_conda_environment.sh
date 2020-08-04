#!/bin/bash
# reproducibly create conda env

gpu=0
envname=SincNet
pytorchversion=1.1.0
torchvision==0.3.0

usage()
{
    command=$(basename -- $0)
    echo "usage: $command [[[-n environment] [-p pytorch_version] [-g]] | [-h]]"
}

while [ "$1" != "" ]; do
    case $1 in
        -n | --name )           shift
                                envname=$1
                                ;;
        -g | --gpu )            gpu=1
                                ;;
        -t | --pytorch_version ) shift
                                version=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done


# Ask the user one more time before creating the environment:
read -p "Create new conda env named $envname (y/n)?" CONT



if [ "$CONT" == "n" ]; then
  echo "exit";
  exit
else
# user chooses to create conda env
    # prompt user for conda env name
    echo "creating and setting up conda environment $envname gpu $gpu pytorch version $pytorchversion"
    conda create -y --name $envname python=3.6.9 numpy=1.16.1

    echo "installing base packages"
    conda install -n $envname -y jupyter pandas matplotlib seaborn nbformat
    conda install -n $envname -y librosa prettytable jupyterlab pysoundfile tqdm -c conda-forge

    eval "$(conda shell.bash hook)"
    conda activate $envname

    echo "pip install in $CONDA_PREFIX"
    pip install nnAudio, nbresuse, torchsummary

    echo "conda install pytorch in $CONDA_PREFIX"
    if [ $gpu -eq "1" ]; then
        echo "install pytorch with gpu support"
        conda install pytorch==$pytorchversion torchvision==$torchvision -c pytorch
    else
        echo "install pytorch on cpu only"
        conda install pytorch==$pytorchversion torchvision==$torchvision cpuonly -c pytorch
    fi
fi
