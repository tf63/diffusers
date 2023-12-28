#!/bin/bash
# code from https://github.com/RUB-SysSec/GANDCTAnalysis/blob/master/docker.sh
# Copyright (c) 2021 Chair for Sys­tems Se­cu­ri­ty, Ruhr University Bochum

# usage ----------------------------------------------
# bash docker.sh build  # build image
# bash docker.sh shell  # run container
# ----------------------------------------------------

DATASET_DIRS="$HOME/dataset"
DATA_DIRS="$HOME/data"

build()
{
    docker build . -f docker/diffusers-pytorch-cuda/Dockerfile -t tf63/diffuser
}

shell() 
{
    docker run --rm --gpus all --shm-size=32g -it -v $(pwd):/app -v $DATASET_DIRS:/dataset -v $DATA_DIRS:/data tf63/diffuser /bin/bash
}


if [[ $1 == "build" ]]; then
    build
elif [[ $1 == "shell" ]]; then
    shell 
else
    echo "error: invalid argument"
fi
