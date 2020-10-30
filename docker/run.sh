#!/bin/bash
docker run \
    -d \
    --init \
    -p6006:6006 -p5000:5000 -p8888:8888 \
    --rm \
    -it \
    --gpus=all \
    --ipc=host \
    --name=PyTorchLightningHydra \
    --env-file=.env \
    --volume=$PWD:/workspace \
    --volume=$DATASET:/dataset \
    pytorch_lightning_hydra:latest \
    ${@-fish}
