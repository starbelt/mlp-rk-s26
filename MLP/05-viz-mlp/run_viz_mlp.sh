#!/bin/bash

NODES=0256
LAYERS=5
ROOT="../../MLP/mlp-cfg/mlp-$NODES/mlp-$NODES-$LAYERS/mlp-$NODES-$LAYERS"

python3 viz_mlp.py \
    $ROOT.json\
    $ROOT.pt\
    $ROOT-norm.pt\
    ../../MLP/02-data/\
    ./