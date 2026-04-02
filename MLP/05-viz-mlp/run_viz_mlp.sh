#!/bin/bash

NODES=0032
LAYERS=2
ROOT="../../MLP/mlp-cfg/mlp-$NODES/mlp-$NODES-$LAYERS/mlp-$NODES-$LAYERS"

python3 viz_mlp.py \
    $ROOT.json\
    $ROOT.pt\
    $ROOT-norm.pt\
    ../../MLP/02-data/\
    ./