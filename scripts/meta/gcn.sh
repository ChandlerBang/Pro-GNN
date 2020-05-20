#!/bin/bash

python train.py \
    --seed 10 \
    --dataset cora \
    --attack meta \
    --ptb_rate 0.15 \
    --lr  1e-2 \
    --epoch 200 \
    --only_gcn \
    
