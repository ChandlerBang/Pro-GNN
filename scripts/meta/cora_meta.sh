#!/bin/bash

python train.py \
    --seed 10 \
    --dataset cora \
    --attack meta \
    --ptb_rate 0.15  \
    --alpha  5e-4 \
    --beta 1.5  \
    --gamma 1 \
    --lambda_ 0.001 \
    --lr  5e-4 \
    --epoch 1000 \
    \
     
