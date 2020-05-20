#!/bin/bash

python train.py \
    --seed 10 \
    --dataset cora \
    --attack random \
    --ptb_rate 0.4 \
    --alpha  0.01 \
    --beta 1  \
    --gamma 1.5 \
    --lambda_ 0.001 \
    --lr  1e-3 \
    --epoch 400 \
    \
     
