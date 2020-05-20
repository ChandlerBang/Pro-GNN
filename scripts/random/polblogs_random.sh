#!/bin/bash

python train.py \
    --seed 10 \
    --dataset polblogs \
    --attack random \
    --ptb_rate 0.6  \
    --alpha  5e-4 \
    --beta 2  \
    --gamma 1 \
    --lambda_ 0 \
    --lr  5e-3 \
    --epoch 1000 \
    \
     
