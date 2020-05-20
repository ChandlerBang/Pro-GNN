#!/bin/bash

python train.py \
    --seed 10 \
    --dataset citeseer \
    --attack random \
    --ptb_rate 0.4 \
    --alpha  5e-4 \
    --beta 1  \
    --gamma 2 \
    --lambda_ 5e-4 \
    --lr  5e-4 \
    --epoch 1000 \
    \

     
