#!/bin/bash

python train.py \
    --seed 10 \
    --dataset citeseer \
    --attack nettack \
    --ptb_rate 3 \
    --alpha  5e-4 \
    --beta 1.5  \
    --gamma 1 \
    --lambda_ 1e-4 \
    --lr  5e-4 \
    --epoch 1000 \
    \

     
