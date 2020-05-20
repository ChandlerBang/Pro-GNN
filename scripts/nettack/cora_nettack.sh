#!/bin/bash

python train.py \
    --seed 10 \
    --dataset cora \
    --attack nettack \
    --ptb_rate 3 \
    --alpha  5e-4 \
    --beta 1.5  \
    --gamma 1 \
    --lambda_ 1e-3 \
    --lr  5e-4 \
    --epoch 1000 \
    \

     
