#!/bin/bash

python train.py \
    --seed 10 \
    --dataset polblogs \
    --attack nettack \
    --ptb_rate 3 \
    --alpha  1e-4 \
    --beta 5  \
    --gamma 1 \
    --lambda_ 0 \
    --lr  1e-3 \
    --epoch 1000 \
    \

     
