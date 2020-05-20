#!/bin/bash

python train.py \
    --seed 10 \
    --dataset polblogs \
    --attack meta \
    --ptb_rate 0.15 \
    --alpha  1e-4 \
    --beta 5  \
    --gamma 1 \
    --lambda_ 0 \
    --lr  5e-4 \
    --epoch 1000 \
    \
     
