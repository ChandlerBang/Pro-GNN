#!/bin/bash

python train.py \
    --seed 10 \
    --dataset polblogs \
    --attack nettack \
    --ptb_rate 3 \
    --alpha  0.3 \
    --beta 0.5  \
    --gamma 1 \
    --lambda_ 0.5 \
    --lr  1e-2 \
    --epoch 100 \
    --inner_steps 30
    \

     
