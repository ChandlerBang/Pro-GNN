#!/bin/bash

python train.py \
    --seed 10 \
    --dataset citeseer \
    --attack meta \
    --ptb_rate 0.15 \
    --alpha  5e-4 \
    --beta 1.5  \
    --gamma 1 \
    --lambda_ 0.0001 \
    --lr  5e-4 \
    --inner_steps 1 \
    --outer_steps 1 \
    --epoch 1000 \
    \

     