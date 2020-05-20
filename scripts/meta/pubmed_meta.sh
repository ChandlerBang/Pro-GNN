#!/bin/bash

python train.py \
    --seed 10 \
    --dataset pubmed \
    --attack meta \
    --ptb_rate 0.05 \
    --alpha  0.3 \
    --beta 2.5  \
    --gamma 1 \
    --lambda_ 0.001 \
    --lr 1e-2 \
    --epoch 100 \
    --inner_steps 30
    \
     
