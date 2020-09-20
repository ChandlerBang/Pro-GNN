#!/bin/bash

python train.py \
    --seed 15 \
    --dataset pubmed \
    --attack meta \
    --ptb_rate 0.0 \
    --lr  0.01 \
    --epoch 200 \
    --only_gcn \
    
