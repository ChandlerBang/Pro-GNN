# Final Project Team 10 - KDD 2020 Graph Structure Learning for Robust Graph Neural Networks

"Graph Structure Learning for Robust Graph Neural Networks" (KDD 2020). [[paper]](https://arxiv.org/abs/2005.10203)

[Team 10 Slide](https://docs.google.com/presentation/d/1Zdad6E2cU635qaVq9PJrZfprByjD3b0GP_yHgO5GOqs/edit?usp=sharing)

## Requirements
See that in https://github.com/DSE-MSU/DeepRobust/blob/master/requirements.txt
```
matplotlib==3.1.1
numpy==1.17.1
torch==1.2.0
scipy==1.3.1
torchvision==0.4.0
texttable==1.6.2
networkx==2.4
numba==0.48.0
Pillow==7.0.0
scikit_learn==0.22.1
skimage==0.0
tensorboardX==2.0
```

## Installation
```
pip install deeprobust
```
or 
```
git clone https://github.com/DSE-MSU/DeepRobust.git
cd DeepRobust
python setup.py install
```

## Run the code
```
git clone https://github.com/ChandlerBang/Pro-GNN.git
cd Pro-GNN
python train.py --dataset cora --attack meta --ptb_rate 0.15 --epoch 1000
```
or 
```
sh scripts/meta/cora_meta.sh
```

## Experiment Result
要重現我們的實驗可以跑，可以跑 scripts/experiment 下的sh，每個都代表一個實驗設定(已把option設好)
例如 `sh scripts/experiment/cora_meta_5_1_1.sh`

## 一些會用到的文件
[deeprobust document](https://deeprobust.readthedocs.io/_/downloads/en/latest/pdf/)