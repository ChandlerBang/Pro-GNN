# Pro-GNN

A PyTorch implementation of "Graph Structure Learning for Robust Graph Neural Networks" (KDD 2020).

The code is based on our Pytorch adversarial repository, DeepRobust [(https://github.com/DSE-MSU/DeepRobust)](https://github.com/DSE-MSU/DeepRobust)

<div align=center><img src="https://raw.githubusercontent.com/ChandlerBang/Pro-GNN/master/ProGNN.png" width="700"/></div>


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
To run the code, first you need to clone DeepRobust
```
git clone https://github.com/DSE-MSU/DeepRobust.git
cd DeepRobust
python setup.py install
```

## Run the code
After installation, you can clone this repository
```
git clone https://github.com/ChandlerBang/Pro-GNN.git
cd Pro-GNN
python train.py --dataset polblogs --attack meta --ratio 0.15 epoch 1000
```

To reproduce the performance reported in the paper, you can run the bash files in folder `scripts`
```
sh scripts/meta/cora_meta.sh
```

## Cite
For more information, you can take a look at the [paper](link) or the detailed [code](https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/defense/prognn.py) shown in DeepRobust.

If you find this repo to be useful, please cite our paper. Thanks
