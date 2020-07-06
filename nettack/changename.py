import numpy as np
import scipy.sparse as sp
import os.path as osp
import os

path = './'
for file in os.listdir(path):

    if file[-3:] != 'npz':
        continue
    old_name = file.split('_')
    print(old_name)
    a, b = old_name[-1].split('.')
    new_name = old_name[0] + '_' + old_name[1] + '_' + a + '_' + old_name[2] + '.npz'
    print(new_name)
    os.system(f"git mv {file} {new_name}")


import ipdb
ipdb.set_trace()

