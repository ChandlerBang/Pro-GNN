import numpy as np
import scipy.sparse as sp
import os.path as osp
import os

path = './'
for file in os.listdir(path):
    if not osp.isfile(path + file):
        continue
    if file[-3:] != 'npy':
        continue

    filename = path + file
    print('Reading %s' % filename)
    adj = np.load(filename)
    adj = sp.csr_matrix(adj)
    sp.save_npz(f'{file[:-4]}.npz', adj)


import ipdb
ipdb.set_trace()

