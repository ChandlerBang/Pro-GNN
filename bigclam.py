"""
Implementation of the bigClAM algorithm.

Throughout the code, we will use tho following variables

  * F refers to the membership preference matrix. It's in [NUM_PERSONS, NUM_COMMUNITIES]
   so index (p,c) indicates the preference of person p for community c.
  * A refers to the adjency matrix, also named friend matrix or edge set. It's in [NUM_PERSONS, NUM_PERSONS]
    so index (i,j) indicates is 1 when person i and person j are friends.
"""

import numpy as np
import pickle
import json
import scipy.sparse as sp

# 資料前處理
def load_npz(file_name):
    with np.load(file_name) as loader:
        # loader = dict(loader)
        #adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
        #                    loader['adj_indptr']), shape=loader['adj_shape'])
        adj = sp.csr_matrix((loader['data'], loader['indices'],
                            loader['indptr']), shape=loader['shape'])
        if 'attr_data' in loader:
            features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                      loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            features = None
        labels = loader.get('labels')
    if features is None:
        features = np.eye(adj.shape[0])
    features = sp.csr_matrix(features, dtype=np.float32)
    return adj, features, labels
    
def get_adj():
    adj, features, labels = load_npz('../../tmp/cora_meta_adj_0.15.npz')
    adj = adj + adj.T
    adj = adj.tolil()
    adj[adj > 1] = 1

    require_lcc=True
    if require_lcc:
        lcc = largest_connected_components(adj)
        adj = adj[lcc][:, lcc]
        features = features[lcc]
        #labels = labels[lcc]
        assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"
    
    # whether to set diag=0?
    adj.setdiag(0)
    adj = adj.astype("float32").tocsr()
    adj.eliminate_zeros()

    assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
    assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"

    return adj, features, labels

def largest_connected_components(adj, n_components=1):
    """Select k largest connected components.

	Parameters
	----------
	adj : scipy.sparse.csr_matrix
		input adjacency matrix
	n_components : int
		n largest connected components we want to select
	"""
    _, component_indices = sp.csgraph.connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep

def sigm(x):
    return np.divide(np.exp(-1.*x),1.-np.exp(-1.*x))

def log_likelihood(F, A):
    """implements equation 2 of 
    https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf"""
    A_soft = F.dot(F.T)

    # Next two lines are multiplied with the adjacency matrix, A
    # A is a {0,1} matrix, so we zero out all elements not contributing to the sum
    FIRST_PART = A*np.log(1.-np.exp(-1.*A_soft))
    sum_edges = np.sum(FIRST_PART)
    SECOND_PART = (1-A)*A_soft
    sum_nedges = np.sum(SECOND_PART)

    log_likeli = sum_edges - sum_nedges
    return log_likeli

def gradient(F, A, i):
    """Implements equation 3 of
    https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf
    
      * i indicates the row under consideration
    
    The many forloops in this function can be optimized, but for
    educational purposes we write them out clearly
    """
    N, C = F.shape

    neighbours = np.where(A[i])
    nneighbours = np.where(1-A[i])

    sum_neigh = np.zeros((C,))
    for nb in neighbours[0]:
        dotproduct = F[nb].dot(F[i])
        sum_neigh += F[nb]*sigm(dotproduct)

    sum_nneigh = np.zeros((C,))
    #Speed up this computation using eq.4
    for nnb in nneighbours[0]:
        sum_nneigh += F[nnb]

    grad = sum_neigh - sum_nneigh
    return grad



def train_labels(A, C, iterations = 100):
    # initialize an F
    N = A.shape[0]
    F = np.random.rand(N,C)

    for n in range(iterations):
        for person in range(N):
            grad = gradient(F, A, person)

            F[person] += 0.005*grad

            F[person] = np.maximum(0.001, F[person]) # F should be nonnegative
        ll = log_likelihood(F, A)
        print('At step %5i/%5i log_likelihood is %5.3f'%(n, iterations, ll))
    return np.argmax(F,1) #F

if __name__ == "__main__":
    # adj = np.load('data/adj.npy')

    #adj = np.load('adj.npy')
    #print("adj", adj, type(adj))
    npz_adj = np.load('../../tmp/cora_meta_adj_0.15.npz')
    attacked_adj = sp.load_npz('../../tmp/cora_meta_adj_0.15.npz')
    attacked_adj = attacked_adj.toarray()
    print(attacked_adj)
    for k in npz_adj.files:
        print(k)
    """indices
       indptr : [    0     5     9 ... 11610 11615 11618] 2486
       format : b'csr'
       shape : [2485 2485]
       data
       """
    #print("npz adj data", npz_adj['data'], npz_adj['indices'], len(npz_adj['indices']))
    #print("format ", npz_adj['format'] )
    #print("indptr", npz_adj['indptr'], len(npz_adj['indptr']))
    
    ## 載入Graph
    adj, features, labels = get_adj()
    adj = adj.toarray()
    print("cora under metattack's adj : \n", adj, type(adj), adj.shape )
    
    
    F = train_labels(adj, 7, iterations=1) #將Graph分成n群 
    print("F", F, "\n", np.argmax(F,1), len(adj[0]), len(np.argmax(F,1)))
    print("data type", type(F), F.shape)
    
    F_argmax = np.argmax(F,1) #F_argmax代表每個node屬於哪一群
    
    