from scipy.sparse import csr_matrix, diags
import numpy as np
from numpy.linalg import eigvalsh
from scipy.linalg import orthogonal_procrustes
from copy import copy


import sys
sys.path += ['/home/lorenzo/Documenti/GitHub/p2vec/Package'] 

from edr import CreateEmbedding


import warnings
warnings.filterwarnings("ignore")
    

def GraphDynamicEmbedding(dft, nodes = [None], dim = 32, n_epochs = 20, k = 1, verbose = False, η0 = 0.85,):
    '''This function computes the embedding of a dynamical graph using the EDRep algorithm

    Use: X = GraphDynamicEmbedding(df, dim, n_epochs, k, verbose, η0)

    Inputs:
        * df (pandas dataframe): input dynamic graph with the format `i, j, t, τ`  

    Optional inputs:  
        * dim (int): dimensionality of the embedding. By default set to 16.
        * n_epochs (int): Number of epochs of the training algorithm. By default set to 8.
        * k (int): Order of the Gaussian approximation of p2vec. By default set to 1.
        * verbose (bool): if `True` (default value) it will print the progress status.
        * η0 (float): initial learning rate. By default set to 0.85
        
    Output:
        * X (array): embedding of the temporal graph
    '''

    df = copy(dft)
    del dft
    
    # compute the duration and number of nodes
    if nodes[0]:
        nodes_scalar = np.arange(len(nodes))
        NodesMapper = dict(zip(nodes, nodes_scalar))
    else:
        all_nodes = np.unique(df[['i', 'j']].values)
        nodes_scalar = np.arange(len(all_nodes))
        NodesMapper = dict(zip(all_nodes, nodes_scalar))

    df.i = df.i.map(lambda x: NodesMapper[x])
    df.j = df.j.map(lambda x: NodesMapper[x])
    n = len(NodesMapper.keys())
    
    df = df.set_index('t')

    # sort time in ascending order
    all_times = np.sort(df.index.unique())
    T = len(all_times)

    At = []

    for t in all_times:

        # select the active nodes @t
        dft = df.loc[t]

        try:
            length = len(dft.i)
            idx1 = np.minimum(dft.i, dft.j)
            idx2 = np.maximum(dft.i, dft.j)
            w = dft.τ.values
        except TypeError:
            length = 1
            idx1 = [np.min([dft.i, dft.j])]
            idx2 = [np.max([dft.i, dft.j])]
            w = [dft.τ]
    
        # build the adjacency matrix @t
        A = csr_matrix((w, (idx1, idx2)), shape = (n,n))
        A = A + A.T
        A = A + diags(np.ones(n))
        At.append(A)
        
    # get the Laplacian matrices for all times
    dt = [A@np.ones(n) for A in At]
    Dt_1 = [diags(d**(-1)) for d in dt]
    Pt = [D_1.dot(A) for D_1, A in zip(Dt_1, At)]

    # get the P matrix
    if len(dft) > n**2:
        P = [np.sum(np.cumprod(Pt))/len(Pt)]

        # create the embedding
        X = CreateEmbedding(P, dim = dim, n_epochs = n_epochs, k = 1, η = η0, verbose = verbose, cov_type = 'full')
    else:
        X = CreateEmbedding(Pt, dim = dim, n_epochs = n_epochs, k = 1, η = η0, verbose = verbose, cov_type = 'full', sum_partials = True)
    
    return X
   

def EmbDistance(X, Y, distance_type = 'global'):
    '''This function computes the distance between 

    Use: d = EmbDistance(X, Y)

    Inputs:
        * X, Y (arrays): input embeddings corresponding to the two temporal graphs. The number of rows of the two matrices must be the same.
        
    Optional inputs:
        * distance_type (string): can be 'global' or 'local'

    Output:
        * d (float): distance between the two graphs.
    '''

    n1, d1 = X.shape
    n2, d2 = Y.shape

    # run initial checks

    if d1 != d2:
        raise DeprecationWarning('The embedding matrices have different dimensions')
    else:
        d = d1

    
    if distance_type not in ['global', 'local']:
        raise DeprecationWarning('The distance type is not valid')
    
    else:
        if (distance_type == 'local') and (n1 != n2):
            raise DeprecationWarning("The input matrices do not have the same size")
        else:
            n = n1

    if distance_type == 'local':
        Mxx = X.T@X
        Mxy = X.T@Y
        Myy = Y.T@Y
        d = np.sqrt(np.abs(np.linalg.norm(Mxx)**2 + np.linalg.norm(Myy)**2 - 2*np.linalg.norm(Mxy)**2))

    else:
        λ1 = np.linalg.eigvalsh(X.T@X)
        λ2 = np.linalg.eigvalsh(Y.T@Y)
        d = np.linalg.norm(λ1-λ2)
  
    return d


def DynamicGraphDistance(df1, df2, distance_type = 'global', dim = 32, n_epochs = 20, k = 1, verbose = False, η0 = 0.85):
    '''This function computes the distance between two temporal graphs

    Use: d = DynamicGraphDistance(df1, df2)

    Inputs:
        * df1, df2 (pandas dataframe): input dynamic graphs with the format `i, j, t, τ`  

    Optional inputs:  
        * distance_type (string): can be 'global' (default) or 'local'
        * dim (int): dimensionality of the embedding. By default set to 16.
        * n_epochs (int): Number of epochs of the training algorithm. By default set to 8.
        * k (int): Order of the Gaussian approximation of p2vec. By default set to 1.
        * verbose (bool): if `True` (default value) it will print the progress status.
        * η0 (float): initial learning rate. By default set to 0.85
        
    Output:
        * d (float): graph distance
    '''

    X = GraphDynamicEmbedding(df1, dim = dim, n_epochs = n_epochs, k = k, verbose = verbose, η0 = η0)
    Y = GraphDynamicEmbedding(df2, dim = dim, n_epochs = n_epochs, k = k, verbose = verbose, η0 = η0)

    d = EmbDistance(X, Y,  distance_type = distance_type)

    return d