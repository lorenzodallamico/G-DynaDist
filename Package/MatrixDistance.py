from scipy.sparse import csr_matrix, diags
import numpy as np
from numpy.linalg import eigvalsh
from scipy.linalg import orthogonal_procrustes
from copy import copy



import sys
sys.path += ['your_directory/utils/'] 

from EDRep import CreateEmbedding


import warnings
warnings.filterwarnings("ignore")
    

def GraphDynamicEmbedding(df, n, symmetric = True, dim = 32, n_epochs = 30, k = 1, verbose = False, η = 0.8):
    '''This function computes the embedding of a dynamical graph using the EDRep algorithm

    Use: X = GraphDynamicEmbedding(df, n)

    Inputs:
        * df (pandas dataframe): input dynamic graph with the format `i, j, t, τ`  
        * n (int): number of nodes
        
    Optional inputs:  
        * symmetric (bool): if True (default) is forces the adjacency matrices at each time point to be symmetric
        * dim (int): dimensionality of the embedding. By default set to 32.
        * n_epochs (int): Number of epochs of the training algorithm. By default set to 30.
        * k (int): Order of the Gaussian approximation of p2vec. By default set to 1.
        * verbose (bool): if `True` (default value) it will print the progress status.
        * η (float): learning rate. By default set to 0.8
        
    Output:
        * X (array): embedding vector
    '''

    # group the graphs by time
    df_grouped = df.groupby('t')
    df_idx1 = df_grouped.i.apply(list)
    df_idx2 = df_grouped.j.apply(list)
    df_w = df_grouped.τ.apply(list)
    all_times = list(df_grouped.indices.keys())

    # get the graph matrix for each time-step
    Pt = []

    for t in all_times[::-1]:

        # build the adjacency matrix @t
        A = csr_matrix((df_w.loc[t], (df_idx1.loc[t], df_idx2.loc[t])), shape = (n,n))
        if symmetric:
            A = A + A.T
        A = A + diags(np.ones(n))   

        # get the Laplacian matrix
        d = A@np.ones(n)
        D_1 = diags(d**(-1))
        Pt.append(D_1.dot(A))

    # get the P matrix
    if len(df) > n*(n-1)/2:
        P = [np.sum(np.cumprod(Pt))/len(Pt)]

        # create the embedding
        res = CreateEmbedding(P, dim = dim, n_epochs = n_epochs, k = 1, η = η, verbose = verbose)
    else:
        res = CreateEmbedding(Pt, dim = dim, n_epochs = n_epochs, k = 1, η = η, verbose = verbose, sum_partials = True)

    return res.X


def EmbDistance(X, Y, distance_type = 'unmatched'):
    '''This function computes the distance between 

    Use: d = EmbDistance(X, Y)

    Inputs:
        * X, Y (arrays): input embeddings corresponding to the two temporal graphs. The number of rows of the two matrices must be the same.
        
    Optional inputs:
        * distance_type (string): can be 'unmatched' or 'matched'

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

    
    if distance_type not in ['unmatched', 'matched']:
        raise DeprecationWarning('The distance type is not valid')
    
    else:
        if (distance_type == 'matched') and (n1 != n2):
            raise DeprecationWarning("The input matrices do not have the same size")
        else:
            n = n1

    if distance_type == 'matched':
        Mxx = X.T@X
        Mxy = X.T@Y
        Myy = Y.T@Y
        d = np.sqrt(np.abs(np.linalg.norm(Mxx)**2 + np.linalg.norm(Myy)**2 - 2*np.linalg.norm(Mxy)**2))

    else:
        λ1 = np.linalg.eigvalsh(X.T@X)
        λ2 = np.linalg.eigvalsh(Y.T@Y)
        d = np.linalg.norm(λ1-λ2)
  
    return d


def DynamicGraphDistance(df1, df2, distance_type = 'unmatched', symmetric = True, n1 = None, n2 = None, dim = 32, n_epochs = 30, k = 1, verbose = False, η = 0.8):
    '''This function computes the distance between two temporal graphs

    Use: d = DynamicGraphDistance(df1, df2)

    Inputs:
        * df1, df2 (pandas dataframe): input dynamic graphs with the format `i, j, t, τ`  

    Optional inputs:  
        * distance_type (string): can be 'unmatched' (default) or 'matched'
        * symmetric (bool): if True (default) it forces the adjacency matrices at each time to be symmetric
        * n1 (int): size of the first graph
        * n2 (int): size of the second graph
        * dim (int): dimensionality of the embedding. By default set to 32.
        * n_epochs (int): Number of epochs of the training algorithm. By default set to 30.
        * k (int): Order of the Gaussian approximation of p2vec. By default set to 1.
        * verbose (bool): if `True` (default value) it will print the progress status.
        * η (float): initial learning rate. By default set to 0.8
        
    Output:
        * d (float): graph distance
    '''

    # number of nodes
    if not n1:
        n1 = len(np.unique(df1[['i', 'j']].values))
    if not n2:
        n2 = len(np.unique(df2[['i', 'j']].values))

    # embeddings
    np.random.seed(123)
    X = GraphDynamicEmbedding(df1, n = n1, symmetric = symmetric, dim = dim, n_epochs = n_epochs, k = k, verbose = verbose, η = η)

    np.random.seed(123)
    Y = GraphDynamicEmbedding(df2, n = n2, symmetric = symmetric, dim = dim, n_epochs = n_epochs, k = k, verbose = verbose, η = η)

    # distance
    d = EmbDistance(X, Y,  distance_type = distance_type)

    return d