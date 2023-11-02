import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import NMF


def tij2tijtau(df, n_jobs = 8, verbose = 4):
    '''Converts a pandas dataframe from the format (i, j, t) to the format (i, j, t, τ)'''
    
    # find all the possible interacting pairs
    df = df.set_index(['i', 'j'])
    df = df.sort_index()
    all_indeces = df.index.unique()

    # run the algorithm in parallel for each pair
    Pl = Parallel(n_jobs = n_jobs, verbose = verbose)
    result = Pl(delayed(tij2tijtauIndex)(df, index) for index in all_indeces)

    dfttau = pd.concat(result)
    dfttau = dfttau.rename(columns = {'t0': 't'})
    dfttau.t = (dfttau.t).astype(int)

    return dfttau


def tij2tijtauIndex(df, index):
    '''Converts a pandas dataframe from the format (i, j, t) to (i, j, t, τ) for a specific pair (i, j)'''
    
    # select the contacts related to a given pair
    ddf = df.loc[index].reset_index()
    
    # find contiguous events
    ddf['diff'] = ddf.t - np.roll(ddf.t,1) - 1
    
    # attribute to all them the same initial time
    v = np.zeros(len(ddf))
    idx = ddf['diff'] != 0
    v[idx] = ddf[idx].t
    ddf['t0'] = v
    ddf.t0 = ddf.t0.replace(0, method='ffill').values
    ddf = ddf[['t', 'i', 'j', 't0']]
    
    # compute the contact duration
    ddf['τ'] = np.ones(len(ddf))
    ddf = ddf.groupby(['t0', 'i', 'j']).sum().reset_index()[['t0', 'i', 'j', 'τ']]
    
    return ddf


def Cluster(M, k):
    '''Performs spectral clustering on a matrix M'''
    
    μ = np.mean(M[M.nonzero()])
    M = np.exp(-(M/μ))
    μ = np.mean(M[M.nonzero()])
    M = M - μ*np.ones(M.shape)
    λ, X = np.linalg.eigh(M)

    idx = np.argsort(np.abs(λ))[::-1]
    X = X[:,idx]

    kmeans = KMeans(n_clusters = k, random_state = 0, n_init = "auto").fit(X[:,:k])
    
    return kmeans.labels_


def ClusterHierarchical(M, k):
    '''Performs hierarchical clustering on a matrix M'''
    distArray = ssd.squareform(M)
    linked = linkage(M, method = 'ward', metric = 'euclidean')
    est_ℓ = fcluster(linked, k, criterion = 'maxclust')

    return est_ℓ

def ClusterNMF(M,k):

    n, _ = M.shape
    Mt = M + np.eye(n)*np.mean(M[M.nonzero()])
    Mt = Mt/np.mean(Mt)
    Y = NMF(n_components = k).fit(Mt).components_
    est_ℓ = KMeans(n_clusters = k, n_init = 10).fit(Y.T).labels_
    return est_ℓ