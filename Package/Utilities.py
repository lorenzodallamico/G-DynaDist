import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage, fcluster


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


def ClusterHierarchical(M, k, return_linked = False):
    '''Performs hierarchical clustering on a matrix M'''
    distArray = ssd.squareform(M)
    linked = linkage(M, method = 'ward', metric = 'euclidean')
    est_ℓ = fcluster(linked, k, criterion = 'maxclust')

    if return_linked:
        return est_ℓ, linked
    else:
        return est_ℓ


