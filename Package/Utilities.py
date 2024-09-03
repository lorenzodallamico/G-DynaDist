import pandas as pd
import numpy as np
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


def NMF_kmeans(M,k):

    n, _ = M.shape
    Mt = M + np.eye(n)*np.mean(M[M.nonzero()])
    Mt = Mt/np.mean(Mt)
    Y = NMF(n_components = k).fit(Mt).components_
    kmeans = KMeans(n_clusters = k, n_init = 10).fit(Y.T)
    return kmeans.labels_, np.abs(kmeans.score(Y.T))


def ClusterNMF(M, k):
    
    est_ℓ, score = NMF_kmeans(M, k)
    
    for i in range(20):
        est_ℓ_, score_ = NMF_kmeans(M, k)
        if score_ < score:
            score = score_
            est_ℓ = est_ℓ_
            
    return est_ℓ


def MakeTemporal(dft, timeFR):
    '''This function takes an edge list (dft) in the form of a pandas dataframe and all the time series stored in
    timeFR and creates a temporal graph'''

    idx1 = []
    idx2 = []
    tv = []

    # map each edge to one of the edges from time_FR
    for x in range(len(dft)):
        i, j = dft.iloc[x].values
        q = np.random.randint(len(timeFR))

        for t in timeFR[q]:
            idx1.append(i)
            idx2.append(j)
            tv.append(t)

    # build the temporal dataframe
    DFT = pd.DataFrame(np.array([idx1, idx2, tv]).T, columns = ['i', 'j', 't'])
    DFT.t -= DFT.t.min()
    DFT.t /= 10*60 # 10 minutes time resolution
    DFT.t = DFT.t.astype(int)
    DFT['τ'] = 1
    DFT = DFT.groupby(['i', 'j', 't']).sum().reset_index()
    
    return DFT

def CosSim(x,y):
    '''Cosine similarity between two vectors'''
    return x@y/np.sqrt(x@x*y@y)
