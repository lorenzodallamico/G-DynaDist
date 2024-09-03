import numpy as np
import random
import itertools
import pandas as pd
from scipy.sparse import csr_matrix
from copy import copy
# from joblib import Parallel, delayed

def generateSequence(outputfolder, model, args, n_graphs, verbose = True, append_name = '.csv'):
    '''This function generates a sequence of graphs that have a common generative model in each block
    
    Use: generateSequence(outputfolder, model, args, n_graphs)
    
    Inputs: 
        * outputfolder : directory to which the files will be saved
        * model (function): function of the generative model to be considered
        * args (list of lists): arguments of model
        * n_graphs (int): number of graphs to be generated

    Optional inputs:
        * verbose (boolean): if True (default) prints the progress bar
        * append_name (string): what goes at the end of the file name. Can be used to change the extension or also part of the name of the file. By default `.csv`
        
    Outputs:
        * this function will output the edgelist representation of the generated graphs numbered in ascending order.       
    '''
    
    for i in range(n_graphs):
            
        df = model(args)
        df.to_csv(outputfolder + '/EL' + str(i+1) + append_name, index = False)

        # print progress bar
        if verbose == 1:
            print("[%-25s] %d%%" % ('='*(int(i/(n_graphs)*25)) + '>', (i+1)/(n_graphs)*100), end = '\r')
        
    return


def DCSBM(args):
    ''' 
    Function that generates a graph from the degree-corrected stochastic block model with
    n nodes and k communities
    
    Note that, with an appropriate choice of the parameters, this function can be used to
    generate an SBM or and ER random graph
    
    Use:
        edge_list = DCSBM(C,c, ℓ, θ, symmetric, make_connected)
    Inputs:
        args = C, c, ℓ, θ
            * C (array of size k x k) : affinity matrix of the network C
            * c (scalar) : average degree of the network
            * ℓ (array of size n) : vector containing the label of each node
            * θ  (array of size n) : vector with the intrinsic probability connection of each node
            * symmetric (bool): if True it returns an undirected graph
            * make_connected (bool): if True forces the graph to be connected adding edges
    
    Outputs:
        * edge_list (pandas dataframe) : edge list representation of the graph (ij)
        
    '''
    
    C, c, ℓ, θ, symmetric, make_connected  = args

    # number of communities
    k = len(np.unique(ℓ))
    
    # number of nodes
    n = len(θ)
    
    # (k x n) matrix where we store the value of the affinity wrt a given label for each node
    c_v = C[ℓ].T
    
    fs = list()
    ss = list()

    # we choose the nodes that should get connected wp = θ_i/n
    if symmetric:
        first = np.random.choice(n,int(n*c/2),p = θ/n)

    else:  
        first = np.random.choice(n,int(n*c),p = θ/n) 

    for i in range(k): 
        v = θ*c_v[i]
        
        # among the nodes of first, select those with label i
        first_selected = first[ℓ[first] == i]
        fs.append(first_selected.tolist())
        
        # choose the nodes to connect to the first_selected
        second_selected = np.random.choice(n,len(first_selected), p = v/np.sum(v))
        ss.append(second_selected.tolist())

    fs = list(itertools.chain(*fs))
    ss = list(itertools.chain(*ss))

    fs = np.array(fs)
    ss  = np.array(ss)


    # add nodes not appearing in the vector first, to ensure the graph is connected
    if make_connected:
        idx = np.arange(n)[np.logical_not(np.isin(np.arange(n), fs))]
        fs = np.concatenate([fs, idx])
        ss = np.concatenate([ss, np.argmax(θ)*np.ones(len(idx))])
    

    # create the edge list from the connection defined earlier
    edge_list = np.column_stack((fs,ss)) 
    
    # remove edges appearing more then once
    edge_list = np.unique(edge_list, axis = 0) 

    # replace self edges
    idx = edge_list[:,0] == edge_list[:,1]
    edge_list[:,1][idx] += 1
    idx = edge_list[:,1] == n
    edge_list[:,1][idx] = 0

    if symmetric:
        df = pd.DataFrame(columns = ['i', 'j'])
        M = np.maximum(edge_list[:,0], edge_list[:,1])
        m = np.minimum(edge_list[:,0], edge_list[:,1])
        df.i = np.concatenate([M, m])
        df.j = np.concatenate([m, M])

    else:
        df = pd.DataFrame(columns = ['i', 'j'])
        df.i = edge_list[:,0]
        df.j = edge_list[:,1]

    # add an edge to nodes with degree 0
    A = csr_matrix((np.ones(len(df)), (df.i, df.j)), shape = (n,n))
    d = A@np.ones(n)
    idx = d == 0


    return df


def GeometricModel(args):
    '''Generates an instance of a modified Geometric model from a given configuration of the points in space
    
    Use: A = GeometricModel(X, d, β)
    
    Inputs: args
        * X (array): coordinates of the points used to form edges
        * d (float): average degree
        * β (float): noise parameter
        
    Ouptput:
        * A (sparse csr_matrix): adjacency matrix of the generated graph'''


    X, d, β = args
    n = np.shape(X)[0]

    # compute the pairwise distance matrix
    no = (X**2)@np.ones(2)
    D = np.zeros((n,n))
    for i in range(n):
        D[i] += no
        D[:,i] += no

    D -= 2*X@X.T
    D = np.sqrt(np.abs(D))

    # compute the probability matrix
    P = np.exp(-β*D)
    P = P - np.diag(np.diag(P))
    p = P@np.ones(n)
    P = np.diag(p**(-1)).dot(P)
    
    # draw the edges
    idx1 = np.concatenate([np.ones(int(d/2))*i for i in range(n)])
    idx2 = np.concatenate([np.random.choice(np.arange(n), int(d/2), p = P[i], replace = False) for i in range(n)])

    df = pd.DataFrame(columns = ['i', 'j'])

    # symmetrize
    df.i = np.concatenate([idx1, idx2])
    df.j = np.concatenate([idx2, idx1])

    return df

def RandomizeGraph(args):
    '''This function uploads a file in the (i, j, t) formats and returns a randomized version of it
    
    Use: df = RandomizeGraph(args)
    
    Inputs: args
        * filename (string): location of the graph file to be updated
        * type_of_shuffling(string): it has to be chosen between
            + `random`: preserves the total number of interactions of the type (i, j, t) but chooses all three indeces at random
            + `random_with_duration`: given a list (i, j, t, τ), it keeps the duration column and randomizes the rest
            + `sequence`: shuffles the order in which the snapshots appear 
            + `time`: shuffles the time column but preserves the interactions in the (i, j, t) format
            + `active_snapshot`: shuffles the edges within each snapshot, but only between active nodes
            + `activity_driven`: preserves the activity of each node
        * tres (int): time resolution. Number of 20 second slices to be aggregated
        
    Output:
        * df (dataframe): shuffled temporal graph in the format (i, j, t, τ)
        
    '''

    DFt, DFttau, type_of_shuffling, tres = args
    dft = copy(DFt)
    dfttau = copy(DFttau)

    # get the network size
    all_nodes = np.unique(dft[['i', 'j']].values)
    n = len(all_nodes)

    # aggregate time
    dft.t = (dft.t/tres).astype(int)
    T = dft.t.max() + 1

    # Shuffle
    if type_of_shuffling == 'random':
        
        df = dft
        idx1 = np.random.choice(np.arange(n), len(df))
        idx2 = np.random.choice(np.arange(n), len(df))
        idxt = np.random.choice(np.arange(T), len(df))
        new_df = pd.DataFrame()
        new_df['i'] = np.minimum(idx1, idx2)
        new_df['j'] = np.maximum(idx1, idx2)
        new_df['t'] = idxt

    elif type_of_shuffling == 'random_delta':

        df = dfttau
        idx1 = np.random.choice(np.arange(n), len(df))
        idx2 = np.random.choice(np.arange(n), len(df))
        idxt = np.random.choice(np.arange(dfttau.t.max()), len(df))
        new_df = pd.DataFrame()
        new_df['i'] = np.minimum(idx1, idx2)
        new_df['j'] = np.maximum(idx1, idx2)
        new_df['t'] = idxt
        new_df['τ'] = df['τ'].values
        new_df = new_df.loc[df.index.repeat(df.τ)]
        new_df['incremental'] = new_df.groupby(level = 0).cumcount()
        new_df.t = new_df.t + new_df.incremental
        new_df = new_df[['i', 'j', 't']]
        new_df.t = (new_df.t/tres).astype(int)
        
    elif type_of_shuffling == 'sequence':

        df = dft
        all_times = df.t.unique()
        shuffled = copy(all_times)
        random.shuffle(shuffled)
        NewTimeMap = dict(zip(all_times, shuffled))
        new_df = pd.DataFrame()
        new_df['i'] = df.i
        new_df['j'] = df.j
        new_df['t'] = df.t.map(lambda x: NewTimeMap[x])
        
    elif type_of_shuffling == 'time':
        
        df = dft
        new_df = pd.DataFrame()
        new_df['i'] = np.minimum(df.i, df.j)
        new_df['j'] = np.maximum(df.i, df.j)
        new_df['t'] = np.random.choice(np.arange(T), len(df))
            
    elif type_of_shuffling == 'active_snapshot':
         
        df = dft
        new_df = df.set_index('t')
        times = new_df.index.unique()

        # number of edges in each time-frame
        Ev = [len(new_df.loc[t]) for t in times]

        # active nodes in each time-frame
        ActiveV = [np.unique(new_df.loc[t][['i', 'j']].to_numpy()) for t in times]
        
        idx1 = np.concatenate([np.random.choice(Active, E) for Active, E in zip(ActiveV, Ev)])
        idx2 = np.concatenate([np.random.choice(Active, E) for Active, E in zip(ActiveV, Ev)])
        idxt = np.concatenate([[t]*E for t, E in zip(times, Ev)])
        new_df = pd.DataFrame()
        new_df['i'] = np.minimum(idx1, idx2)
        new_df['j'] = np.maximum(idx1, idx2)
        new_df['t'] = idxt

    elif type_of_shuffling == 'activity_driven':
        df = dft

        # create a mapping so that between all possible edges and scalars
        mapped = []
        for i in range(n):
            for j in range(i+1,n):
                mapped.append([i,j])
                
        mapped = np.array(mapped)
        mapping = dict(zip(np.arange(len(mapped[:,0])), mapped))

        # number of edges
        E = len(df)

        # create the adjacency matrix of the original graph
        A = csr_matrix((np.ones(len(df)), (df.i, df.j)), shape = (n,n))
        A = A + A.T

        # compute the activity of each node and create a dictionary
        a = A@np.ones(n)
        ActivityDict = dict(zip(np.arange(n), a))

        # associate to each edge a probability to be picked proportional to the product of the activity of the 
        # corresponding nodes
        P = np.array([ActivityDict[x[0]]*ActivityDict[x[1]] for x in mapped])
        P /= np.sum(P)

        # create the edge list at random from this distribution
        new_edges = np.random.choice(np.arange(len(mapped[:,0])), E, p = P)
        new_edges = np.array([mapping[x] for x in new_edges])    

        # create the dataframe
        new_df = pd.DataFrame(columns = ['i', 'j', 't'])
        new_df.i = new_edges[:,0]
        new_df.j = new_edges[:,1]
        new_df.t = np.random.choice(np.arange(T), len(new_df))

        # manually add excluded nodes
        all_new_nodes = np.unique(new_edges)
        excluded = np.arange(n)[np.logical_not(np.isin(np.arange(n), all_new_nodes))]

        for exc in excluded:
            j = np.random.randint(n)
            while j == exc:
                j = np.random.randint(n)

            new_row = {'i': np.min([j, exc]), 'j': np.max([j, exc]), 't': np.random.randint(T)}
            new_df = new_df.append(new_row, ignore_index = True)
            
    else:
        raise DeprecationWarning('Invalid type of shuffling')
    

    # add an interaction weight
    new_df['τ'] = 1
    new_df = new_df.groupby(['i', 'j', 't']).sum().reset_index()

    # if the number of nodes changed, rerun the code
    all_new_nodes = np.unique(new_df[['i', 'j']].values)
    if len(all_new_nodes) != n:
        RandomizeGraph(args)

    return new_df

