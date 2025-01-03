# G-DynaDist

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13880695.svg)](https://doi.org/10.5281/zenodo.13880695)


These are the codes related to the paper 
* Dall'Amico, Barrat, Cattuto - *An embedding-based distance for temporal graphs*

These codes implement our definition of distance between temporal graphs and can be used to reproduce the results in the related paper.

> If you make use of these codes, please reference the article

```
@article{dallamico2024embeddingbased,
   title={An embedding-based distance for temporal graphs},
   volume={15},
   ISSN={2041-1723},
   url={http://dx.doi.org/10.1038/s41467-024-54280-4},
   DOI={10.1038/s41467-024-54280-4},
   number={1},
   journal={Nature Communications},
   publisher={Springer Science and Business Media LLC},
   author={Dall’Amico, Lorenzo and Barrat, Alain and Cattuto, Ciro},
   year={2024},
   month=nov }
```

## Dependence on the EDRep package

The main codes to define our distance build upon the `EDRep` function that can be found at https://github.com/lorenzodallamico/EDRep. If you make use of this package, please refer to the original article in which `EDRep` was introduced.

```
@article{dallamico2023efficient,
      title={Efficient distributed representations with linear-time attention scores normalization}, 
      author={Lorenzo Dall'Amico and Enrico Maria Belliardo},
      year={2023},
      eprint={2303.17475},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Content of the package
* `Data/SPData`:  hosts the original contact data as downloaded from the [SocioPatterns](http://www.sociopatterns.org/) web page.

* `Package`: this directory contains the main functions needed to run our simulations. In particular
    * `MatrixDistance` implements the distance between two temporal graphs as proposed in our paper. For reference, see the sections below or the documentation of the functions themselves.
    * `GraphCreation` implements some functions needed to generate random graphs and random shuffles of temporal graphs. 
    * `Utilities` implements some useful functions to deal with the data

* The notebooks can be used to obtain the pictures appearing in the paper. In particular `Recognize graphs` generates Figure 3b, `Recognize synthetic graphs` generates Figure 3a, `Recognize_shufflings` generates Figure 2.

* The notebook `Example` is gives an illustratory example of use of the distance and of the pre-processing that one needs to make on the input data.

## Folders to be added
* `Data/Graphs`: the data that should be inside of it are generated with the notebook named `Prepare_graphs` and they consists in a simple preprocessing of the data contained in `Data/SPData` so to make them completely uniform in terms of format and represent them both in the format `(i,j,t)` and `(i,j,t,τ)`.

* `Data/Shuffled`: the data that should be inside of it are generated with the notebook named `Shuffle`. Once compiled, it generates a sequence of folders each one contianing several shuffles of every graph in the `Graphs` folder for all the shuffling types considered in the paper.

* `Data/Embeddings`: the data that should be put inside of it are generated with the notebook named `Get embeddings`. They contain the embeddings for each temporal graph contained in the `Shuffled` folder.

* `Data/SP_Classes_2_per_day`: the data that should be put inside of it are generated with the notebook named `PrepareSP_day`. They represent the contacts related to the schools in which every dataset corresponds to one day of contact between two classes. The name of the file is in the format `school_name-class_1-class_2-date`.



# Example of use 

Here you can find a summary of the main points needed to deploy our code. A more detailed example is provided in the notebook `Example.ipynb`.

We consider as valid temporal graph inputs only pandas dataframes with columns `i, j, t, τ`. 

> **IMPORTANT NOTE**: the time indices should be expressed in multiples of the time resolution and the initial time should be set to zero. For instance, if the temporal resolution is one minute and the initial time is $17:00$, then the temporal edges $\{(i = 1, j = 2, t = 17:00, τ = 1), (i = 1, j = 2, t = 17:01, τ = 2)\}$ should be represented as $\{(i =1, j = 2, t = 0, τ = 1), (i = 1, j = 2, t = 1, τ = 2)\}$. If this operation is not performed, the codes will be much slower. Moreover, note that $\tau$ is a temporal edge weight a not a duration.


For this type of graph we have three main functions:

* `GraphDynamicEmbedding`: this function computes the embedding of a dynamical graph using the EDRep algorithm

    ```python
    X = GraphDynamicEmbedding(df, n, symmetric, dim, n_epochs, k, verbose, η)
    ```

    The only inputs that must be specified is `df` (the temporal graph) and `n` the number of nodes, while the others are `dim` (the embedding dimensionality), and other parameters related to the *EDRep* algorithm. The output is the array *X*.

* `EmbDistance`: this function computes the distance between two embeddings

    ```python
    d = EmbDistance(X, Y, distance_type)
    ```

    *X, Y* are the two embedding matrices that must have the same number of columns but that can have a different number of rows. The distance type can be `global` or `local` as described in our paper. Note that is `distance_type = local`, the two matrices must have the same size

* `DynamicGraphDistance`: this function computes the distance between two temporal graphs.

    ```python
        d = DynamicGraphDistance(df1, df2, distance_type, symmetric, n1 , n2, dim, n_epochs, k, verbose, η0)
    ```

    This is just a combination of the former two functions.

> **Remark**: the function `DynamicGraphDistance` indeeed implements our distance definition, but we provide also the other two underlying building blocks because it might be convenient to first (or separately) compute all the embedding and then work on them.
    
## Author

[Lorenzo Dall'Amico](https://lorenzodallamico.github.io/) - lorenzo.dallamico@isi.it

## Licence

This software is released under the GNU AFFERO GENERAL PUBLIC LICENSE (see included file LICENSE)
