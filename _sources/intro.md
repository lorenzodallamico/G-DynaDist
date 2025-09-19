![](logo_with_text.png)

<font size="5"><span style = "color:#6e9db1"> Dall'Amico, Barrat, Cattuto<br />

# Welcome!

This is the documentation for the `GDyanDist` Python package, which enables the computation of distances between pairs of temporal graphs. For further information, please refer to the [**article**](https://www.nature.com/articles/s41467-024-54280-4).



## Installation

You can install the `GDynaDist` package using pip by running the following command in the terminal.

```bash
pip install
```

We also shared an `Anaconda` environment in which all codes were run and tested. You can create it by running the following commands in the terminal

```bash
conda env create -f EDRep_env.yml
conda activate EDRep
```

## Content

In the following pages, we describe the main features implemented in this package and provide some use examples.

```{tableofcontents}
```

## Dependency on `EDRep`

The `GDynaDist` package relies on the `EDRep` algorithm to compute temporal graph embeddings. Here, the user can find the related paper and package documentation.

| **[Documentation](https://lorenzodallamico.github.io/EDRep/intro.html)** 
| **[Paper](https://openreview.net/pdf?id=9M4NKMZOPu)** | 


## Citation

If you make use of these codes, please use the following citations:

This is the [**Nature Communications article**](https://www.nature.com/articles/s41467-024-54280-4) in which the distance was proposed.

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


This is the [**TMLR paper**](https://openreview.net/pdf?id=9M4NKMZOPu) in which we developed the EDRep algorithm on which the distance definition relies.

``` 
@article{
dall'amico2025learning,
title={Learning distributed representations with efficient SoftMax normalization},
author={Lorenzo Dall'Amico and Enrico Maria Belliardo},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=9M4NKMZOPu},
note={}
}
```