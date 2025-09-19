# Quick tutorial

We provide a quick tutorial on how to use our package to compute the distance between a graph pair. This is meant to be a simple use-example to introduce the reader to the main functions and to eventually copy and paste some code if necessary. The details on the inputs and function parameters are discussed in depth in [**the next notebook**](Example.ipynb).

## Installation

Our package can be installed with the `pip` command.

```bash
pip install
```

## Basic use

The temporal graphs are dealt with pandas, as thoroughly described in [**this section**](Example.ipynb).

```python
import pandas as pd
from GDynaDist import Graphs4Distance

# load the two temporal graphs to compare
df1 = pd.read_csv('df1.csv')
df2 = pd.read_csv('df2.csv')

# initialite the Data class
Data = Graphs4Distance()

# add the two datasets to the Data class and give them a name
Data.LoadDataset(df1, 'first_dataset')
Data.LoadDataset(df2, 'second_dataset')

# compute the distance between the two graphs
unmatched_distance = Data.GetDistance('fist_graph', 'second_graph', distance_type = 'unmatched')
```

If the user wants to compute the `matched` distance, the known bijective mapping between the two graphs' nodes has to be specified. This is passed as a dictionary. If the two graphs have the same labels, the distance can be directly computed by running

```python
matched_distance = Data.GetDistance('first_graph', 'second_graph', distance_type = 'matched', node_mapping = 'Same')
```

---

<div style="background-color: #e7f3f9; border-left: 4px solid #2196F3; padding: 12px 16px; border-radius: 6px; font-family: sans-serif; color: #084b63; margin: 1em 0;">
  <span style="font-weight: bold; display: inline-flex; align-items: center;">
    <svg style="height: 16px; width: 16px; margin-right: 6px; fill: #084b63;" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
      <path d="M8 1a7 7 0 100 14A7 7 0 008 1zm.93 10.412c0 .337-.274.588-.61.588h-.64c-.336 0-.61-.251-.61-.588V7.589c0-.337.274-.589.61-.589h.64c.336 0 .61.252.61.589v3.823zM8 5.2a.867.867 0 110-1.734.867.867 0 010 1.733z"/>
    </svg>
    Note:
  </span>
  The function has several parameters that can be adjusted to improve the efficiency of the distance calculation and memory usage.
</div>

You can refer to the [**next tutorial**](Example.ipynb) for a discussion about all parameters.

---

<div style="background-color: #e7f3f9; border-left: 4px solid #2196F3; padding: 12px 16px; border-radius: 6px; font-family: sans-serif; color: #084b63; margin: 1em 0;">
  <span style="font-weight: bold; display: inline-flex; align-items: center;">
    <svg style="height: 16px; width: 16px; margin-right: 6px; fill: #084b63;" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
      <path d="M8 1a7 7 0 100 14A7 7 0 008 1zm.93 10.412c0 .337-.274.588-.61.588h-.64c-.336 0-.61-.251-.61-.588V7.589c0-.337.274-.589.61-.589h.64c.336 0 .61.252.61.589v3.823zM8 5.2a.867.867 0 110-1.734.867.867 0 010 1.733z"/>
    </svg>
    Note:
  </span>
   Remember that a distance is an absolute measure of similarity, not a relative one. When using the distance for temporal graph analysis, it should always be compared with a baseline (for instance with a null model) to understand if the two graphs under analysis are more - or less-similar than expected. 
</div>

You can refer to [**this tutorial**](Experiment.ipynb) for some examples of how the distance can be used for data analysis.