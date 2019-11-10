# Case Study

Author: Alejandro Notario

Date: November 7, 2019

<hr>

<br>

## Overview

KNN model building to predict the frequency of claims a customer’s going to make with the company.

__2 methods__

- Method 1: Selected customers: with one year period, so the prediction is the claims count for the first year of a customer, it makes more sense than the first method, it is "probability of number of claims for this customer whithin their first policy year.

- Method 2: Selected customers: The wohle dataset. It means that the predictions accuracy could be misleading because the model takes the tiem periods as a variable and in actually the prediction requieres exactly the same period to work right.


## Steps

__Data clean up__

- Exploring data
- Setting up the target
- Imputation
- Handling ouliers

__Feature engineering__

- Scaling 
- Encoding
- New variables

__Modeling__

- Selecting model
- Tunning hyperparameters
- Applying

__Evaluation__

- Comparing predictions vs actual values
- Getting score
- Simulating a prediction game

## Links:

- https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier

- https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

## Environment requirements

Python 3

Libraries:

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier
import warnings
import matplotlib.pyplot as plt
```
## Folders

- __code__: it contains:
	- scripts: model applications (on of each method explained) to run at command line with python "simulator.py". It contains data processing and training for different numbers of neighbors inputs
	with a simulator which asks for a policy id to predict the number of claims group in which it is
	- notebooks: Jupyter notebooks that allows to run the code visualizing results step by step with markdown explanations
- __presentation__: case study presentation as Rpres. To see in your browser run case_study.html with double click from your cloned repository presentation folder once you have downloaded it.

