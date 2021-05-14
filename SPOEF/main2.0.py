import os
import pandas as pd
from dateutil.relativedelta import relativedelta
import timeit
import scipy.signal  # .stft #.argrelmax  for finding max in plots or smth
import scipy.fft
import numpy as np
import matplotlib.pyplot as plt
import pywt
from functools import reduce
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from sklearn import decomposition

os.chdir("/Users/Jan/Desktop/Thesis/Thesis-FE-SP")

# from spoef import features
# from spoef import utils

from spoef.features import compute_features_monthly, compute_features_yearly, compute_features_overall


#%%

dataset = pd.read_csv("personal/data/dataset.csv")
dataset.date = pd.to_datetime(dataset.date, format="%y%m%d")

dataset_orig = dataset.copy()

#%% make test dataset
dataset = dataset.iloc[0:2000,:]
# dataset = dataset[dataset.account_id == 1787]
# dataset = dataset[dataset.account_id == 276]
# dataset = dataset[dataset.account_id == 1843]


#%%

list_featuretypes = ["B","F","W_B"]
list_featuretypes = ["B"]


# data = dataset

# transaction_features_monthly = compute_features_monthly(data[["date","transaction"]], "transaction", list_featuretypes=list_of_featuretypes)
# balance_features_monthly = compute_features_monthly(data[["date","balance"]], "balance", list_of_featuretypes)
current = timeit.default_timer()

transaction_features_monthly = dataset[["account_id", "date", "transaction"]].groupby("account_id").apply(compute_features_monthly, combine_fill_method="transaction", list_featuretypes=list_featuretypes).reset_index(level=1, drop=True)
print(timeit.default_timer() - current); current = timeit.default_timer()







#%% TRANSFORMS

# list_of_transforms = ["ICA", "PCA"] # transforms gebeuren erbuiten 
# ICA_data = ICA(data)
# PCA_data = PCA(data)
# ICA_1_features_monthly = compute_features_monthly(ICA_data[["date", "ICA_1"]])



































































