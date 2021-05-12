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

import pretty_errors

os.chdir("/Users/Jan/Desktop/Thesis/Thesis-FE-SP")

from ..spoef import utils

#%%

dataset = pd.read_csv("personal/data/dataset.csv")
dataset.date = pd.to_datetime(dataset.date, format="%Y-%m-%d")

dataset_orig = dataset.copy()

#%% make test dataset
dataset = dataset.iloc[0:2000,:]
# dataset = dataset[dataset.account_id == 1787]
# dataset = dataset[dataset.account_id == 276]
# dataset = dataset[dataset.account_id == 1843]


#%%
# def determine_monthly_observation_period(data_date, observation_length):
    
#     return start_date, end_date


# def determine_yearly_observation_period(data_date, observation_length):
    
#     return start_date, end_date

# def combine_dates(data, fill_combine_method):
    
#     return combined_dates

# def fill_empty_dates(data, fill_combine_method, start_date, end_date):
    
#     return filled_data

def compute_list_of_featuretypes(data, list_of_featuretypes):
    if "B" in list_of_featuretypes:
        features_basic = compute_basic(data)
    if "F" in list_of_featuretypes:
        features_fourier = compute_fourier(data)
    if "W" in list_of_featuretypes:
        features_wavelet = compute_wavelet(data)
    if "W_B" in list_of_featuretypes:
        features_wavelet_basic = compute_wavelet_basic(data)
    features = pd.concat([features_basic, features_fourier, features_wavelet, features_wavelet_basic], axis=1)
    return features
        
def compute_fourier(data):
    return features

def compute_basic(data):
    return features

def compute_wavelet(data):
    return features

def compute_wavelet_basic(data):
    return features

def split_monthly():
    #groupby?
    return split_data

def split_yearly():
    #groupby?
    return split_data

def transform_data(data, list_of_transforms):
    if "ICA" in list_of_transforms:
        data = [data, ICA(data)]
    if "PCA" in list_of_transforms:
        data = [data, PCA(data)]
    return data

#%%

def compute_features_monthly(data, fill_combine_method, observation_length=12, list_of_featuretypes=["B"], list_of_transforms=[]):
    
    start_date, end_date = determine_monthly_observation_period(data.date, observation_length)
    
    combined_days = combine_days(data, fill_combine_method)
    filled_data = fill_empty_dates(combined_days, fill_combine_method, start_date, end_date)
    
    filled_transformed_data = transform_data(filled_data, list_of_transforms)
    
    for data in filled_transformed_data:
        
        # split

        monthly_features = compute_list_of_featuretypes(data, list_of_featuretypes)
    
    return monthly_features

def compute_features_yearly(data, fill_combine_method, observation_length=12, list_of_featuretypes=["B"], list_of_transforms=[]):
    
    start_date, end_date = determine_yearly_observation_period(data.date, observation_length)
    
    combined_dates = combine_dates(data, fill_combine_method)
    filled_data = fill_empty_dates(combined_dates, fill_combine_method, start_date, end_date)
    
    filled_transformed_data = transform_data(filled_data, list_of_transforms)
    
    for data in filled_transformed_data:
        
        # split

        yearly_features = compute_list_of_featuretypes(data, list_of_featuretypes)
    
    return yearly_features

def compute_features_overall(data, fill_combine_method, list_of_featuretypes=["B"], list_of_transforms=[]):
    
    combined_dates = combine_dates(data, fill_combine_method)
    filled_data = fill_empty_dates(combined_dates, fill_combine_method)
    
    filled_transformed_data = transform_data(filled_data, list_of_transforms)
    
    for data in filled_transformed_data:
        
        # split
        
        overall_features = compute_list_of_featuretypes(data, list_of_featuretypes)
    
    return overall_features    

#%%
    
list_of_featuretypes = ["B","F","W","W_B"]
list_of_transforms = ["ICA", "PCA"]


transaction_features_monthly = compute_features_monthly(data[["date","transaction"]], "transaction", list_of_featuretypes, list_of_transforms)
balance_features_monthly = compute_features_monthly(data[["date","balance"]], "balance", list_of_featuretypes, list_of_transforms)









































































