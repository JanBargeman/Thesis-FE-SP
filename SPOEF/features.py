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

from spoef.utils import prepare_data_monthly, prepare_data_yearly, prepare_data_overall

#%%

dataset = pd.read_csv("personal/data/dataset.csv")
dataset.date = pd.to_datetime(dataset.date, format="%y%m%d")

dataset_orig = dataset.copy()

#%% make test dataset
dataset = dataset.iloc[0:2000,:]
data_acc = dataset[dataset.account_id == 1787]
data_used = data_acc[["date","balance"]]
# dataset = dataset[dataset.account_id == 276]
# dataset = dataset[dataset.account_id == 1843]


#%%

def compute_list_featuretypes(data, list_featuretypes, fourier_n_largest_frequencies, wavelet_depth, mother_wavelet):
    features_basic = pd.DataFrame()
    features_fourier = pd.DataFrame()
    features_wavelet = pd.DataFrame()
    features_wavelet_basic = pd.DataFrame()
    if "B" in list_featuretypes:
        features_basic = compute_basic(data)
    if "F" in list_featuretypes:
        features_fourier = compute_fourier(data, fourier_n_largest_frequencies)
    if "W" in list_featuretypes:
        features_wavelet = compute_wavelet(data, wavelet_depth, mother_wavelet)
    if "W_B" in list_featuretypes:
        features_wavelet_basic = compute_wavelet_basic(data, wavelet_depth, mother_wavelet)
    features = pd.concat([features_basic, features_fourier, features_wavelet, features_wavelet_basic], axis=1)
    return features

def compute_fourier(data, fourier_n_largest_frequencies):
    fft = scipy.fft.fft(data.values)
    fft_abs = abs(fft[range(int(len(data) / 2))])

    larges_indexes = np.argsort(-fft_abs)[:fourier_n_largest_frequencies]
    largest_values = fft_abs[larges_indexes]
    largest_values = [int(a) for a in largest_values]

    features = [*larges_indexes.tolist(), *largest_values]
    col_names_index = ["fft index " + str(i+1) + "/" + str(fourier_n_largest_frequencies) for i in range(int(len(features)/2))]
    col_names_size = ["fft size " + str(i+1) + "/" + str(fourier_n_largest_frequencies) for i in range(int(len(features)/2))]
    col_names = [*col_names_index, *col_names_size]
    features = pd.DataFrame([features], columns=col_names)    
    return features

def compute_basic(data):
    col_names = ['min', 'max', 'mean', 'skew', 'kurt', 'std']
    features = pd.DataFrame([[data.min(), data.max(), data.mean(), data.skew(), data.kurt(), data.std()]], columns=col_names) 
    return features

def compute_wavelet(data, wavelet_depth, mother_wavelet):
    wavelet = pywt.wavedec(data, wavelet=mother_wavelet, level=wavelet_depth)
    features = [item for sublist in wavelet for item in sublist]  # flatten list
    col_names = ["wav " + str(i+1) for i in range(len(features))]
    features = pd.DataFrame([features], columns=col_names)
    return features

def compute_wavelet_basic(data, wavelet_depth, mother_wavelet):
    features = pd.DataFrame()
    data_wavelet = data
    for i in range(wavelet_depth):
        data_wavelet, coeffs = pywt.dwt(data_wavelet, wavelet=mother_wavelet)
        featuresAtDepth = compute_basic(pd.Series(data_wavelet))
        featuresAtDepth.columns = ["wav depth " + str(i+1) + " " + str(col) for col in featuresAtDepth.columns]
        features = pd.concat([features, featuresAtDepth], axis = 1)            
    return features

#%%

def compute_features(data, list_timespans, list_featuretypes):
    features=data
    return features


def compute_features_monthly(data, combine_fill_method, observation_length=12, list_featuretypes=["B"], fourier_n_largest_frequencies=20, wavelet_depth=3, mother_wavelet="db2"):
    
    data = data.drop(data.columns[0],axis=1)#drop account_id_column
    
    prepared_data = prepare_data_monthly(data, combine_fill_method, observation_length)
    
    start_date = prepared_data.iloc[0,0]

    features = pd.DataFrame()
    for month in range(0, observation_length):
        data_month = prepared_data[
            (prepared_data.iloc[:,0] >= start_date + relativedelta(months=month))
            & (prepared_data.iloc[:,0] < start_date + relativedelta(months=month + 1))
        ]
        monthly_features = compute_list_featuretypes(data_month.iloc[:,1], list_featuretypes, fourier_n_largest_frequencies, wavelet_depth, mother_wavelet)
        monthly_features.columns = [data.columns[1][:2] + " M " + str(month+1) + "/" + str(observation_length) + " " + col for col in monthly_features.columns]
        features = pd.concat([features, monthly_features], axis=1)   
    return features


def compute_features_yearly(data, combine_fill_method, observation_length=2, list_featuretypes=["B"], fourier_n_largest_frequencies=40, wavelet_depth=6, mother_wavelet="db2"):
    
    data = data.drop(data.columns[0],axis=1)#drop account_id_column
    
    prepared_data = prepare_data_yearly(data, combine_fill_method, observation_length)
       
    start_date = prepared_data.iloc[0,0]

    features = pd.DataFrame()
    for year in range(0, observation_length):
        data_year = prepared_data[
            (prepared_data.iloc[:,0] >= start_date + relativedelta(years=year))
            & (prepared_data.iloc[:,0] < start_date + relativedelta(years=year + 1))
        ]
        yearly_features = compute_list_featuretypes(data_year.iloc[:,1], list_featuretypes, fourier_n_largest_frequencies, wavelet_depth, mother_wavelet)
        yearly_features.columns = [data.columns[1][:2]+ " Y " + str(year+1) + "/" + str(observation_length) + col for col in features.columns]
        features = pd.concat([features, yearly_features], axis=1)
    return features

def compute_features_overall(data, combine_fill_method, list_featuretypes=["B"], fourier_n_largest_frequencies=60, wavelet_depth=6, mother_wavelet="db2"):
    
    data = data.drop(data.columns[0],axis=1) #drop account_id_column

    prepared_data = prepare_data_overall(data, combine_fill_method)

    features = compute_list_featuretypes(prepared_data.iloc[:,1], list_featuretypes, fourier_n_largest_frequencies, wavelet_depth, mother_wavelet)
    features.columns = [data.columns[1][:2] + " O " + col for col in features.columns]
    return features 


#%%
    
# observation_length=24
# combine_fill_method = "balance"
# list_featuretypes = ["B","F","W","W_B"]

# #%%

# start_date, end_date = utils.determine_observation_period_monthly(data_used.iloc[:,0], observation_length)
# data_filled = utils.fill_empty_dates(data_used, combine_fill_method, start_date, end_date)

# data = data_filled
# balance_features_monthly = compute_features_monthly(data_used[["date","balance"]], "balance", observation_length, list_featuretypes)



# #%%


# test = pd.DataFrame([[1,2],[3,4]])
# test_stack = test.stack().reset_index(drop=True)






# data = data_filled.iloc[:,1]
# fourier_n_largest_frequencies = 20


# #%%

# df_names = ["a", "b"]
# df_list = [pd.DataFrame() for df in df_names]
# df_dict = dict(zip(df_names , df_list))

# #%%

# df_total = pd.DataFrame()
# my_list = ["monthly_featuresB", "monthly_featuresSP", "yearly_featuresB", "yearly_featuresSP", "overall_featuresB", "overall_featuresSP"]
# for item in my_list:
#     df_total = pd.concat([df_total,eval(item)], axis=1)



