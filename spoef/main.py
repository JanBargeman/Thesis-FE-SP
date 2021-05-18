import os
import pandas as pd

os.chdir("/Users/Jan/Desktop/Thesis/Thesis-FE-SP")

from spoef.features import create_all_features
from spoef.utils import combine_features_dfs, select_features_subset
from spoef.benchmarking import grid_search_LGBM, grid_search_RF, search_mother_wavelet


#%% Read in data
data = pd.read_csv("personal/data/data.csv")
data.date = pd.to_datetime(data.date, format="%y%m%d")
status = data[["account_id", "status"]].drop_duplicates().set_index("account_id")

#%% Make test data
data = data.iloc[0:2000, :]
# data = data[data.account_id == 1787]
# data = data[data.account_id == 276]
# data = data[data.account_id == 1843]

#%% Create features
list_featuretypes = ["B", "F", "W_B", "W"]
mother_wavelet = "db2"

all_features = create_all_features(data, list_featuretypes, mother_wavelet)
B_features = select_features_subset(all_features, ["B"])
SP_features = select_features_subset(all_features, ["F", "W_B", "W"])

#%% Writeout all_features
all_features.to_csv("personal/all_features_db2.csv", index=False)

#%% Read in all_features
all_features = pd.read_csv("personal/all_features.csv", index_col="account_id")


#%% Grid search for optimal parameters
data_all = combine_features_dfs([status, all_features])
data_B = combine_features_dfs([status, B_features])
data_SP = combine_features_dfs([status, SP_features])

best_lgbm_all = grid_search_LGBM(data_all)  # db2: 92.79;    db4: 92.22
best_lgbm_B = grid_search_LGBM(data_B)  # db2: 87.12;    db4: 87.12
best_lgbm_SP = grid_search_LGBM(data_SP)  # db2: 86.24;    db4: 85.29

best_RF_all = grid_search_RF(data_all)  # db2: 88.71;    db4: 86.55
best_RF_B = grid_search_RF(data_B)  # db2: 85.89;    db4: 85.89
best_RF_SP = grid_search_RF(data_SP)  # db2: 86.27;    db4: 84.86


#%% Find optimal mother wavelet with only yearly features
list_mother_wavelets = ["db3", "db5", "db6"]  # db2 0.7834, db4 0.788, 3&5 0.771/0.774
test_size = 0.4

auc_list = search_mother_wavelet(
    data, status, best_lgbm_all, list_mother_wavelets, test_size
)


#%%
# problems:
#   wavelets soms verschillende lengte per maand, neem nu eerste 28 maar wil liever iets beters

# TODO:
#   fft power spectrum?
#   AUC plot ROC om eindes te vergelijken, area of interest
#   computational complexity scaling (On^2?)
#   list_of_transforms = ["ICA", "PCA", "norm"] # transforms gebeuren erbuiten


# scipy.detrending removal of (changing) mean
# scipy signal butter bandpass filter


#%% TRANSFORMS

# list_of_transforms = ["ICA", "PCA", "norm"] # transforms gebeuren erbuiten
# ICA_data = ICA(data)
# PCA_data = PCA(data)
# ICA_1_features_monthly = compute_features_monthly(ICA_data[["date", "ICA_1"]])

# def ICA(data):
#     data_to_transform = data.iloc[:,1:]
#     transformed_data = ICA_global.fit_transform(data_to_transform)
#     transformed_data.columns = ["ICA " + x for x in range(data.shape[1]-1)]
#     return pd.concat([data.iloc[:,0], transformed_data],axis=1)

# def PCA(data):
#     data_to_transform = data.iloc[:,1:]
#     transformed_data = PCA_global.fit_transform(data_to_transform)
#     transformed_data.columns = ["PCA " + x for x in range(data.shape[1]-1)]
#     return pd.concat([data.iloc[:,0], transformed_data],axis=1)
