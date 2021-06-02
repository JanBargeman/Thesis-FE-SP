import os
import pandas as pd

# import os as os
os.chdir("/Users/Jan/Desktop/Thesis/Thesis-FE-SP")

from spoef.features import create_all_features
from spoef.utils import combine_features_dfs, select_features_subset
from spoef.benchmarking import grid_search_LGBM, grid_search_RF, search_mother_wavelet



#%% Read in data
data = pd.read_csv("personal/data/data.csv")
data.date = pd.to_datetime(data.date, format="%y%m%d")
status = data[["account_id", "status"]].drop_duplicates().set_index("account_id")

#%% Make test data
data = data.iloc[0:20000, :]
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
# all_features.to_csv("personal/all_features_normalized.csv")

#%% Read in all_features
# all_features = pd.read_csv("personal/all_features_db2.csv", index_col="account_id")


#%% Grid search for optimal parameters

data_all = combine_features_dfs([status, all_features])
data_B = combine_features_dfs([status, B_features])
data_SP = combine_features_dfs([status, SP_features])

best_lgbm_all, list_all = grid_search_LGBM(data_all)  # db2: 0.9279 db4: 0.9222 ICA: 0.9293 PCA: 0.9310 samen: 0.9225 norm: 0.9377
best_lgbm_B, list_B = grid_search_LGBM(data_B)      # db2: 0.8712 db4: 0.8712 ICA: 0.8903 PCA: 0.8664 samen: 0.8854 norm: 0.8805
best_lgbm_SP, list_SP = grid_search_LGBM(data_SP)    # db2: 0.8624 db4: 0.8529 ICA: 0.8670 PCA: 0.8707 samen: 0.8654 norm: 0.9151

best_RF_all = grid_search_RF(data_all)      # db2: 0.8871 db4: 0.8655 ICA: 0.8804 PCA: 0.8774 samen: 0.8760 norm: 0.8848
best_RF_B = grid_search_RF(data_B)          # db2: 0.8589 db4: 0.8589 ICA: 0.8606 PCA: 0.8469 samen: 0.8544 norm: 0.8567
best_RF_SP = grid_search_RF(data_SP)        # db2: 0.8627 db4: 0.8486 ICA: 0.8658 PCA: 0.8636 samen: 0.8578 norm: 0.8717


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
#   if transforms contain transform dan --> transform functies
#   feature importance


# scipy.detrending removal of (changing) mean
# scipy signal butter bandpass filter


#%% TRANSFORMS

# all_features_normalized = create_all_features(data, list_featuretypes, mother_wavelet, normalize=True)
# all_features_normalized.columns = ["norm " + col for col in all_features_normalized.columns]
# all_features = combine_features_dfs([all_features, all_features_normalized])






#%% random

def get_last_date(data):
    return pd.DataFrame([[data.account_id.iloc[0], data.date.iloc[0]]], columns=["account_id", "final_date"])

test = data.groupby('account_id').apply(get_last_date).reset_index(drop=True)
test = test.merge(status, on="account_id").sort_values("final_date")

amnt = 500
test2 = test.iloc[amnt:,:]
print((682-amnt)/682)
print(test2.status.value_counts()/test.status.value_counts())

list_dif = [a_i - b_i for a_i, b_i in zip(list_all,list_B)] #[0.06329500003186406, 0.069000075079381, 0.06709838339687524, 0.069000075079381, 0.06848070061608902, 0.07137258741935126, 0.07002946695005097, 0.07149106480684897, 0.07119960000591274, 0.07234458189746507, 0.072048849683222, 0.07336155929717758, 0.07209798323393646, 0.07091320208325225, 0.06998132664270496, 0.06907693417280858, 0.06900075757660973, 0.06821406266678764, 0.06828847248604275, 0.06774512882685757, 0.067787941612234, 0.0673231983093936, 0.06755948714003002, 0.06758326357705369, 0.06703081831347402, 0.0665841773329634, 0.06610965713498562, 0.06572781769517178, 0.06526863707087338, 0.06515556612043949, 0.06430321930646743, 0.06416994706430701, 0.06349258440392558, 0.063387630389655, 0.06299671936675955, 0.06298136630389217]
