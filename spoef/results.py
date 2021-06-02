import os
import pandas as pd

# import os as os
os.chdir("/Users/Jan/Desktop/Thesis/Thesis-FE-SP")

from spoef.features import create_all_features
from spoef.utils import combine_features_dfs, select_features_subset, write_out_list_dfs
from spoef.benchmarking import grid_search_LGBM, grid_search_RF, search_mother_wavelet
from spoef.transforms import create_all_features_transformed


#%% Read in data
data = pd.read_csv("personal/data/data.csv")
data.date = pd.to_datetime(data.date, format="%y%m%d")
status = data[["account_id", "status"]].drop_duplicates().set_index("account_id")

results_location = "public_czech"

#%% Make test data
data = data.iloc[0:20000, :]
# data = data[data.account_id == 1787]
# data = data[data.account_id == 276]
# data = data[data.account_id == 1843]


#%% Generate regular features

list_featuretypes = ["B", "F", "W", "W_B"]
mother_wavelet = "db2"

regular_features = create_all_features(data, list_featuretypes, mother_wavelet)


#%% Model construction
#%% Grid search for optimal parameters

data_all = combine_features_dfs([status, regular_features])
best_lgbm_all = grid_search_LGBM(data_all)  # db2: 0.9279 db4: 0.9222 ICA: 0.9293 PCA: 0.9310 samen: 0.9225 norm: 0.9377
best_RF_all = grid_search_RF(data_all)      # db2: 0.8871 db4: 0.8655 ICA: 0.8804 PCA: 0.8774 samen: 0.8760 norm: 0.8848

# B_features = select_features_subset(regular_features, ["B"])
# SP_features = select_features_subset(regular_features, ["F", "W", "W_B"])

# data_B = combine_features_dfs([status, B_features])
# data_SP = combine_features_dfs([status, SP_features])

# best_lgbm_B = grid_search_LGBM(data_B)      # db2: 0.8712 db4: 0.8712 ICA: 0.8903 PCA: 0.8664 samen: 0.8854 norm: 0.8805
# best_lgbm_SP = grid_search_LGBM(data_SP)    # db2: 0.8624 db4: 0.8529 ICA: 0.8670 PCA: 0.8707 samen: 0.8654 norm: 0.9151

# best_RF_B = grid_search_RF(data_B)          # db2: 0.8589 db4: 0.8589 ICA: 0.8606 PCA: 0.8469 samen: 0.8544 norm: 0.8567
# best_RF_SP = grid_search_RF(data_SP)        # db2: 0.8627 db4: 0.8486 ICA: 0.8658 PCA: 0.8636 samen: 0.8578 norm: 0.8717

#%% Find optimal mother wavelet with only yearly features
list_mother_wavelets = ["db3", "db5", "db6"]  # db2 0.7834, db4 0.788, 3&5 0.771/0.774
test_size = 0.4

auc_list_lgbm = search_mother_wavelet(
    data, status, best_lgbm_all, list_mother_wavelets, test_size
)
auc_list_rf = search_mother_wavelet(
    data, status, best_RF_all, list_mother_wavelets, test_size
)


best_mother_wavelet = find_mother_wavelet()


#%% Generate all features
mother_wavelet = best_mother_wavelet

regular_features = create_all_features(data, list_featuretypes, mother_wavelet)
normalized_features = create_all_features(data, list_featuretypes, mother_wavelet, normalize=True)
PCA_features = create_all_features_transformed(data, 'PCA', list_featuretypes, mother_wavelet)
ICA_features = create_all_features_transformed(data, 'ICA', list_featuretypes, mother_wavelet)

write_out_list_dfs([regular_features, normalized_features, PCA_features, ICA_features], results_location)

#%% Feature selection





#%% Performance comparison






#%% Benchmarking




#%% overview
'''
- create regular features
- grab regular features: grid search optimal model (5x2cv + McNemar) (take best model, not sign better than others?)
- grab regular features: try different mother wavelets (5x2cv + McNemar) (take best mother, not sign?)
- create all features (normalized, regular, PCA, ICA)
- features selection (transformtypes, featuretypes, windowlengths, featureproperties)
    = toss out part of features, check if better
    = look at feature importances to get an idea?
        + look at selection of features and see for example which depth or Fourier n'th freq is more important?
        + AUC plots for increasing n_largest_freqs to see where it saturates
        + similar for wavelet_depth
- goal: 50 features
- analyze predictive improvement on HOLDOUT (5x2cv + McNemar) (sign better compared to Basic?)
- benchmark selective feature generation time







'''

