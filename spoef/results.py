import os
import pandas as pd
import joblib
from lightgbm import plot_importance

os.chdir("/Users/Jan/Desktop/Thesis/Thesis-FE-SP")

from spoef.feature_generation import create_all_features
from spoef.utils import combine_features_dfs, select_features_subset, take_last_year
from spoef.benchmarking import grid_search_LGBM, grid_search_RF, search_mother_wavelet
from spoef.transforms import create_all_features_transformed
from spoef.feature_selection import return_without_column_types, check_transforms, check_balances_transactions, check_feature_types, check_timewindows, check_months_lengths, check_signal_processing_properties

def write_out_list_dfs(list_dfs, location):
    for item in list_dfs:
        print("Writing %s" %item)
        eval("%s" %item).to_csv(f"{location}/{item}.csv")
    return


#%% Read in data

data = pd.read_csv("personal/data/data.csv")
data.date = pd.to_datetime(data.date, format="%Y-%m-%d")
status = data[["account_id", "status"]].drop_duplicates().set_index("account_id")

results_location = "personal/results/public_czech"

save = False
test = True

#%% Make test data

data = data.iloc[0:2000, :]
# data = data[data.account_id == 1787]
# data = data[data.account_id == 276]
# data = data[data.account_id == 5180]

#%%
data = data.groupby('account_id', as_index=False).apply(take_last_year).reset_index(drop=True)


#%%

list_featuretypes = ["F2"]
base_mother_wavelet = "db2"
features_PCA = create_all_features_transformed(data, 'PCA', list_featuretypes, base_mother_wavelet)



#%% Generate regular features
list_featuretypes = ["B", "F", "W", "W_B", "F2"]
mother_wavelet = "db2"
features_reg = create_all_features(data, list_featuretypes, mother_wavelet)

if save:
    features_reg.to_csv(f"{results_location}/features_reg.csv")

#%%
    
    

#%%
# features_reg.to_csv("personal/temp/features_reg.csv")
# features_reg = pd.read_csv("personal/temp/features_reg.csv", index_col="account_id")

features_reg = pd.read_csv(f"{results_location}/features_reg.csv", index_col="account_id")


#%% Model construction
#%% Grid search for optimal parameters
data_all = combine_features_dfs([status, features_reg])
# data_all.iloc[[1,5,12,7,37],0] = 1
base_lgbm_all, auc_list = grid_search_LGBM(data_all)
# joblib.dump(base_lgbm_all, f"{results_location}/base_lgbm_all.joblib")



#%% Find optimal mother wavelet with only yearly features
list_mother_wavelets = ["db2", "db4", "coif2", "haar", "sym3"]
auc_list_lgbm = search_mother_wavelet(
    data, status, base_lgbm_all, list_mother_wavelets, test
)
list_mother_wavelets = ["db1", "db2", "db3", "db4", "db5", "db6", "db10", "db20"]  
auc_list_lgbm = search_mother_wavelet(
    data, status, base_lgbm_all, list_mother_wavelets, test
)
base_mother_wavelet = "db2"



#%% Generate all features
features_reg = create_all_features(data, list_featuretypes, base_mother_wavelet)
features_norm = create_all_features(data, list_featuretypes, base_mother_wavelet, normalize=True)
features_PCA = create_all_features_transformed(data, 'PCA', list_featuretypes, base_mother_wavelet)
features_ICA = create_all_features_transformed(data, 'ICA', list_featuretypes, base_mother_wavelet)

if save:
    write_out_list_dfs(["features_reg", "features_norm", "features_PCA", "features_ICA"], results_location)

#%% Feature selection

features_reg = pd.read_csv(f"{results_location}/features_reg.csv", index_col="account_id")
features_norm = pd.read_csv(f"{results_location}/features_norm.csv", index_col="account_id")
features_PCA = pd.read_csv(f"{results_location}/features_PCA.csv", index_col="account_id")
features_ICA = pd.read_csv(f"{results_location}/features_ICA.csv", index_col="account_id")


fs_data = combine_features_dfs([status, features_reg, features_norm, features_PCA, features_ICA])

def select_non_default_subset(data, n):
    data_def = data[data.status==1]
    data_non_def = data[data.status==0]
    data_subset_non_def = data_non_def.sample(n=n, random_state=n, axis=0)    
    data_subset = pd.concat([data_def, data_subset_non_def], axis=0)    
    return data_subset
    
fs_data = select_non_default_subset(fs_data, 150)

# if test:
#     fs_data.iloc[[1,5,12,7,37],0] = 1


shap, lgbm = check_transforms(fs_data, debug=True)

fs_data_1 = return_without_column_types(fs_data, ["ICA"], [0])

shap_1, lgbm = check_balances_transactions(fs_data_1)

fs_data_2 = return_without_column_types(fs_data_1, ["tr"], [1])

shap_2, lgbm = check_feature_types(fs_data_2, debug=True)

fs_data_3 = return_without_column_types(fs_data_2, ["fft", "wavelet", "wav_B"], [3,3,3])

shap_3, lgbm = check_timewindows(fs_data_3)

fs_data_4 = return_without_column_types(fs_data_3, ["M", "Y"], [2,2])


#%% 
check_months_lengths(fs_data_4)
fs_data = reduce_months(fs_data)
check_signal_processing_properties(fs_data)
fs_data = reduce_n_largest(fs_data)
fs_data = reduce_wavelet_depth(fs_data)
fs_data = reduce_wav_B_depth(fs_data)

#%% Feature importances
# 
X = fs_data_2.iloc[:, 1:]


plot_importance(lgbm, max_num_features = 10)

feat_importances = pd.Series(lgbm.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')





shap.force_plot(shap_2.expected_value[0], shap_2(X).values[0])
shap.waterfall_plot(shap_2.base_values[0], shap_2(X).values[0], X[0])

test = shap.force_plot(shap_2.expected_value[0], shap_2(X).values[0])




from spoef.feature_selection import return_without_column_types, check_transforms, check_balances_transactions, check_feature_types, check_timewindows, check_months_lengths, check_signal_processing_properties











#%% Performance comparison






#%% Benchmarking
# generating only the relevant features



#%%repeat for Basic features only

B_features = select_features_subset(features_reg, ["B"])
data_B = combine_features_dfs([status, B_features])
base_lgbm_B, auc_list = grid_search_LGBM(data_B)      
# base_RF_B, auc_list = grid_search_RF(data_B)
joblib.dump(base_lgbm_B, "personal/temp/lgbm_b.joblib")





#%% overview
'''
- create regular features
- grab regular features: grid search optimal model (5x2cv + McNemar) (take base model, not sign better than others?)
- grab regular features: try different mother wavelets (5x2cv + McNemar) (take base mother, not sign?)
- create all features (normalized, regular, PCA, ICA)
- features selection (transformtypes, featuretypes, windowlengths, featureproperties)
    = toss out part of features, check if better
    = look at feature importances to get an idea?
        + look at selection of features and see for example which depth or Fourier n'th freq is more important?
        + AUC plots for increasing n_largest_freqs to see where it saturates
        + similar for wavelet_depth
- goal: 50 features
- analyze predictive improvement on HOLDOUT -> 5x2cv + McNemar (sign better compared to Basic?)
- benchmark selective feature generation time







'''

