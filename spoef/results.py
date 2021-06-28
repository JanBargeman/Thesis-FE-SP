import os
import pandas as pd

os.chdir("/Users/Jan/Desktop/Thesis/Thesis-FE-SP")

from spoef.feature_generation import create_all_features
from spoef.utils import combine_features_dfs, take_last_year, get_reduced_data, count_occurences_features
from spoef.benchmarking import grid_search_LGBM, search_mother_wavelet, gridsearchLGBM, gridsearchRF
from spoef.transforms import create_all_features_transformed
from spoef.feature_selection import shap_fs, return_without_column_types, assess_5x2cv, assess_McNemar

def write_out_list_dfs(list_dfs, location):
    for item in list_dfs:
        print("Writing %s" %item)
        eval("%s" %item).to_csv(f"{location}/{item}.csv")
    return

# pd.set_option("display.max_rows", 20, "display.max_columns", None)
# pd.reset_option("display.max_rows", "display.max_columns")

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

#%% Generate regular features
list_featuretypes = ["B", "F", "W", "W_B", "F2"]
mother_wavelet = "db2"
features_reg = create_all_features(data, list_featuretypes, mother_wavelet)

if save:
    features_reg.to_csv(f"{results_location}/features_reg.csv")

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


# #%%
# features_reg.columns = [col.replace("fft2", "f2") for col in features_reg.columns]    
# features_norm.columns = [col.replace("fft2", "f2") for col in features_norm.columns]    
# features_PCA.columns = [col.replace("fft2", "f2") for col in features_PCA.columns]    
# features_ICA.columns = [col.replace("fft2", "f2") for col in features_ICA.columns]    

#%% Feature selection

features_reg = pd.read_csv(f"{results_location}/features_reg.csv", index_col="account_id")
features_norm = pd.read_csv(f"{results_location}/features_norm.csv", index_col="account_id")
features_PCA = pd.read_csv(f"{results_location}/features_PCA.csv", index_col="account_id")
features_ICA = pd.read_csv(f"{results_location}/features_ICA.csv", index_col="account_id")


fs_data = combine_features_dfs([status, features_reg, features_norm, features_PCA, features_ICA])

#%%
def select_non_default_subset(data, n):
    data_def = data[data.status==1]
    data_non_def = data[data.status==0]
    data_subset_non_def = data_non_def.sample(n=n, random_state=n, axis=0)
    data_subset = pd.concat([data_def, data_subset_non_def], axis=0)
    return data_subset
    
fs_data = select_non_default_subset(fs_data, 100)


#%% SHAP




#%%

#%% Feature Selection

data_all = fs_data
data_B = return_without_column_types(fs_data, ["fft", "f2", "wavelet", "wav_B"], [3,3,3,3])
data_SP = return_without_column_types(fs_data, ["B"], [3,3,3])

#%% Feature selection: basic features

# Random Forest: round 1
shap_elim_B_RF_1 = shap_fs(data_B, 'RF', step=0.2)
set_of_feats_B_RF_1 = shap_elim_B_RF_1.get_reduced_features_set(num_features=24)
data_B_RF_1 = get_reduced_data(data_B, set_of_feats_B_RF_1)

# Random Forest: round 2
shap_elim_B_RF_2 = shap_fs(data_B_RF_1, 'RF', step=0.1)
set_of_feats_B_RF_2 = shap_elim_B_RF_2.get_reduced_features_set(num_features=9)
data_B_RF_2 = get_reduced_data(data_B, set_of_feats_B_RF_2)

# LightGBM: round 1
shap_elim_B_LGBM_1 = shap_fs(data_B, 'LGBM', step=0.2)
set_of_feats_B_LGBM_1 = shap_elim_B_LGBM_1.get_reduced_features_set(num_features=24)
data_B_LGBM_1 = get_reduced_data(data_B, set_of_feats_B_LGBM_1)

# LightGBM: round 2
shap_elim_B_LGBM_2 = shap_fs(data_B_LGBM_1, 'LGBM', step=0.1)
set_of_feats_B_LGBM_2 = shap_elim_B_LGBM_2.get_reduced_features_set(num_features=13)
data_B_LGBM_2 = get_reduced_data(data_B, set_of_feats_B_LGBM_2)

count_occurences_features(data_B_RF_2.columns)
count_occurences_features(data_B_LGBM_2.columns)


#%% Feature selection: signal processing features

# Random Forest: round 1
shap_elim_SP_RF_1 = shap_fs(data_SP, 'RF', step=0.3)
set_of_feats_SP_RF_1 = shap_elim_SP_RF_1.get_reduced_features_set(num_features=114)
data_SP_RF_1 = get_reduced_data(data_SP, set_of_feats_SP_RF_1)

# Random Forest: round 2
shap_elim_SP_RF_2 = shap_fs(data_SP_RF_1, 'RF', step=0.2)
set_of_feats_SP_RF_2 = shap_elim_SP_RF_2.get_reduced_features_set(num_features=60)
data_SP_RF_2 = get_reduced_data(data_SP, set_of_feats_SP_RF_2)

# Random Forest: round 3
shap_elim_SP_RF_3 = shap_fs(data_SP_RF_2, 'RF', step=0.1)
set_of_feats_SP_RF_3 = shap_elim_SP_RF_3.get_reduced_features_set(num_features=10)
data_SP_RF_3 = get_reduced_data(data_SP, set_of_feats_SP_RF_3)

# LightGBM: round 1
shap_elim_SP_LGBM_1 = shap_fs(data_SP, 'LGBM', step=0.3)
set_of_feats_SP_LGBM_1 = shap_elim_SP_LGBM_1.get_reduced_features_set(num_features=669)
data_SP_LGBM_1 = get_reduced_data(data_SP, set_of_feats_SP_LGBM_1)

# LightGBM: round 2
shap_elim_SP_LGBM_2 = shap_fs(data_SP_LGBM_1, 'LGBM', step=0.2)
set_of_feats_SP_LGBM_2 = shap_elim_SP_LGBM_2.get_reduced_features_set(num_features=39)
data_SP_LGBM_2 = get_reduced_data(data_SP, set_of_feats_SP_LGBM_2)

# LightGBM: round 3
shap_elim_SP_LGBM_3 = shap_fs(data_SP_LGBM_2, 'LGBM', step=0.1)
set_of_feats_SP_LGBM_3 = shap_elim_SP_LGBM_3.get_reduced_features_set(num_features=9)
data_SP_LGBM_3 = get_reduced_data(data_SP, set_of_feats_SP_LGBM_3)

count_occurences_features(data_SP_RF_3.columns)
count_occurences_features(data_SP_LGBM_3.columns)


#%% Feature selection: all features

# Random Forest: round 1
shap_elim_all_RF_1 = shap_fs(data_all, 'RF', step=0.3)
set_of_feats_all_RF_1 = shap_elim_all_RF_1.get_reduced_features_set(num_features=973)
data_all_RF_1 = get_reduced_data(data_all, set_of_feats_all_RF_1)

# Random Forest: round 2
shap_elim_all_RF_2 = shap_fs(data_all_RF_1, 'RF', step=0.2)
set_of_feats_all_RF_2 = shap_elim_all_RF_2.get_reduced_features_set(num_features=44)
data_all_RF_2 = get_reduced_data(data_all, set_of_feats_all_RF_2)

# Random Forest: round 3
shap_elim_all_RF_3 = shap_fs(data_all_RF_2, 'RF', step=0.1)
set_of_feats_all_RF_3 = shap_elim_all_RF_3.get_reduced_features_set(num_features=8)
data_all_RF_3 = get_reduced_data(data_all, set_of_feats_all_RF_3)

# LightGBM: round 1
shap_elim_all_LGBM_1 = shap_fs(data_all, 'LGBM', step=0.3)
set_of_feats_all_LGBM_1 = shap_elim_all_LGBM_1.get_reduced_features_set(num_features=478)
data_all_LGBM_1 = get_reduced_data(data_all, set_of_feats_all_LGBM_1)

# LightGBM: round 2
shap_elim_all_LGBM_2 = shap_fs(data_all_LGBM_1, 'LGBM', step=0.2)
set_of_feats_all_LGBM_2 = shap_elim_all_LGBM_2.get_reduced_features_set(num_features=102)
data_all_LGBM_2 = get_reduced_data(data_all, set_of_feats_all_LGBM_2)

# LightGBM: round 3
shap_elim_all_LGBM_3 = shap_fs(data_all_LGBM_2, 'LGBM', step=1)
set_of_feats_all_LGBM_3 = shap_elim_all_LGBM_3.get_reduced_features_set(num_features=17)
data_all_LGBM_3 = get_reduced_data(data_all, set_of_feats_all_LGBM_3)

count_occurences_features(data_all_RF_3.columns)
count_occurences_features(data_all_LGBM_3.columns)


#%% Performance comparison

# LGBM
fs_data_B_LGBM_final = data_B_LGBM_2
fs_data_SP_LGBM_final = data_all_LGBM_3
fs_data_all_LGBM_final = data_SP_LGBM_3

# gridsearch for LGBM for B, SP, all feature sets
lgbm_B = gridsearchLGBM(fs_data_B_LGBM_final, cv=2)
lgbm_SP = gridsearchLGBM(fs_data_SP_LGBM_final, cv=2)
lgbm_all = gridsearchLGBM(fs_data_all_LGBM_final, cv=2)

# performance comparison
assess_5x2cv(fs_data_B_LGBM_final, fs_data_SP_LGBM_final, lgbm_B, lgbm_SP)
assess_5x2cv(fs_data_B_LGBM_final, fs_data_all_LGBM_final, lgbm_B, lgbm_all)
assess_5x2cv(fs_data_all_LGBM_final, fs_data_SP_LGBM_final, lgbm_all, lgbm_SP)

assess_McNemar(fs_data_B_LGBM_final, fs_data_SP_LGBM_final, lgbm_B, lgbm_SP)
assess_McNemar(fs_data_B_LGBM_final, fs_data_all_LGBM_final, lgbm_B, lgbm_all)
assess_McNemar(fs_data_all_LGBM_final, fs_data_SP_LGBM_final, lgbm_all, lgbm_SP)



# Random Forest
fs_data_B_RF_final = data_B_RF_2
fs_data_SP_RF_final = data_all_RF_3
fs_data_all_RF_final = data_SP_RF_3

# gridsearch for RF for B, SP, all feature sets
RF_B = gridsearchRF(fs_data_B_RF_final, cv=2)
RF_SP = gridsearchRF(fs_data_SP_RF_final, cv=2)
RF_all = gridsearchRF(fs_data_all_RF_final, cv=2)

# performance comparison
assess_5x2cv(fs_data_B_RF_final, fs_data_SP_RF_final, RF_B, RF_SP)
assess_5x2cv(fs_data_B_RF_final, fs_data_all_RF_final, RF_B, RF_all)
assess_5x2cv(fs_data_all_RF_final, fs_data_SP_RF_final, RF_all, RF_SP)

assess_McNemar(fs_data_B_RF_final, fs_data_SP_RF_final, RF_B, RF_SP)
assess_McNemar(fs_data_B_RF_final, fs_data_all_RF_final, RF_B, RF_all)
assess_McNemar(fs_data_all_RF_final, fs_data_SP_RF_final, RF_all, RF_SP)



#%% Benchmarking
# generating only the relevant features




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


TODO:
    - logistic regression
    - 




'''

