import os
import pandas as pd
import matplotlib.pyplot as plt
import timeit

os.chdir("/Users/Jan/Desktop/Thesis/Thesis-FE-SP")

from spoef.feature_generation import create_all_features
from spoef.utils import shap_elim_to_reduce, select_non_default_subset_data, combine_features_dfs, take_last_year, get_reduced_data, count_occurences_features, fill_empty_dates
from spoef.benchmarking import search_mother_wavelet, gridsearchLGBM, gridsearchRF
from spoef.transforms import create_all_features_transformed
from spoef.feature_selection import shap_fs, assess_5x2cv, assess_McNemar, return_with_only_column_types

results_location = "personal/results/public_czech"

def write_out_list_dfs(list_dfs, location):
    for item in list_dfs:
        print("Writing %s" %item)
        eval("%s" %item).to_csv(f"{location}/{item}.csv")
    return

def save_shap_elim_plot(shap_elim, name):
    plot = shap_elim.plot()
    fig = plot.get_figure()
    fig.savefig(f"{results_location}/figures/{name}", dpi=200)
    return

#%% Read in data
data = pd.read_csv("personal/data/data.csv")
data.date = pd.to_datetime(data.date, format="%Y-%m-%d")
status = data[["account_id", "status"]].drop_duplicates().set_index("account_id")

save = False

#%% 3.3 & 4.1.1 - Preprocessing
# 4.1.1 - Grab last year
data = data.groupby('account_id', as_index=False).apply(take_last_year).reset_index(drop=True)
# 3.3 - Grab all defaults and equal amount of non-defaults
data = select_non_default_subset_data(data, 1)


#%% 5.1.2 - Find optimal mother wavelet
list_mother_wavelets_various = ["db2", "db4", "coif2", "sym3"]
mother_wavelets_performance_various = search_mother_wavelet(
    data, status, list_mother_wavelets_various
) # ['0.8667 (0.028981)', '0.8667 (0.028981)', '0.8667 (0.028981)', '0.8667 (0.028981)']

list_mother_wavelets_debauchies = ["db3", "db5", "db6", "db10"]  
mother_wavelets_performance_debauchies = search_mother_wavelet(
    data, status, list_mother_wavelets_debauchies
) # ['0.8667 (0.028981)', '0.8667 (0.028981)', '0.8667 (0.028981)', '0.8667 (0.028981)']


#%% 5.1.1 - Generate all features
list_featuretypes = ["Basic", "FourierComplete", "FourierNLargest", "WaveletComplete", "WaveletBasic"]
mother_wavelet = "db2"


current = timeit.default_timer() 
features_reg = create_all_features(data, list_featuretypes, mother_wavelet, normalize=False)
print("Regular:", int(timeit.default_timer() - current), "seconds") # 66 s
features_norm = create_all_features(data, list_featuretypes, mother_wavelet, normalize=True)
features_PCA = create_all_features_transformed(data, 'PCA', list_featuretypes, mother_wavelet)
features_ICA = create_all_features_transformed(data, 'ICA', list_featuretypes, mother_wavelet)
print("All:", int(timeit.default_timer() - current), "seconds") # 270 s

if save:
    write_out_list_dfs(["features_reg", "features_norm", "features_PCA", "features_ICA"], results_location)  

#%% 5.1.1 - Read in data
features_reg = pd.read_csv(f"{results_location}/features_reg.csv", index_col="account_id")
features_norm = pd.read_csv(f"{results_location}/features_norm.csv", index_col="account_id")
features_PCA = pd.read_csv(f"{results_location}/features_PCA.csv", index_col="account_id")
features_ICA = pd.read_csv(f"{results_location}/features_ICA.csv", index_col="account_id")


#%% 5.1.3 Feature Selection
feature_selection_data = combine_features_dfs([status, features_reg, features_norm, features_PCA, features_ICA])

Basic_set = return_with_only_column_types(feature_selection_data, ["reg", "Basic"], [0,3])
Regular_set = return_with_only_column_types(feature_selection_data, ["reg"], [0])
All_set = feature_selection_data

#%% 5.1.3 - Grid search 
cv = 5

B_lgbm = gridsearchLGBM(Basic_set, cv=cv)
B_RF = gridsearchRF(Basic_set, cv=cv)

Reg_lgbm_L = gridsearchLGBM(Regular_set, cv=cv)
Reg_RF_L = gridsearchRF(Regular_set, cv=cv)

all_lgbm_L = gridsearchLGBM(All_set, cv=cv)
all_RF_L = gridsearchRF(All_set, cv=cv)

#%% 5.1.3 - Feature selection: Basic set
# Random Forest: round 1
shap_elim_B_RF_1 = shap_fs(Basic_set, B_RF, step=0.2)
Basic_set_RF_1 = shap_elim_to_reduce(Basic_set, shap_elim_B_RF_1, 24)

# Random Forest: round 2
shap_elim_B_RF_2 = shap_fs(Basic_set_RF_1, B_RF, step=0.1)
Basic_set_RF_2 = shap_elim_to_reduce(Basic_set_RF_1, shap_elim_B_RF_2, 10)


# LightGBM: round 1
shap_elim_B_LGBM_1 = shap_fs(Basic_set, B_lgbm, step=0.2)
Basic_set_LGBM_1 = shap_elim_to_reduce(Basic_set, shap_elim_B_LGBM_1, 20)

# LightGBM: round 2
shap_elim_B_LGBM_2 = shap_fs(Basic_set_LGBM_1, B_lgbm, step=0.1)
Basic_set_LGBM_2 = shap_elim_to_reduce(Basic_set_LGBM_1, shap_elim_B_LGBM_2, 8)

count_occurences_features(Basic_set_RF_2.columns)
count_occurences_features(Basic_set_LGBM_2.columns)

#%% 5.1.3 - Feature selection: Regular set
# Random Forest: round 1
shap_elim_Reg_RF_1 = shap_fs(Regular_set, Reg_RF_L, step=0.3)
Regular_set_RF_1 = shap_elim_to_reduce(Regular_set, shap_elim_Reg_RF_1, 229)

# Random Forest: round 2
shap_elim_Reg_RF_2 = shap_fs(Regular_set_RF_1, Reg_RF_L, step=0.2)
Regular_set_RF_2 = shap_elim_to_reduce(Regular_set_RF_1, shap_elim_Reg_RF_2, 26)

Reg_RF_s = gridsearchRF(Regular_set_RF_2, cv=cv)

# Random Forest: round 3
shap_elim_Reg_RF_3 = shap_fs(Regular_set_RF_2, Reg_RF_s, step=0.1)
Regular_set_RF_3 = shap_elim_to_reduce(Regular_set_RF_2, shap_elim_Reg_RF_3, 10)


# LightGBM: round 1
shap_elim_Reg_LGBM_1 = shap_fs(Regular_set, Reg_lgbm_L, step=0.3)
Regular_set_LGBM_1 = shap_elim_to_reduce(Regular_set, shap_elim_Reg_LGBM_1, 161)

# LightGBM: round 2
shap_elim_Reg_LGBM_2 = shap_fs(Regular_set_LGBM_1, Reg_lgbm_L, step=0.2)
save_shap_elim_plot(shap_elim_Reg_LGBM_2, 'shap_elim_Reg_LGBM_2')
Regular_set_LGBM_2 = shap_elim_to_reduce(Regular_set_LGBM_1, shap_elim_Reg_LGBM_2, 36)

Reg_lgbm_s = gridsearchLGBM(Regular_set_LGBM_2, cv=cv)

# LightGBM: round 3
shap_elim_Reg_LGBM_3 = shap_fs(Regular_set_LGBM_2, Reg_lgbm_s, step=0.1)
save_shap_elim_plot(shap_elim_Reg_LGBM_3, 'shap_elim_Reg_LGBM_3')
Regular_set_LGBM_3 = shap_elim_to_reduce(Regular_set_LGBM_2, shap_elim_Reg_LGBM_3, 9)

count_occurences_features(Regular_set_RF_3.columns)
count_occurences_features(Regular_set_LGBM_3.columns)

#%% 5.1.3 - Feature selection: All set
# Random Forest: round 1
shap_elim_all_RF_1 = shap_fs(All_set, all_RF_L, step=0.3)
All_set_RF_1 = shap_elim_to_reduce(All_set, shap_elim_all_RF_1, 446)

# Random Forest: round 2
shap_elim_all_RF_2 = shap_fs(All_set_RF_1, all_RF_L, step=0.2)
All_set_RF_2 = shap_elim_to_reduce(All_set_RF_1, shap_elim_all_RF_2, 50)

all_RF_s = gridsearchRF(All_set_RF_2, cv=cv)

# Random Forest: round 3
shap_elim_all_RF_3 = shap_fs(All_set_RF_2, all_RF_s, step=0.1)
All_set_RF_3 = shap_elim_to_reduce(All_set_RF_2, shap_elim_all_RF_3, 16)


# LightGBM: round 1
shap_elim_all_LGBM_1 = shap_fs(All_set, all_lgbm_L, step=0.3)
All_set_LGBM_1 = shap_elim_to_reduce(All_set, shap_elim_all_LGBM_1, 446)

# LightGBM: round 2
shap_elim_all_LGBM_2 = shap_fs(All_set_LGBM_1, all_lgbm_L, step=0.2)
All_set_LGBM_2 = shap_elim_to_reduce(All_set_LGBM_1, shap_elim_all_LGBM_2, 50)

all_lgbm_s = gridsearchLGBM(All_set_LGBM_2, cv=cv)

# LightGBM: round 3
shap_elim_all_LGBM_3 = shap_fs(All_set_LGBM_2, all_lgbm_s, step=0.1)
All_set_LGBM_3 = shap_elim_to_reduce(All_set_LGBM_2, shap_elim_all_LGBM_3, 10)

count_occurences_features(All_set_RF_3.columns)
count_occurences_features(All_set_LGBM_3.columns)


#%% 5.1.4 - Performance comparison: optimal
cv = 5
color_b = '#363636'
color_reg = '#ff7f0e'
color_all = '#2279b5'


# LGBM
fs_Basic_set_LGBM_final = Basic_set_LGBM_2
fs_Regular_set_LGBM_final = All_set_LGBM_3
fs_All_set_LGBM_final = Regular_set_LGBM_3

# gridsearch for LGBM for B, Reg, all feature sets
lgbm_B = gridsearchLGBM(fs_Basic_set_LGBM_final, cv=cv) # 0.0103
lgbm_Reg = gridsearchLGBM(fs_Regular_set_LGBM_final, cv=cv) # 0.0125
lgbm_all = gridsearchLGBM(fs_All_set_LGBM_final, cv=cv) # 0.009

# performance comparison
assess_5x2cv(fs_Basic_set_LGBM_final, fs_Regular_set_LGBM_final, lgbm_B, lgbm_Reg, results_location, 'lgbm_b_Reg_opt', color_b, color_reg)
assess_5x2cv(fs_Basic_set_LGBM_final, fs_All_set_LGBM_final, lgbm_B, lgbm_all, results_location, 'lgbm_b_all_opt', color_b, color_all)
assess_5x2cv(fs_All_set_LGBM_final, fs_Regular_set_LGBM_final, lgbm_all, lgbm_Reg, results_location, 'lgbm_Reg_all_opt', color_all, color_reg)

assess_McNemar(fs_Basic_set_LGBM_final, fs_Regular_set_LGBM_final, lgbm_B, lgbm_Reg)
assess_McNemar(fs_Basic_set_LGBM_final, fs_All_set_LGBM_final, lgbm_B, lgbm_all)
assess_McNemar(fs_All_set_LGBM_final, fs_Regular_set_LGBM_final, lgbm_all, lgbm_Reg)



# Random Forest
fs_Basic_set_RF_final = Basic_set_RF_2
fs_Regular_set_RF_final = All_set_RF_3
fs_All_set_RF_final = Regular_set_RF_3

# gridsearch for RF for B, Reg, all feature sets
RF_B = gridsearchRF(fs_Basic_set_RF_final, cv=cv) # 6.270
RF_Reg = gridsearchRF(fs_Regular_set_RF_final, cv=cv) # 6.095
RF_all = gridsearchRF(fs_All_set_RF_final, cv=cv) # 6.313

# performance comparison
assess_5x2cv(fs_Basic_set_RF_final, fs_Regular_set_RF_final, RF_B, RF_Reg)
assess_5x2cv(fs_Basic_set_RF_final, fs_All_set_RF_final, RF_B, RF_all)
assess_5x2cv(fs_All_set_RF_final, fs_Regular_set_RF_final, RF_all, RF_Reg)

assess_McNemar(fs_Basic_set_RF_final, fs_Regular_set_RF_final, RF_B, RF_Reg)
assess_McNemar(fs_Basic_set_RF_final, fs_All_set_RF_final, RF_B, RF_all)
assess_McNemar(fs_All_set_RF_final, fs_Regular_set_RF_final, RF_all, RF_Reg)

if save:
    write_out_list_dfs(["fs_Basic_set_LGBM_final", "fs_Regular_set_LGBM_final", "fs_All_set_LGBM_final", "fs_Basic_set_RF_final", "fs_Regular_set_RF_final", "fs_All_set_RF_final"], results_location)

#%% 5.1.4 - Performance comparison: last 15
    
cv = 5
last = 15

# LGBM
fs_last_Basic_set_LGBM_final = get_reduced_data(Basic_set, shap_elim_B_LGBM_2.get_reduced_features_set(num_features=last))
fs_last_Regular_set_LGBM_final = get_reduced_data(Regular_set, shap_elim_Reg_LGBM_3.get_reduced_features_set(num_features=last))
fs_last_All_set_LGBM_final = get_reduced_data(All_set, shap_elim_all_LGBM_3.get_reduced_features_set(num_features=last))

# gridsearch for LGBM for B, Reg, all feature sets
lgbm_B = gridsearchLGBM(fs_last_Basic_set_LGBM_final, cv=cv)
lgbm_Reg = gridsearchLGBM(fs_last_Regular_set_LGBM_final, cv=cv)
lgbm_all = gridsearchLGBM(fs_last_All_set_LGBM_final, cv=cv)

# performance comparison
assess_5x2cv(fs_last_Basic_set_LGBM_final, fs_last_Regular_set_LGBM_final, lgbm_B, lgbm_Reg, results_location, 'lgbm_b_Reg_last_15')
assess_5x2cv(fs_last_Basic_set_LGBM_final, fs_last_All_set_LGBM_final, lgbm_B, lgbm_all)
assess_5x2cv(fs_last_All_set_LGBM_final, fs_last_Regular_set_LGBM_final, lgbm_all, lgbm_Reg)

assess_McNemar(fs_last_Basic_set_LGBM_final, fs_last_Regular_set_LGBM_final, lgbm_B, lgbm_Reg)
assess_McNemar(fs_last_Basic_set_LGBM_final, fs_last_All_set_LGBM_final, lgbm_B, lgbm_all)
assess_McNemar(fs_last_All_set_LGBM_final, fs_last_Regular_set_LGBM_final, lgbm_all, lgbm_Reg)



# Random Forest
fs_last_Basic_set_RF_final = get_reduced_data(Basic_set, shap_elim_B_RF_2.get_reduced_features_set(num_features=last))
fs_last_Regular_set_RF_final = get_reduced_data(Regular_set, shap_elim_Reg_RF_3.get_reduced_features_set(num_features=last))
fs_last_All_set_RF_final = get_reduced_data(All_set, shap_elim_all_RF_3.get_reduced_features_set(num_features=last))

# gridsearch for RF for B, Reg, all feature sets
RF_B = gridsearchRF(fs_last_Basic_set_RF_final, cv=cv)
RF_Reg = gridsearchRF(fs_last_Regular_set_RF_final, cv=cv)
RF_all = gridsearchRF(fs_last_All_set_RF_final, cv=cv)

# performance comparison
assess_5x2cv(fs_last_Basic_set_RF_final, fs_last_Regular_set_RF_final, RF_B, RF_Reg)
assess_5x2cv(fs_last_Basic_set_RF_final, fs_last_All_set_RF_final, RF_B, RF_all)
assess_5x2cv(fs_last_All_set_RF_final, fs_last_Regular_set_RF_final, RF_all, RF_Reg)

assess_McNemar(fs_last_Basic_set_RF_final, fs_last_Regular_set_RF_final, RF_B, RF_Reg)
assess_McNemar(fs_last_Basic_set_RF_final, fs_last_All_set_RF_final, RF_B, RF_all)
assess_McNemar(fs_last_All_set_RF_final, fs_last_Regular_set_RF_final, RF_all, RF_Reg)

if save:
    write_out_list_dfs(["fs_last_Basic_set_LGBM_final", "fs_last_Regular_set_LGBM_final", "fs_last_All_set_LGBM_final", "fs_last_Basic_set_RF_final", "fs_last_Regular_set_RF_final", "fs_last_All_set_RF_final"], results_location)


#%% Appendix 1 - Finding the final sets of features names

pd.Series(fs_Basic_set_LGBM_final.columns).sort_values()
pd.Series(fs_Regular_set_LGBM_final.columns).sort_values()
pd.Series(fs_All_set_LGBM_final.columns).sort_values()

pd.Series(fs_Basic_set_RF_final.columns).sort_values()
pd.Series(fs_Regular_set_RF_final.columns).sort_values()
pd.Series(fs_All_set_RF_final.columns).sort_values()

#%% 5.1.5 - Benchmarking

# How long does it take to generate the "Basic set" features?
current = timeit.default_timer()   
Basic_features = create_all_features(data,["Basic"])
print("Basic:", int(timeit.default_timer() - current), "seconds") # 34s

# For Regular and All set generation times, see 5.1.1

# For model training times, see 5.1.4







#%% FIGURES

#%% 4.2.1 - Averaged yearly fourier transactions and balances
    
blue_color = '#2279b5'
 
data_f2_tr = return_with_only_column_types(feature_selection_data, ["f2", "reg", "real", "Y", "tr"], [3, 0, 4, 2, 1]).drop(columns='status').mean(axis=0).reset_index(drop=True).iloc[1:]
data_f2_ba = return_with_only_column_types(feature_selection_data, ["f2", "reg", "real", "Y", "ba"], [3, 0, 4, 2, 1]).drop(columns='status').mean(axis=0).reset_index(drop=True).iloc[1:]

data_f2_tr.index = data_f2_tr.index+1
data_f2_ba.index = data_f2_ba.index+1

fourier_plot = plt.figure()
plt.plot(data_f2_tr, color=blue_color)
plt.savefig(f"{results_location}/figures/yearly_fourier_transactions",dpi=200)
plt.show()

plt.plot(data_f2_ba, color=blue_color)
plt.savefig(f"{results_location}/figures/yearly_fourier_balances",dpi=200)
plt.show()

#%% 5.1.4 - Performance comparison plot: optimal
import numpy as np
import matplotlib.pyplot as plt

N = 4
ind = np.arange(N)  # the x locations for the groups
width = 0.17       # the width of the bars

fig = plt.figure(figsize=(10,6), dpi=60)
ax = fig.add_subplot(111)

plt.ylim(0,100)
# plt.title('AUC comparison for optimal feature sets with 95% confidence interval')

Basic_vals = [69, 71, 91, 89]
Basic_errs = [2, 2, 5.4, 2.6]
rects1 = ax.bar(ind, Basic_vals, width, color='#363636', alpha=0.6, yerr=Basic_errs, capsize=7)
SP_vals = [80, 73, 98, 94]
SP_errs = [2.2, 0.8, 1.8, 2.8]
rects2 = ax.bar(ind+width, SP_vals, width, color='#ff7f0e', alpha=0.8, yerr=SP_errs, capsize=7)
All_vals = [79, 73, 96, 94]
All_errs = [2, 1.2, 2.1, 3.7]
rects3 = ax.bar(ind+width*2, All_vals, width, color='#2279b5', alpha=0.8, yerr=All_errs, capsize=7)

ax.set_ylabel('AUC')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('private data, LightGBM', 'private data, Random Forest', 'public data, LightGBM', 'public data, Random Forest') )
ax.legend( (rects1[0], rects2[0], rects3[0]), ('Basic set', 'Regular set', 'All set') )

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 0.91*h, '%d'%int(h),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.savefig(f"{results_location}/figures/performance_comp_optimal",dpi=200)
plt.show()

#%% 5.1.4 - Performance comparison plot: last 15
import numpy as np
import matplotlib.pyplot as plt

N = 4
ind = np.arange(N)  # the x locations for the groups
width = 0.17       # the width of the bars

fig = plt.figure(figsize=(10,6), dpi=60)
ax = fig.add_subplot(111)

plt.ylim(0,100)
# plt.title('AUC comparison for last-15-features sets with 95% confidence interval')

Basic_vals = [71, 71, 91, 89]
Basic_errs = [0.8, 1.4, 4.2, 3.7]
rects1 = ax.bar(ind, Basic_vals, width, color='#363636', alpha=0.6, yerr=Basic_errs, capsize=7)
SP_vals = [75, 73, 98, 93]
SP_errs = [1.3, 0.7, 0.9, 3.3]
rects2 = ax.bar(ind+width, SP_vals, width, color='#ff7f0e', alpha=0.8, yerr=SP_errs, capsize=7)
All_vals = [75, 74, 97, 94]
All_errs = [1.5, 1.6, 1.4, 3.4]
rects3 = ax.bar(ind+width*2, All_vals, width, color='#2279b5', alpha=0.8, yerr=All_errs, capsize=7)

ax.set_ylabel('AUC')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('private data, LightGBM', 'private data, Random Forest', 'public data, LightGBM', 'public data, Random Forest') )
ax.legend( (rects1[0], rects2[0], rects3[0]), ('Basic set', 'Regular set', 'All set') )

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 0.91*h, '%d'%int(h),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.savefig(f"{results_location}/figures/performance_comp_last_15",dpi=200)
plt.show()

#%% 3.1 - One client transactions
data_one_client = data[data.account_id == 1787]
data_one_client_tr = fill_empty_dates(data_one_client[['date', 'transaction']], 'transaction', data_one_client.date.iloc[0], data_one_client.date.iloc[-1])
plt.plot(data_one_client_tr.transaction, color=blue_color)
plt.xlabel('Time (days)')
plt.ylabel('Transaction (euro)')
plt.gcf().subplots_adjust(left=0.17)
plt.savefig(f"{results_location}/figures/data_transactions",dpi=200)
plt.show()
#%% 3.1 - One client balances
data_one_client_ba = fill_empty_dates(data_one_client[['date', 'balance']], 'balance', data_one_client.date.iloc[0], data_one_client.date.iloc[-1])
plt.plot(data_one_client_ba.balance, color=blue_color)
plt.xlabel('Time (days)')
plt.ylabel('Balance (euro)')
plt.gcf().subplots_adjust(left=0.17)
plt.savefig(f"{results_location}/figures/Basic_setalances",dpi=200)
plt.show()

