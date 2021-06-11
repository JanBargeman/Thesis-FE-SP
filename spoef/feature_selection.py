from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import statistics
import math
import scipy.stats
from sklearn.model_selection import train_test_split
from spoef.utils import combine_features_dfs, select_features_subset, write_out_list_dfs
from spoef.benchmarking import grid_search_LGBM, grid_search_RF, search_mother_wavelet
import joblib
import pandas as pd
import os
os.chdir("/Users/Jan/Desktop/Thesis/Thesis-FE-SP")




# #%% For testing functions

# data = pd.read_csv("personal/data/data.csv")
# data.date = pd.to_datetime(data.date, format="%Y-%m-%d")
# status = data[["account_id", "status"]].drop_duplicates().set_index("account_id")


# #%% make test data
# data = data.iloc[0:2000,:]
# # data_acc = data[data.account_id == 1787]
# # data_used = data_acc[["date","balance"]]
# # data = data[data.account_id == 276]
# # data = data[data.account_id == 1843]


#%%
def assess_5x2cv(dataset1, dataset2, model1, model2):
    
    mean1, stdev1 = perform_5x2cv(dataset1, model1)
    mean2, stdev2 = perform_5x2cv(dataset2, model2)
    
    print(f"\nFirst: {mean1} ({stdev1})")
    print(f"Second: {mean2} ({stdev2})")
    
    try:
        Z = (mean1 - mean2) / math.sqrt(stdev1*stdev1/math.sqrt(5) + stdev2*stdev2/math.sqrt(5))
    except:
        print("Performance equal, div by zero error.")
        return
    
    print("Z-value: " + str(Z))
    
    if Z > -2 and Z < 2:
        print("The performance is equal.")
    elif Z >= 2:
        print("The first set performs significantly better.")
    elif Z <= -2:
        print("The second set performs significantly better.")
    
    print("p-value: " + str(scipy.stats.norm.sf(abs(Z))*2))
    
    return 


def assess_McNemar(dataset1, dataset2, model1, model2, test_size=0.4):
    
    Y1 = dataset1.iloc[:, 0].values
    X1 = dataset1.iloc[:, 1:].values

    X_train1, X_valid1, y_train1, y_valid1 = train_test_split(
        X1, Y1, test_size=test_size, random_state=0
    )

    Y2 = dataset2.iloc[:, 0].values
    X2 = dataset2.iloc[:, 1:].values

    X_train2, X_valid2, y_train2, y_valid2 = train_test_split(
        X2, Y2, test_size=test_size, random_state=0
    )
    
    model1.fit(X_train1, y_train1)
    pred1 = model1.predict(X_valid1)
    
    model2.fit(X_train2, y_train2)
    pred2 = model2.predict(X_valid2)
    
    a,b,c,d = calc_contingency_table(y_valid1, pred1, pred2)
    
    try: 
        McNemar = (b-c)*(b-c) / (b+c)
    except:
        print("Division by zero, the data sets perform equally well")
        return
    
    if (b+c) < 25:
        if b > c:
            print("The first set performs better.")
            print("p-value: " + str("%.4f" %scipy.stats.binom.cdf(b, b+c, 0.5)))
        elif b < c:
            print("The second set performs better.")
            print("p-value: " + str("%.4f" %scipy.stats.binom.cdf(c, b+c, 0.5)))
        else:
            raise ValueError("b==c?")
            
    else:    
        if McNemar < 3.841:
            print("The data sets perform equally well.")
        else:
            if b > c:
                print("The first set performs significantly better.")
                print("p-value: " + str("%.4f" %(1-scipy.stats.chi2.cdf(McNemar,1))))    
            elif b < c:
                print("The second set performs significantly better.")
                print("p-value: " + str("%.4f" %(1-scipy.stats.chi2.cdf(McNemar,1))))    
    return         
        
def calc_contingency_table(valid, pred1, pred2):
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range(len(pred1)):
        if valid[i] == 1 and pred1[i] == 1 and pred2[i] == 1:
            a = a + 1
        elif valid[i] == 0 and pred1[i] == 0 and pred2[i] == 0:
            a = a + 1
        elif valid[i] == 1 and pred1[i] == 1 and pred2[i] == 0:
            b = b + 1
        elif valid[i] == 0 and pred1[i] == 0 and pred2[i] == 1:
            b = b + 1
        elif valid[i] == 1 and pred1[i] == 0 and pred2[i] == 1:
            c = c + 1
        elif valid[i] == 0 and pred1[i] == 1 and pred2[i] == 0:
            c = c + 1
        elif valid[i] == 1 and pred1[i] == 0 and pred2[i] == 0:
            d = d + 1
        elif valid[i] == 0 and pred1[i] == 1 and pred2[i] == 1:
            d = d + 1
        else:
            raise ValueError("Contingency table error.")

    return a,b,c,d   

def feat_importance_RF(dataset, model):
    return

def feat_importance_LGBM(dataset, model):
    return


def perform_5x2cv(data, model):

    X = data.iloc[:, 1:].values
    Y = data.iloc[:, 0].values
    
    AUC_score_avg_list = []
    
    for i in range(5):
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=i)
        auc_list = []
        for train_index, test_index in skf.split(X,Y):
            X_train, X_valid = X[train_index], X[test_index]
            y_train, y_valid = Y[train_index], Y[test_index]
            model.fit(X_train, y_train)
            
            auc_list.append(roc_auc_score(
                y_valid, model.predict_proba(X_valid)[:,1]#, multi_class="ovo"
            ))
            # print(auc_list)
            
        AUC_score_avg_list.append(sum(auc_list)/len(auc_list))
        # print("avg", AUC_score_avg_list)
        
    mean = float("%.4f" %statistics.mean(AUC_score_avg_list))
    stdev = float("%.6f" %statistics.stdev(AUC_score_avg_list))
    
    # print("Mean (std): " + str(mean) + " (" + str(stdev) + ")")
    
    return mean, stdev

def select_time_window_subset(data, time_window_list):
    return

def select_transforms_subset(data, transforms_list):
    return


def return_without_column_types(data, string_list, index_list):
    data = data.copy()
    for string, index in zip(string_list, index_list):
        data = data[["status"]+[col for col in data.columns[1:] if not col.split(" ")[index].startswith(string)]]
    return data


def check_transforms(data, debug=False):
    data = data.copy()
    
    best_lgbm, auc_list, explainer = grid_search_LGBM(data, debug=debug)
    
    data_wo_reg = return_without_column_types(data, ["reg"], [0])
    data_wo_norm = return_without_column_types(data, ["norm"], [0])
    data_wo_PCA = return_without_column_types(data, ["PCA"], [0])
    data_wo_ICA = return_without_column_types(data, ["ICA"], [0])
    
    assess_5x2cv(data, data_wo_reg, best_lgbm, best_lgbm)
    assess_5x2cv(data, data_wo_norm, best_lgbm, best_lgbm)
    assess_5x2cv(data, data_wo_PCA, best_lgbm, best_lgbm)
    assess_5x2cv(data, data_wo_ICA, best_lgbm, best_lgbm)
    
    return explainer, best_lgbm

def check_balances_transactions(data):
    data = data.copy()
    
    best_lgbm, auc_list, explainer = grid_search_LGBM(data)

    data_wo_balances = return_without_column_types(data, ["ba"], [1])
    data_wo_transactions = return_without_column_types(data, ["tr"], [1])
    
    assess_5x2cv(data, data_wo_balances, best_lgbm, best_lgbm)
    assess_5x2cv(data, data_wo_transactions, best_lgbm, best_lgbm)
    
    return explainer, best_lgbm

def check_timewindows(data):
    data = data.copy()
    
    best_lgbm, auc_list, explainer = grid_search_LGBM(data)
    
    data_wo_months = return_without_column_types(data, ["M"], [2]) 
    data_wo_years = return_without_column_types(data, ["Y"], [2])
    data_wo_overall = return_without_column_types(data, ["O"], [2])
    
    assess_5x2cv(data, data_wo_months, best_lgbm, best_lgbm)
    assess_5x2cv(data, data_wo_years, best_lgbm, best_lgbm)
    assess_5x2cv(data, data_wo_overall, best_lgbm, best_lgbm)
    
    return explainer, best_lgbm



def check_feature_types(data, debug=False):
    data = data.copy()
    
    best_lgbm, auc_list, explainer = grid_search_LGBM(data, debug=debug)
        
    data_wo_Basic = return_without_column_types(data, ["B"], [3]) # 0.9657 -> 0.8759
    data_wo_Fourier = return_without_column_types(data, ["fft"], [3])
    data_wo_Wavelet = return_without_column_types(data, ["wavelet"], [3])
    data_wo_Wav_Basic = return_without_column_types(data, ["wav_B"], [3])
    
    assess_5x2cv(data, data_wo_Basic, best_lgbm, best_lgbm)
    assess_5x2cv(data, data_wo_Fourier, best_lgbm, best_lgbm)
    assess_5x2cv(data, data_wo_Wavelet, best_lgbm, best_lgbm)    
    assess_5x2cv(data, data_wo_Wav_Basic, best_lgbm, best_lgbm)
    
    return explainer, best_lgbm

def check_signal_processing_properties(data):
    data = data.copy()
    
    best_lgbm, auc_list = grid_search_LGBM(data)

    
    return

def check_months_lengths(data):
    data = data.copy()
    data_compare = data.copy()
    
    best_lgbm, auc_list = grid_search_LGBM(data)
    
    
    for i in [12,11,10,9,8,7,6,5,4,3,2,1]:
        data = return_without_column_types(data, "M_"+[str(i)+"/12"], [2])
        assess_5x2cv(data_compare, data, best_lgbm, best_lgbm)
    
    return
    

#%%



if __name__ == "__main__":
    
    regular_features = pd.read_csv("personal/results/public_czech/reg_features_db2.csv", index_col="account_id")
    # regular_features = pd.read_csv("personal/temp/regular_features.csv", index_col="account_id")
    
    data = combine_features_dfs([status, regular_features])
    
    
    
    check_transforms(data)
    
    check_timewindows(data)
    check_months_lengths(data)
    
    
    
    # regular_features = pd.read_csv("personal/all_features_db2.csv", index_col="account_id")
    regular_features = pd.read_csv("personal/temp/regular_features.csv", index_col="account_id")

    data_all = combine_features_dfs([status, regular_features])
    data_wo_fft = return_without_column_types(regular_features, ["fft"], [3])
    data_wo_B = return_without_column_types(regular_features, ["B"], [3])
    
    B_features = select_features_subset(regular_features, ["B"])
    data_B = combine_features_dfs([status, B_features])
    best_lgbm_all = joblib.load("personal/temp/lgbm.joblib")
    
    best_lgbm_B = joblib.load("personal/temp/lgbm_b.joblib")
    
    
    
    
    # assess_5x2cv(data_all, data_B, best_lgbm_all, best_lgbm_B)
    
    assess_McNemar(data_all, data_B, best_lgbm_all, best_lgbm_B)
