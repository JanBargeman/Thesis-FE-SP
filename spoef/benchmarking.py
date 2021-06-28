from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import os

os.chdir("/Users/Jan/Desktop/Thesis/Thesis-FE-SP")

from spoef.utils import combine_features_dfs, count_na
from spoef.feature_generation import feature_creation_yearly

def gridsearchLGBM(data, cv=3):
    y = data.iloc[:, 0]
    X = data.iloc[:, 1:]
    
    parameters = {
        'n_estimators':[20, 50],
        'max_depth':[3,6],
        'num_leaves':[20],
        'learning_rate':[0.1, 0.3],
        'max_bin':[63],
        'min_child_samples':[20],
        'scale_pos_weight':[1.0, 3.0],
        }
    lgbm = LGBMClassifier()
    clf = GridSearchCV(lgbm, parameters, scoring='roc_auc', n_jobs=1, cv=cv)
    clf.fit(X,y)
    
    return clf.best_estimator_


def gridsearchRF(data, cv=3):
    y = data.iloc[:, 0]
    X = data.iloc[:, 1:]
    
    parameters = {
        'n_estimators':[50, 100, 500],
        'max_depth':[3, 6],
        }
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, parameters, scoring='roc_auc', n_jobs=1, cv=cv)
    clf.fit(X,y)
    
    return clf.best_estimator_

def grid_search_LGBM(data, test_size=0.4, debug=False):
    """
    PERSONAL FUNCTION, not part of open source:
    This function performs a grid search for the LightGBM model.

    Args:
        data (pd.DataFrame()) : data of features to train lgbm model on
        test_size (0 < float < 1) : test size for AUC determination

    Returns:
        base_LGBM (lightgbm.LGBMClassifier) : the model with best performing params

    """
    Y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, Y, test_size=test_size, random_state=0
    )

    list_n_est = [5000, 10000] #10 (100, 300, 1000)
    list_max_depth = [6] #2, 4
    list_num_leaves = [20] #3, 6 
    list_learn_rate = [0.1, 0.3] #0.05
    list_max_bin = [63]
    list_min_child_samples = [20] #40
    list_scale_pos_weight = [1.0] # 14: 318,0,21,2, 140: 318,0,23,0 #340 voor 1-3 #800.0
    
    # list_reg_alpha = [0.0, 10]
    # list_reg_lambda = [0.0, 10]

    length_search = (
        len(list_n_est)
        * len(list_max_depth)
        * len(list_num_leaves)
        * len(list_learn_rate)
        * len(list_max_bin)
        * len(list_min_child_samples)
        * len(list_scale_pos_weight)
    )
    i = 0

    max_auc = 0
    AUC_score_avg_list = []

    for n_est in list_n_est:
        for max_depth in list_max_depth:
            for num_leaves in list_num_leaves:
                for learn_rate in list_learn_rate:
                    for max_bin in list_max_bin:
                        for min_child_samples in list_min_child_samples:
                            for scale_pos_weight in list_scale_pos_weight:
                                i = i + 1
                                if debug:
                                    print("\n\n")
                                print(
                                    "\r",
                                    "\rLoading, please wait: "
                                    + "%.1f" % (100 * i / length_search)
                                    + "%",
                                    end="",
                                )                                
                                lgbm = LGBMClassifier(
                                    objective="binary",
                                    n_estimators=n_est,
                                    max_depth=max_depth,
                                    num_leaves=num_leaves,
                                    learning_rate=learn_rate,
                                    max_bin=max_bin,
                                    min_child_samples=min_child_samples,
                                    random_state=0,
                                    scale_pos_weight=scale_pos_weight,
                                    n_jobs=8,
                                )                    
                                
                                auc_list=[]
                                skf = StratifiedKFold(n_splits=2)
                                
                                for train_index, test_index in skf.split(X,Y):
                                    X_train, X_valid = X[train_index], X[test_index]
                                    y_train, y_valid = Y[train_index], Y[test_index]              
                                    
            
                                    lgbm.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric="AUC", verbose=False, early_stopping_rounds=50)
                                    
                                    current_params = [n_est, max_depth, num_leaves, learn_rate, scale_pos_weight]
                                    AUC_score = roc_auc_score(
                                        y_valid, lgbm.predict_proba(X_valid)[:,1]
                                    )
                                    AUC_train = roc_auc_score(
                                        y_train, lgbm.predict_proba(X_train)[:,1]
                                    )
                                    current_conf_matrix = confusion_matrix(
                                        y_valid, lgbm.predict(X_valid)
                                    )
                                    # print(current_params, ":")
                                    # print("\n", current_conf_matr)

                                    if debug:
                                        print("\nTrain AUC:", AUC_train)
                                        print("Test AUC:", AUC_score)
                                    # print("\n\n")

                                    auc_list.append(AUC_score)
                                AUC_score_avg = sum(auc_list)/len(auc_list)
                                AUC_score_avg_list.append(AUC_score_avg)
                                if AUC_score_avg > max_auc:
                                    max_auc = AUC_score
                                    base_params = current_params
                                    base_conf_matrix = current_conf_matrix
                                    base_LGBM = lgbm

    print("\n\nMax AUC: " + str(max_auc) + " at " + str(base_params) + "\n")
    print(str(base_conf_matrix))
    plt.hist(AUC_score_avg_list)
    plt.show()
    print("\nMean AUC: " + str(sum(AUC_score_avg_list)/len(AUC_score_avg_list)))
        
    return base_LGBM, AUC_score_avg_list


def grid_search_RF(data, test_size=0.4, debug=False):
    """
    PERSONAL FUNCTION, not part of open source:
    This function performs a grid search for the RandomForest model.

    Args:
        data (pd.DataFrame()) : data of features to train rf model on
        test_size (0 < float < 1) : test size for AUC determination

    Returns:
        base_RF (RandomForestClassifier) : the model with best performing params

    """
    Y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, Y, test_size=test_size, random_state=0
    )

    list_n_trees = [50, 100, 300]
    list_max_depth = [10, 20, 50]

    length_search = len(list_n_trees) * len(list_max_depth)
    i = 0

    max_auc = 0
    auc_list = []

    for n_trees in list_n_trees:
        for max_depth in list_max_depth:
            i = i + 1
            rf = RandomForestClassifier(
                n_estimators=n_trees, max_depth=max_depth, random_state=0, n_jobs=8,
            )
            rf.fit(X_train, y_train)
            current_params = [n_trees, max_depth]
            AUC_score = roc_auc_score(
                y_valid, rf.predict_proba(X_valid)[:,1]
            )
            current_conf_matrix = confusion_matrix(y_valid, rf.predict(X_valid))
            # print(current_params, ":")
            # print("\n", current_conf_matr)
            if debug:
                print("AUC:", AUC_score)
            # print("\n\n"
            print(
                "\r",
                "\rLoading, please wait: " + "%.1f" % (100 * i / length_search) + "%",
                end="",
            )
            auc_list.append(AUC_score)
            if AUC_score > max_auc:
                max_auc = AUC_score
                base_params = current_params
                base_conf_matrix = current_conf_matrix
                base_RF = rf
    print("\n\nMax AUC: " + str(max_auc) + " at " + str(base_params) + "\n")
    print(str(base_conf_matrix))
    plt.hist(auc_list)
    plt.show()
    return base_RF, auc_list


def create_yearly_features_mother(data, list_featuretypes, mother_wavelet):
    """
    PERSONAL FUNCTION, not part of open source:
    This function is more of a quick feature engineering part to test the mother
    wavelet shape. As the transactions and balances vary more on yearly basis
    this seems like the datasets to test the mother wavelets on. Here the shape
    of the mother wavelet will be more clearly compared to the shape of the data.

    Args:
        data (pd.DataFrame()) : data for which to make features.
        list_featuretypes (list) : list of feature types to be computed.
        mother_wavelet (str) : type of mother wavelet

    Returns:
        features (pd.DataFrame()) : df with row of features for each identifier.

    """

    transaction_features_yearly = feature_creation_yearly(
        data[["account_id", "date", "transaction"]],
        "account_id",
        "transaction",
        list_featuretypes,
        mother_wavelet=mother_wavelet,
    )
    balance_features_yearly = feature_creation_yearly(
        data[["account_id", "date", "balance"]],
        "account_id",
        "balance",
        list_featuretypes,
        mother_wavelet=mother_wavelet,
    )

    list_features_dfs = [
        transaction_features_yearly,
        balance_features_yearly,
    ]
    count_na(list_features_dfs)

    all_features = combine_features_dfs(list_features_dfs)

    return all_features


def search_mother_wavelet(data, status, base_lgbm_all, list_mother_wavelets, test=False):
    """
    PERSONAL FUNCTION, not part of open source:
    This function uses different mother wavelets for creating features. It then
    inputs those features in the LGBM model that got the best performance on
    the data with all features. Those features were created with the "db2"
    mother wavelet.

    Args:
        data (pd.DataFrame()) : data for which to make features.
        status : the list of identifiers and their label
        base_LGBM (lightgbm.LGBMClassifier) : the model with best performing params
        list_mother_wavelets : list of mother wavelets to try

    Returns:
        auc_list : list of AUC's for each mother wavelet

    """
    list_featuretypes = ["W_B", "W"]
    max_auc = 0
    
    data = data.copy()
    for mother_wavelet in list_mother_wavelets:

        mother_features = create_yearly_features_mother(
            data, list_featuretypes, mother_wavelet
        )

        data_all = combine_features_dfs([status, mother_features])
        
        if test:    
            data_all.iloc[[1,5,12,7,37],0] = 1


        Y = data_all.iloc[:, 0].values
        X = data_all.iloc[:, 1:].values
        
        skf = StratifiedKFold(n_splits=2)
        
        auc_list = []
        avg_auc_list = []

        for train_index, test_index in skf.split(X,Y):
            
            X_train, X_valid = X[train_index], X[test_index]
            y_train, y_valid = Y[train_index], Y[test_index]              
        
            base_lgbm_all.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric="AUC", verbose=False, early_stopping_rounds=5)

            AUC_score = roc_auc_score(
                y_valid, base_lgbm_all.predict_proba(X_valid)[:,1]
            )
            auc_list.append(AUC_score)
        
        avg_auc = sum(auc_list)/len(auc_list)
        avg_auc_list.append(avg_auc)
        
        if avg_auc > max_auc:
            max_auc = avg_auc
            best_mother_wavelet = mother_wavelet

    print("\n\nMax AUC: " + str(max_auc) + " with " + str(best_mother_wavelet) + " wavelet.\n")
    print(str(list_mother_wavelets))
    print(str(avg_auc_list))
    plt.hist(avg_auc_list)
    plt.show()
    return avg_auc_list






