from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import os
import numpy as np
import timeit


os.chdir("/Users/Jan/Desktop/Thesis/Thesis-FE-SP")

from spoef.utils import combine_features_dfs
from spoef.feature_generation import feature_generation
from spoef.feature_selection import perform_5x2cv

def gridsearchLGBM(data, cv=5, size='s', debug=False):
    Y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values

    if size == 's':
        list_n_est = [100, 300]
        list_max_depth = [3,6]
    if size == 'L':
        list_n_est = [1000, 3000]
        list_max_depth = [6,9]

    list_num_leaves = [20] #3, 6 
    list_learn_rate = [0.1] #0.05
    list_min_child_samples = [20] #40
    list_scale_pos_weight = [1.0, 2.0]
    
    list_reg_alpha = [0, 20]
    list_reg_lambda = [0, 50]

    length_search = (
        len(list_n_est)
        * len(list_max_depth)
        * len(list_num_leaves)
        * len(list_learn_rate)
        * len(list_min_child_samples)
        * len(list_scale_pos_weight)
        * len(list_reg_alpha)
        * len(list_reg_lambda)
    )
    i = 0

    max_auc = 0
    mean_train_scores = []
    mean_test_scores = []

    for n_est in list_n_est:
        for max_depth in list_max_depth:
            for num_leaves in list_num_leaves:
                for learn_rate in list_learn_rate:
                        for min_child_samples in list_min_child_samples:
                            for scale_pos_weight in list_scale_pos_weight:
                                for reg_alpha in list_reg_alpha:
                                    for reg_lambda in list_reg_lambda:
                                        i = i + 1

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
                                            min_child_samples=min_child_samples,
                                            reg_alpha=reg_alpha,
                                            reg_lambda=reg_lambda,
                                            random_state=0,
                                            scale_pos_weight=scale_pos_weight,
                                            n_jobs=1,
                                        )                    
                                        
                                        train_scores = []
                                        test_scores = []
                                        skf = StratifiedKFold(n_splits=cv)
                                        
                                        for train_index, test_index in skf.split(X,Y):
                                            X_train, X_valid = X[train_index], X[test_index]
                                            y_train, y_valid = Y[train_index], Y[test_index]              
                                            
                                            current = timeit.default_timer()   
                                            lgbm.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric="AUC", verbose=False, early_stopping_rounds=30)
                                            train_time = timeit.default_timer() - current
                                        
                                            current_params = [n_est, max_depth, num_leaves, learn_rate, scale_pos_weight, reg_alpha, reg_lambda]
                                            test_score = roc_auc_score(
                                                y_valid, lgbm.predict_proba(X_valid)[:,1]
                                            )
                                            train_score = roc_auc_score(
                                                y_train, lgbm.predict_proba(X_train)[:,1]
                                            )
                                            # print(train_score == lgbm.best_score_.get('training').get('auc'))
                                            current_conf_matrix = confusion_matrix(
                                                y_valid, lgbm.predict(X_valid)
                                            )
        
                                            train_scores.append(train_score)
                                            test_scores.append(test_score)
                                        
                                        mean_train_scores.append(sum(train_scores)/len(train_scores))
                                        mean_test_scores.append(sum(test_scores)/len(test_scores))
                                        
                                        if sum(test_scores)/len(test_scores) > max_auc:
                                            max_auc = sum(test_scores)/len(test_scores)
                                            base_params = current_params
                                            base_conf_matrix = current_conf_matrix
                                            base_LGBM = lgbm
                                            base_train_time = train_time

    print('\nMean training score:', np.mean(mean_train_scores))
    print('Mean validation score:', np.mean(mean_test_scores))
    print('Best parameters:', base_params)
    print('Best train time:', base_train_time)
    
    if debug:
        print("\n\nMax AUC: " + str(max_auc) + " at " + str(base_params) + "\n")
        print(str(base_conf_matrix))
        plt.hist(mean_train_scores, bins=50)
        plt.show()
        plt.hist(mean_test_scores, bins=50)
        plt.show()
        plt.hist([a_i - b_i for a_i, b_i in zip(mean_train_scores, mean_test_scores)], bins=50)
        plt.show()
        print("\nMean AUC: " + str(sum(mean_test_scores)/len(mean_test_scores)))
        
    return base_LGBM




def gridsearchRF(data, cv=3, debug=False):
    y = data.iloc[:, 0]
    X = data.iloc[:, 1:]
    
    parameters = {
        'n_estimators':[20, 30, 50, 100],
        'max_depth':[3, 6, 9],
        }
    rf = RandomForestClassifier(n_jobs=1)
    clf = GridSearchCV(rf, parameters, scoring='roc_auc', n_jobs=1, cv=cv, return_train_score=True)
    
    current = timeit.default_timer()   
    clf.fit(X,y)
    train_time = timeit.default_timer() - current
  
    print('Mean training score:', np.mean(clf.cv_results_.get('mean_train_score')))
    print('Mean validation score:', np.mean(clf.cv_results_.get('mean_test_score')))
    print('Best parameters:', clf.best_params_)
    print('Best train time:', train_time)
    
    if debug:        
        plt.hist(clf.cv_results_.get('mean_train_score'), bins=50)
        plt.show()
        plt.hist(clf.cv_results_.get('mean_test_score'), bins=50)
        plt.show()
        plt.hist(clf.cv_results_.get('mean_train_score')-clf.cv_results_.get('mean_test_score'), bins=50)
        plt.show()
        for i in range(len(clf.cv_results_.get('mean_train_score'))):
            print(clf.cv_results_.get('params')[i], f'{clf.cv_results_.get("mean_train_score")[i]:.4f}', f'{clf.cv_results_.get("mean_test_score")[i]:.4f}')
    
    return clf.best_estimator_


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

    transaction_features_yearly = feature_generation(
        data[["account_id", "date", "transaction"]],
        "account_id",
        "transaction",
        list_featuretypes,
        time_window='year',
        mother_wavelet=mother_wavelet,
    )
    balance_features_yearly = feature_generation(
        data[["account_id", "date", "balance"]],
        "account_id",
        "balance",
        list_featuretypes,
        time_window='year',
        mother_wavelet=mother_wavelet,
    )

    list_features_dfs = [
        transaction_features_yearly,
        balance_features_yearly,
    ]

    all_features = combine_features_dfs(list_features_dfs)

    return all_features


def search_mother_wavelet(data, status, list_mother_wavelets, test=False):
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
    mean_list = []
    std_list = []
    max_auc = 0
    
    data = data.copy()
    
    for mother_wavelet in list_mother_wavelets:

        mother_features = create_yearly_features_mother(
            data, list_featuretypes, mother_wavelet
        )

        data_all = combine_features_dfs([status, mother_features])
        
        base_lgbm_all = gridsearchLGBM(data_all)
        
        mean, std = perform_5x2cv(data_all, base_lgbm_all)[0:2]
        
        mean_list.append(mean)
        std_list.append(std)
      
        if mean > max_auc:
            max_auc = mean
            best_mother_wavelet = mother_wavelet

    print("\n\nMax AUC: " + str(max_auc) + " with " + str(best_mother_wavelet) + " wavelet.\n")
    print(str(list_mother_wavelets))
    
    wavelet_perf_list = [f'{i} ({j})' for i,j in zip(mean_list,std_list)]
    print(wavelet_perf_list)

    plt.hist(mean_list)
    plt.show()
    return wavelet_perf_list


