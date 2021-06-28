from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import statistics
import math
import scipy.stats
from sklearn.model_selection import train_test_split
import pandas as pd
from dateutil.relativedelta import relativedelta

from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from functools import reduce
import scipy.signal  # .stft #.argrelmax  for finding max in plots or smth
import scipy.fft
import numpy as np
import pywt
import timeit

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from probatus.feature_elimination import ShapRFECV
from sklearn.linear_model import LogisticRegression

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


def shap_fs(data, classifier_type, step=0.2, cv=5, scoring='roc_auc', n_iter=5):
    
    if classifier_type == 'LGBM':
        classifier = LGBMClassifier(
                        objective="binary",
                        n_estimators=30, # 5000
                        max_depth=3, # 6
                        num_leaves=20,
                        learning_rate=0.1,
                        random_state=0,
                        scale_pos_weight=1.0,
                        # is_unbalance=True,
                    )
        
    elif classifier_type == 'LGBM_cv':
        parameters = {
            'n_estimators':[5000, 10000],
            'max_depth':[6],
            'num_leaves':[20],
            'learning_rate':[0.1, 0.3],
            'max_bin':[63],
            'min_child_samples':[20],
            'scale_pos_weight':[1.0, 3.0],
            }
        clf = LGBMClassifier(objective='binary')
        classifier = RandomizedSearchCV(clf, parameters, scoring='roc_auc', n_jobs=1, cv=cv, n_iter=n_iter)
        
    elif classifier_type == 'RF':
        classifier = RandomForestClassifier()
        
    elif classifier_type == 'RF_cv':
        parameters = {
            'n_estimators':[5000, 10000],
            'max_depth':[6],
            }
        clf = RandomForestClassifier()
        classifier = RandomizedSearchCV(clf, parameters, scoring='roc_auc', n_jobs=1, cv=cv, n_iter=n_iter)
        
    elif classifier_type == 'LR':
        classifier = LogisticRegression()
        
    else:
        raise ValueError('classifier_type should be one of "LGBM", "RF" or "LR"')
    
    shap_elim = ShapRFECV(classifier, step=step, cv=cv, scoring=scoring, n_jobs=1)
    
    y = data.iloc[:, 0]
    X = data.iloc[:, 1:]
    
    report = shap_elim.fit_compute(X,y, check_additivity=False)
    
    performance_plot = shap_elim.plot()

        
    return shap_elim

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


def create_global_transformer_PCA():
    global transformer 
    transformer = PCA(n_components=1)
def create_global_transformer_ICA():
    global transformer 
    transformer = FastICA(n_components=1)


def create_all_features_transformed(data, transform_type, list_featuretypes, mother_wavelet="db2"):

    if transform_type == 'PCA':
        create_global_transformer_PCA()    
    elif transform_type == 'ICA':
        create_global_transformer_ICA()    


    # current = timeit.default_timer()
    transaction_features_quarterly = feature_creation_quarterly_transformed(
        data[["account_id", "date", "transaction", "balance"]],
        "account_id",
        "transaction",
        list_featuretypes,
        mother_wavelet=mother_wavelet,
    )
    # print("monthly:", int(timeit.default_timer() - current), "seconds")
    # current = timeit.default_timer()  # 527

    transaction_features_yearly = feature_creation_yearly_transformed(
        data[["account_id", "date", "transaction", "balance"]],
        "account_id",
        "transaction",
        list_featuretypes,
        mother_wavelet=mother_wavelet,
    )



    # print("overall:", int(timeit.default_timer() - current), "seconds")  # 533

    list_features_dfs = [
        transaction_features_quarterly,
        transaction_features_yearly,
    ]
    count_na(list_features_dfs)

    all_features = combine_features_dfs(list_features_dfs)
    
    if transform_type == 'PCA':
        all_features.columns = ["PCA " + col for col in all_features.columns]
    elif transform_type == 'ICA':
        all_features.columns = ["ICA " + col for col in all_features.columns]
    
    return all_features



def feature_creation_yearly_transformed(
    data,
    grouper,
    combine_fill_method,
    list_featuretypes=["B"],
    observation_length=1,
    fourier_n_largest_frequencies=30,
    wavelet_depth=5,
    mother_wavelet="db2",
):
    features = (
        data.groupby(grouper)
        .apply(
            compute_features_yearly_transformed,
            combine_fill_method=combine_fill_method,
            list_featuretypes=list_featuretypes,
            observation_length=observation_length,
            fourier_n_largest_frequencies=fourier_n_largest_frequencies,
            wavelet_depth=wavelet_depth,
            mother_wavelet=mother_wavelet,
        )
        .reset_index(level=1, drop=True)
    )
    return features




def compute_features_yearly_transformed(
    data,
    combine_fill_method,
    list_featuretypes,
    observation_length,
    fourier_n_largest_frequencies,
    wavelet_depth,
    mother_wavelet,
):

    # drop identifier column
    data = data.drop(data.columns[0], axis=1)

    # select only relevant period and fill the empty date
    trans_data = prepare_data_yearly(data[["date", "transaction"]], "transaction", observation_length)
    bal_data = prepare_data_yearly(data[["date", "balance"]], "balance", observation_length)
    prepared_data = trans_data.merge(bal_data, on="date")

    start_date = prepared_data.iloc[0, 0]

    # create features per year
    features = pd.DataFrame()
    for year in range(0, observation_length):
        data_year = prepared_data[
            (prepared_data.iloc[:, 0] >= start_date + relativedelta(years=year))
            & (prepared_data.iloc[:, 0] < start_date + relativedelta(years=year + 1))
        ]
        data_transformed = pd.DataFrame(transformer.fit_transform(data_year.iloc[:,[1,2]])).iloc[:,0]
        yearly_features = compute_list_featuretypes(
            data_transformed,
            list_featuretypes,
            fourier_n_largest_frequencies,
            wavelet_depth,
            mother_wavelet,
        )
        # name columns
        yearly_features.columns = [
            "xf Y_"
            + str(year + 1)
            + "/"
            + str(observation_length)
            + " "
            + col
            for col in yearly_features.columns
        ]
        features = pd.concat([features, yearly_features], axis=1)
    return features


def feature_creation_quarterly_transformed(
    data,
    grouper,
    combine_fill_method,
    list_featuretypes=["B"],
    observation_length=4,
    fourier_n_largest_frequencies=10,
    wavelet_depth=4,
    mother_wavelet="db2",
):

    features = (
        data.groupby(grouper)
        .apply(
            compute_features_quarterly_transformed,
            combine_fill_method=combine_fill_method,
            list_featuretypes=list_featuretypes,
            observation_length=observation_length,
            fourier_n_largest_frequencies=fourier_n_largest_frequencies,
            wavelet_depth=wavelet_depth,
            mother_wavelet=mother_wavelet,
        )
        .reset_index(level=1, drop=True)
    )
    return features


def compute_features_quarterly_transformed(
    data,
    combine_fill_method,
    list_featuretypes,
    observation_length,
    fourier_n_largest_frequencies,
    wavelet_depth,
    mother_wavelet,
):
    # drop identifier column
    data = data.drop(data.columns[0], axis=1)

    # select only relevant period and fill the empty date
    trans_data = prepare_data_quarterly(data[["date", "transaction"]], "transaction", observation_length)
    bal_data = prepare_data_quarterly(data[["date", "balance"]], "balance", observation_length)
    prepared_data = trans_data.merge(bal_data, on="date")
    
    start_date = trans_data.iloc[0, 0]

    # create features per month
    features = pd.DataFrame()
    for quarter in range(0, observation_length):
        data_quarter = prepared_data[
            (prepared_data.iloc[:, 0] >= start_date + relativedelta(months=3*quarter))
            & (prepared_data.iloc[:, 0] < start_date + relativedelta(months=3*quarter + 3))
        ]
        data_transformed = pd.DataFrame(transformer.fit_transform(data_quarter.iloc[:,[1,2]])).iloc[:,0]
        quarterly_features = compute_list_featuretypes(
            data_transformed,
            list_featuretypes,
            fourier_n_largest_frequencies,
            wavelet_depth,
            mother_wavelet,
        )
        # name columns
        quarterly_features.columns = [
            "xf Q_"
            + str(quarter + 1)
            + "/"
            + str(observation_length)
            + " "
            + col
            for col in quarterly_features.columns
        ]
        features = pd.concat([features, quarterly_features], axis=1)
    return features
# #%% PCA

# PCA_features = create_all_features_transformed(data, 'PCA', ["B", "F", "W", "W_B"], "db2")

# #%% Writeout PCA_features
# PCA_features.to_csv("personal/PCA_features.csv")
# #%% Read in PCA_features
# PCA_features = pd.read_csv("personal/PCA_features.csv", index_col="account_id")




# #%% ICA

# ICA_features = create_all_features_transformed(data, 'ICA', ["B", "F", "W", "W_B"], "db2")

# #%% Writeout ICA_features
# ICA_features.to_csv("personal/ICA_features.csv")
# #%% Read in ICA_features
# ICA_features = pd.read_csv("personal/ICA_features.csv", index_col="account_id")
    
def get_reduced_data(data, set_of_feats):
    set_of_feats = [i + 1 for i in set_of_feats]
    set_of_feats.insert(0,0)
    return data.iloc[:,set_of_feats]

def count_occurences_features(pd_feat_names):
    pd_split = pd.Series(pd_feat_names).str.split(" ", expand=True)
    for col in pd_split.columns:
        print(pd_split[col].value_counts(), "\n")
    return


def determine_observation_period_yearly(data_date, observation_length):
    """
    This function determines the desired start and end date for the yearly analysis.

    Args:
        data_date (pd.DataFrame) :  dates on which actions occur in datetime format.
        observation_length (int) : amount of recent months you want for the analysis.

    Returns:
        start_date (timestamp) : start date for the analysis.
        end_date (timestamp) : end date for the analysis.

    """
    last_date = data_date.iloc[-1].to_period('M').to_timestamp() + relativedelta(months=1)
    start_date = last_date.to_period("M").to_timestamp() - relativedelta(
        years=observation_length
    )
    end_date = last_date.to_period("M").to_timestamp() - relativedelta(days=1)
        
    # first_date = data_date.iloc[0].to_period('M').to_timestamp() 
    # last_date = data_date.iloc[-1].to_period('M').to_timestamp() + relativedelta(months=1)
    # if first_date  <= last_date - relativedelta(years=observation_length):
    #     start_date = last_date.to_period("M").to_timestamp() - relativedelta(
    #         years=observation_length
    #     )
    #     end_date = last_date.to_period("M").to_timestamp() - relativedelta(days=1)
    # else:
    #     raise ValueError("data is not full observation length")

    return start_date, end_date


def combine_multiple_datapoints_on_one_date(data, combine_fill_method):
    """
    This function combines, for example, multiple transactions that occur on the
    same day into one large transaction. Transactions have to be summed on one
    day, but for balances the last one is taken. Hence the combine_fill_method
    variable.

    Args:
        data (pd.DataFrame()) : column with datetime and column to be combined.
        combine_fill_method (str) : 'balance' or 'transaction'.

    Returns:
        combined_data (pd.DataFrame()) : column with datetime and combined column.

    """
    if combine_fill_method == "balance":
        combined_data = pd.DataFrame(
            [[data.iloc[0, 0], data.iloc[-1, 1]]], columns=data.columns
        )
    elif combine_fill_method == "transaction":
        combined_data = pd.DataFrame(
            [[data.iloc[0, 0], data.iloc[:, 1].sum()]], columns=data.columns
        )
    else:
        raise ValueError(
            "invalid combine_fill_method, please choose 'balance' or 'transaction'"
        )
    return combined_data


def fill_empty_dates(data, combine_fill_method, start_date, end_date):
    """
    This function fills the empty dates where no actions occur. This is
    necessary for the signal processing methods to work. A column with transactions
    is filled with 0's, but for balance it is forwardfilled and then backward.

    Args:
        data (pd.DataFrame()) : dataframe with only dates where actions occur.
        combine_fill_method (str) : 'balance' or 'transaction'.
        start_date (timestamp) : start date for the analysis.
        end_date (timestamp) : end date for the analysis.

    Returns:
        data_filled (pd.DataFrame()) : filled dataframe with length of observation period.


    """
    # create range of dates for analysis, then merge to select only relevant data
    dates = pd.DataFrame(
        pd.date_range(start_date, end_date, freq="D", name=data.columns[0])
    )
    data_period = dates.merge(data, how="inner")

    # move data on leap-year-day (29 feb) to 28 feb  ## FUNCTIONING BADLY
    data_period.iloc[:, 0][
        data_period.iloc[:, 0].astype(str).str.endswith("02-29")
    ] = data_period.iloc[:, 0][
        data_period.iloc[:, 0].astype(str).str.endswith("02-29")
    ] - pd.Timedelta(
        days=1
    )  # deze is niet goed, moet het niet gewoon zijn = 28feb?

    # Trying: not working
    # data_period.iloc[:,0][data_period.iloc[:,0].astype(str).str.endswith("02-29")] -= pd.Timedelta(days=1) #deze is niet goed, moet het niet gewoon zijn = 28feb?

    # combine multiple datapoints that occur on same day
    data_combined = (
        data_period.groupby(data_period.iloc[:, 0].dt.to_period("D"))
        .apply(
            combine_multiple_datapoints_on_one_date,
            combine_fill_method=combine_fill_method,
        )
        .reset_index(drop=True)
    )

    # drop 29th of feb and merge with range of dates to get nans
    dates = dates[~dates.date.astype(str).str.endswith("02-29")]
    data_empty = dates.merge(data_combined, how="outer")

    # fill the nans in the dataframe in specific way
    if combine_fill_method == "balance":
        data_filled = data_empty.fillna(method="ffill")
        data_filled = data_filled.fillna(method="bfill")
    elif combine_fill_method == "transaction":
        data_filled = data_empty.fillna(0)
    else:
        ValueError(
            "invalid combine_fill_method, please choose 'balance' or 'transaction'"
        )

    return data_filled



def prepare_data_yearly(data, fill_combine_method, observation_length):
    """
    This function selects the desired observation length and fills the dataframe.
    This is done specifically for the yearly analysis. The data is handled
    differently depending on whether it's transaction or balance data.

    Args:
        data (pd.DataFrame()) : dataframe with only dates where actions occur.
        combine_fill_method (str) : 'balance' or 'transaction'.
        observation_length (int) : amount of recent months you want for the analysis.

    Returns:
        data_filled (pd.DataFrame()) : filled dataframe with length of observation period.

    """
    start_date, end_date = determine_observation_period_yearly(
        data.iloc[:, 0], observation_length
    )
    data_filled = fill_empty_dates(data, fill_combine_method, start_date, end_date)
    return data_filled



def count_na(list_of_dfs):
    """
    PERSONAL FUNCTION, not part of open source (possible unit test):
    This function counts the amount of NaN's in a list of dataframes. This is
    useful for check if feature creation malfunctions.

    Args:
        list_of_dfs (list of pd.DataFrames()) : dataframes to count NaN's in

    Returns:
        None

    """
    na_list = []
    for df in list_of_dfs:
        na_list.append(df.isna().sum().sum())
    if sum(na_list) != 0:
        print("\nNo success:\n\n", na_list)
    else:
        print("\nSuccess\n")
    return


def combine_features_dfs(list_of_dfs):
    """
    This function merges a list of dataframes into one large dataframe.
    Merge happens on index, which in this case is the identifier.

    Args:
        list_of_dfs (list of pd.DataFrames()) : dataframes to combine

    Returns:
        combined_dfs (pd.DataFrame()) :

    """
    combine_these_dfs = []
    for df in list_of_dfs:
        if len(df) > 0:
            combine_these_dfs.append(df)

    combined_dfs = reduce(
        lambda left, right: pd.merge(
            left, right, left_index=True, right_index=True, how="inner"
        ),
        combine_these_dfs,
    )
    return combined_dfs


def select_features_subset(data, list_subset):
    """
    This function allows users to test different subsets of features against
    each other. By comparing for example the basic ["B"] features subset with the
    the basic and fourier ["B", "F"] features subset, one can see the added
    value of the fourier features.

    list_subset:
        "B" for Basic - min, max, mean, kurt ,skew, std.
        "F" for Fourier - n largest frequencies and their values.
        "W" for Wavelet - all approximation and details coefficients at each depth.
        "W_B" for Wavelet Basic - "B"/Basic (min, max, etc) at each depth.

    Args:
        data (pd.DataFrame()) : dataframe with all features
        list_subset (list of str) : list with desired subset of features

    Returns:
        data_subset (pd.DataFrame()) : dataframe with subset of featuers

    """
    data_B = pd.DataFrame()
    data_F = pd.DataFrame()
    data_W = pd.DataFrame()
    data_W_B = pd.DataFrame()
    data_tr = pd.DataFrame()
    data_ba = pd.DataFrame()
    list_subset_dfs = []

    if "B" in list_subset:
        data_B = data[
            [
                col
                for col in data.columns
                if ("fft" not in col and "wavelet" not in col and "wav_B" not in col)
            ]
        ]
        list_subset_dfs.append(data_B)
    if "F" in list_subset:
        data_F = data[[col for col in data.columns if "fft" in col]]
        list_subset_dfs.append(data_F)
    if "W" in list_subset:
        data_W = data[[col for col in data.columns if "wavelet" in col]]
        list_subset_dfs.append(data_W)
    if "W_B" in list_subset:
        data_W_B = data[[col for col in data.columns if "wav_B" in col]]
        list_subset_dfs.append(data_W_B)
    if "tr" in list_subset:
        data_tr = data[[col for col in data.columns if "tr" in col]]
        list_subset_dfs.append(data_tr)
    if "ba" in list_subset:
        data_ba = data[[col for col in data.columns if "ba" in col]]
        list_subset_dfs.append(data_ba)

    data_subset = combine_features_dfs(list_subset_dfs)
    if len(data_subset) == 0:
        raise ValueError("Please pick from types of features")

    return data_subset


def write_out_list_dfs(list_names, list_dfs, location):
    for name, item in zip(list_names,list_dfs):
        print("Writing %s" %name)
        eval("%s" %name).to_csv(f"personal/results/{location}/{name}.csv", index=False)
    return



#%% quarterly
    


def determine_observation_period_quarterly(data_date, observation_length):
    """
    This function determines the desired start and end date for the monthly analysis.

    Args:
        data_date (pd.DataFrame()) :  dates on which actions occur in datetime format.
        observation_length (int) : amount of recent months you want for the analysis.

    Returns:
        start_date (timestamp) : start date for the analysis.
        end_date (timestamp) : end date for the analysis.

    """
    last_date = data_date.iloc[-1].to_period('M').to_timestamp() + relativedelta(months=1)
    start_date = last_date.to_period("M").to_timestamp() - relativedelta(
        months=(3*observation_length) - 1
    )
    end_date = (
        last_date.to_period("M").to_timestamp()
        + relativedelta(months=1)
        - relativedelta(days=1)
    )

        
    # first_date = data_date.iloc[0].to_period('M').to_timestamp() 
    # last_date = data_date.iloc[-1].to_period('M').to_timestamp() + relativedelta(months=1)
    # if first_date <= last_date - relativedelta(months=3*observation_length):
    #     start_date = last_date.to_period("M").to_timestamp() - relativedelta(
    #         months=(3*observation_length) - 1
    #     )
    #     end_date = (
    #         last_date.to_period("M").to_timestamp()
    #         + relativedelta(months=1)
    #         - relativedelta(days=1)
    #     )
    # else:
    #     raise ValueError("data is not full observation length")

    return start_date, end_date



def prepare_data_quarterly(data, fill_combine_method, observation_length):
    """
    This function selects the desired observation length and fills the dataframe.
    This is done specifically for the monthly analysis. The data is handled
    differently depending on whether it's transaction or balance data.

    Args:
        data (pd.DataFrame()) : dataframe with only dates where actions occur.
        combine_fill_method (str) : 'balance' or 'transaction'.
        observation_length (int) : amount of recent months you want for the analysis.

    Returns:
        data_filled (pd.DataFrame()) : filled dataframe with length of observation period.

    """
    start_date, end_date = determine_observation_period_quarterly(
        data.iloc[:, 0], observation_length
    )
    data_filled = fill_empty_dates(data, fill_combine_method, start_date, end_date)
    return data_filled


def take_last_year(data):
    first_date = data.date.iloc[0].to_period('M').to_timestamp() 
    last_date = data.date.iloc[-1].to_period('M').to_timestamp() + relativedelta(months=1) - relativedelta(days=1)    
    start_date = last_date - relativedelta(years=1) + relativedelta(days=1)
    if start_date < first_date:
        return
    else:
        data_sel = data[data.date>start_date]
    return data_sel



def compute_list_featuretypes(
    data,
    list_featuretypes,
    fourier_n_largest_frequencies,
    wavelet_depth,
    mother_wavelet,
):
    """
    This function lets the user choose which combination of features they
    want to have computed. Please note that "W" will not be computed for the
    overall data. This is because "W" depends on len(data), which varies for overall.

    list_featuretypes:
        "B" for Basic - min, max, mean, kurt ,skew, std, sum.
        "F" for Fourier - n largest frequencies and their values.
        "W" for Wavelet - all approximation and details coefficients at each depth.
        "W_B" for Wavelet Basic - takes "B"/Basic (min, max, etc) at each depth.

    Args:
        data (pd.DataFrame()) : one column from which to make features.
        list_featuretypes (list) : list of feature types to be computed.
        fourier_n_largest_frequencies (int) : amount of fourier features.
        wavelet_depth (int) : level of depth up to which the wavelet is computed.
        mother_wavelet (str) : type of wavelet used for the analysis.

    Returns:
        features (pd.DataFrame()) : row of features.

    """
    
    if type(list_featuretypes) != list:
        raise AttributeError("'list_featuretypes' must be a list.")
    
    allowed_components = ["B", "F", "F2", "W", "W_B"]
    for argument in list_featuretypes:
        if argument not in allowed_components:
            raise ValueError(f"argument must be one of {allowed_components}")
    
    features_basic = pd.DataFrame()
    features_fourier = pd.DataFrame()
    features_wavelet = pd.DataFrame()
    features_wavelet_basic = pd.DataFrame()
    features_fft2 = pd.DataFrame()
    if "B" in list_featuretypes:
        features_basic = compute_basic(data)
        features_basic.columns = [
            "B " + str(col)
            for col in features_basic.columns
        ]
    if "F" in list_featuretypes:
        features_fourier = compute_fourier(data, fourier_n_largest_frequencies)
    if "F2" in list_featuretypes:
        features_fft2 = compute_fft2(data)
    if "W" in list_featuretypes:
        features_wavelet = compute_wavelet(data, wavelet_depth, mother_wavelet)
    if "W_B" in list_featuretypes:
        features_wavelet_basic = compute_wavelet_basic(
            data, wavelet_depth, mother_wavelet
        )
    features = pd.concat(
        [features_basic, features_fourier, features_fft2, features_wavelet, features_wavelet_basic],
        axis=1,
    )
    return features


def compute_fourier(data, fourier_n_largest_frequencies):
    """
    This function takes the Fast Fourier Transform and returns the n largest
    frequencies and their values.

    "F" for Fourier - n largest frequencies and their values.

    Args:
        data (pd.DataFrame()) : one column from which to make fourier features.
        fourier_n_largest_frequencies (int) : amount of fourier features.
            possible values: less than len(data)

    Returns:
        features (pd.DataFrame()) : (1 x 2n) row of largest frequencies and values.

    """
    # Fast Fourier Transform
    fft = scipy.fft.fft(data.values)
    fft_abs = abs(fft[range(int(len(data) / 2))])

    # Select largest indexes (=frequencies) and their values
    largest_indexes = np.argsort(-fft_abs)[:fourier_n_largest_frequencies]
    largest_values = fft_abs[largest_indexes]
    largest_values = [int(a) for a in largest_values]

    # Name the columns
    features = [*largest_indexes.tolist(), *largest_values]
    col_names_index = [
        "fft index_" + str(i + 1) + "/" + str(fourier_n_largest_frequencies)
        for i in range(int(len(features) / 2))
    ]
    col_names_size = [
        "fft size_" + str(i + 1) + "/" + str(fourier_n_largest_frequencies)
        for i in range(int(len(features) / 2))
    ]
    col_names = [*col_names_index, *col_names_size]
    features = pd.DataFrame([features], columns=col_names)
    return features



def compute_fft2(data):
    """
    This function takes the Fast Fourier Transform and returns the n largest
    frequencies and their values.

    "F2" for Fourier2 - frequencies and values which are largest for all accounts.

    Args:
        data (pd.DataFrame()) : one column from which to make fourier features.
        fourier_n_largest_frequencies (int) : amount of fourier features.
            possible values: less than len(data)

    Returns:
        features (pd.DataFrame()) : (1 x 2n) row of largest frequencies and values.

    """
    if (
        len(data) < 35 and len(data) > 23
    ):  # due to varying month lengths only first 28 days are used ...
        data = data[:28]

    if (
        len(data) < 95 and len(data) > 85
    ):  # due to varying quarter lengths only first 88 days are used ...
        data = data[:88]
        
    # Fast Fourier Transform
    fft = scipy.fft.fft(data.values)
    fft_sel = fft[range(int(len(data) / 2))]
    
    fft_real = [abs(np.real(a)) for a in fft_sel]
    fft_imag = [np.imag(a) for a in fft_sel]
    


    # Name the columns
    features = [*fft_real, *fft_imag]
    col_names_real = [
        "f2 real_" + str(i + 1) + "/" + str(len(data)/2)
        for i in range(int(len(features) / 2))
    ]
    col_names_imag = [
        "f2 imag_" + str(i + 1) + "/" + str(len(data)/2)
        for i in range(int(len(features) / 2))
    ]
    col_names = [*col_names_real, *col_names_imag]
    features = pd.DataFrame([features], columns=col_names)
    return features




def compute_basic(data):
    """
    This function creates basic features.

    "B" for Basic - min, max, mean, kurt ,skew, std, sum.

    Args:
        data (pd.DataFrame()) : one column from which to make basic features.

    Returns:
        features (pd.DataFrame()) : (1 x 7) row of basic features.

    """
    col_names = ["min", "max", "mean", "skew", "kurt", "std", "sum"]
    features = pd.DataFrame(
        [
            [
                data.min(),
                data.max(),
                data.mean(),
                data.skew(),
                data.kurt(),
                data.std(),
                data.sum(),
            ]
        ],
        columns=col_names,
    )
    return features


def compute_wavelet(data, wavelet_depth, mother_wavelet):
    """
    This function takes the Wavelet Transform and returns all approximation
    and details coefficients at each depth.

    "W" for Wavelet - all approximation and details coefficients at each depth.

    Args:
        data (pd.DataFrame()) : one column from which to make wavelet features.
        wavelet_depth (int) : level of depth up to which the wavelet is computed.
            possible values: depends on len(data), approx 2^wavelet_depth = len(data)
        mother_wavelet (str) : type of wavelet used for the analysis.
            possible values: "db2", "db4", "haar", see pywt.wavelist(kind="discrete")

    Returns:
        features (pd.DataFrame()) : row of wavelet features.

    """
    if (
        len(data) < 35 and len(data) > 23
    ):  # due to varying month lengths only first 28 days are used ...
        data = data[:28]

    if (
        len(data) < 95 and len(data) > 85
    ):  # due to varying quarter lengths only first 88 days are used ...
        data = data[:88]

    wavelet = pywt.wavedec(data, wavelet=mother_wavelet, level=wavelet_depth)
    features = [item for sublist in wavelet for item in sublist]  # flatten list

    col_names = ["wavelet depth_" + str(i + 1) for i in range(len(features))]
    features = pd.DataFrame([features], columns=col_names)
    return features


def compute_wavelet_basic(data, wavelet_depth, mother_wavelet):
    """
    This function takes the Wavelet Transform and at each depth makes basic
    features for the approximation/DETAIL? coefficients.

    "W_B" for Wavelet Basic - takes "B"/Basic (min, max, etc) at each depth.

    Args:
        data (pd.DataFrame()) : one column from which to make basic wavelet features.
        wavelet_depth (int) : level of depth up to which the wavelet is computed.
            possible values: depends on len(data), approx 2^wavelet_depth = len(data)
        mother_wavelet (str) : type of wavelet used for the analysis.
            possible values: "db2", "db4", "haar", see pywt.wavelist(kind="discrete")

    Returns:
        features (pd.DataFrame()) : (2 x 7 x wavelet_depth) row of wavelet features.

    """
    data_wavelet = data
    features = pd.DataFrame()
    for i in range(wavelet_depth):
        data_wavelet, coeffs = pywt.dwt(data_wavelet, wavelet=mother_wavelet)
        features_at_depth = compute_basic(pd.Series(data_wavelet))
        features_at_depth.columns = [
            "wav_B depth_" + str(i + 1) + "_" + str(col)
            for col in features_at_depth.columns
        ]
        features_at_depth_high = compute_basic(pd.Series(coeffs))
        features_at_depth_high.columns = [
            "wav_B_high depth_" + str(i + 1) + " " + str(col)
            for col in features_at_depth_high.columns
        ]
        features = pd.concat(
            [features, features_at_depth, features_at_depth_high], axis=1
        )
    return features


#%%


def compute_features_yearly(
    data,
    combine_fill_method,
    list_featuretypes,
    observation_length,
    fourier_n_largest_frequencies,
    wavelet_depth,
    mother_wavelet,
    normalize,
):
    """
    This function computes different types of features for one identifier.
    It does this yearly for a specified length of the data. The feature creation
    can be tweaked through several variables.

    list_featuretypes:
        "B" for Basic - min, max, mean, kurt ,skew, std, sum.
        "F" for Fourier - n largest frequencies and their values.
        "W" for Wavelet - all approximation and details coefficients at each depth.
        "W_B" for Wavelet Basic - takes "B"/Basic (min, max, etc) at each depth.

    Args:
        data (pd.DataFrame()) : data from one identifier for which to make features.
        combine_fill_method (str) : 'balance' or 'transaction'.
        observation_length (int) : amount of recent months you want for the analysis.
        list_featuretypes (list) : list of feature types to be computed.
        fourier_n_largest_frequencies (int) : amount of fourier features.
        wavelet_depth (int) : level of depth up to which the wavelet is computed.
            possible values: 6 is the max, depends on len(used_data)
        mother_wavelet (str) : type of wavelet used for the analysis.
            possible values: "db2", "db4", "haar", see pywt.wavelist(kind="discrete")

    Returns:
        features (pd.DataFrame()) : row of yearly features for one identifier.

    """
    # drop identifier column
    data = data.drop(data.columns[0], axis=1)

    # select only relevant period and fill the empty date
    prepared_data = prepare_data_yearly(data, combine_fill_method, observation_length)

    start_date = prepared_data.iloc[0, 0]

    # create features per year
    features = pd.DataFrame()
    for year in range(0, observation_length):
        data_year = prepared_data[
            (prepared_data.iloc[:, 0] >= start_date + relativedelta(years=year))
            & (prepared_data.iloc[:, 0] < start_date + relativedelta(years=year + 1))
        ]
        used_data = data_year.iloc[:,1]
        if normalize == True:
            used_data = (used_data-used_data.min())/(used_data.max()-used_data.min())
        yearly_features = compute_list_featuretypes(
            used_data,
            list_featuretypes,
            fourier_n_largest_frequencies,
            wavelet_depth,
            mother_wavelet,
        )
        # name columns
        yearly_features.columns = [
            data.columns[1][:2]
            + " Y_"
            + str(year + 1)
            + "/"
            + str(observation_length)
            + " "
            + col
            for col in yearly_features.columns
        ]
        features = pd.concat([features, yearly_features], axis=1)
    return features



def create_all_features(data, list_featuretypes, mother_wavelet="db2", normalize=False):
    """
    PERSONAL FUNCTION, not part of open source:
    This function creates all features for transactions and balances.
    It also times how long it takes for the monthly, yearly and overall creation.
    It also checks whether there are any NaN's in the result and then combines
    it into one large dataframe.
    
    list_featuretypes:
    "B" for Basic - min, max, mean, kurt ,skew, std, sum.
    "F" for Fourier - n largest frequencies and their values.
    "W" for Wavelet - is NOT APPLICABLE for overall
    "W_B" for Wavelet Basic - takes "B"/Basic (min, max, etc) at each depth.

    Args:
        data (pd.DataFrame()) : data for which to make features.
        list_featuretypes (list) : list of feature types to be computed.
        mother_wavelet (str) : type of wavelet used for the analysis.
            possible values: "db2", "db4", "haar", see pywt.wavelist(kind="discrete")

    Returns:
        features (pd.DataFrame()) : dataframe with all features for all identifiers.

    """
    list_featuretypes = list_featuretypes.copy()
    
    current = timeit.default_timer()  
    transaction_features_quarterly = feature_creation_quarterly(
        data[["account_id", "date", "transaction"]],
        "account_id",
        "transaction",
        normalize,
        list_featuretypes,
        mother_wavelet=mother_wavelet,
    )
    balance_features_quarterly = feature_creation_quarterly(
        data[["account_id", "date", "balance"]],
        "account_id",
        "balance",
        normalize,
        list_featuretypes,
        mother_wavelet=mother_wavelet,
    )
    print("quarterly:", int(timeit.default_timer() - current), "seconds")
    current = timeit.default_timer()    
    
    transaction_features_yearly = feature_creation_yearly(
        data[["account_id", "date", "transaction"]],
        "account_id",
        "transaction",
        normalize,
        list_featuretypes,
        mother_wavelet=mother_wavelet,
    )
    balance_features_yearly = feature_creation_yearly(
        data[["account_id", "date", "balance"]],
        "account_id",
        "balance",
        normalize,
        list_featuretypes,
        mother_wavelet=mother_wavelet,
    )
    print("yearly:", int(timeit.default_timer() - current), "seconds")

    list_features_dfs = [
        transaction_features_quarterly,
        balance_features_quarterly,
        transaction_features_yearly,
        balance_features_yearly,
    ]
    count_na(list_features_dfs)

    all_features = combine_features_dfs(list_features_dfs)
    
    if normalize == True:
        all_features.columns = ["norm " + col for col in all_features.columns]
    else:
        all_features.columns = ["reg " + col for col in all_features.columns]
    
    return all_features


#%%


def feature_creation_yearly(
    data,
    grouper,
    combine_fill_method,
    normalize,
    list_featuretypes=["B"],
    observation_length=1,
    fourier_n_largest_frequencies=30,
    wavelet_depth=6,
    mother_wavelet="db2",
):
    """
    This function splits the data per identifier and performs the yearly feature
    creation.

    list_featuretypes:
        "B" for Basic - min, max, mean, kurt ,skew, std, sum.
        "F" for Fourier - n largest frequencies and their values.
        "W" for Wavelet - all approximation and details coefficients at each depth.
        "W_B" for Wavelet Basic - takes "B"/Basic (min, max, etc) at each depth.

    Args:
        data (pd.DataFrame()) : data from one identifier for which to make features.
        combine_fill_method (str) : 'balance' or 'transaction'.
        observation_length (int) : amount of recent months you want for the analysis.
        list_featuretypes (list) : list of feature types to be computed.
        fourier_n_largest_frequencies (int) : amount of fourier features.
        wavelet_depth (int) : level of depth up to which the wavelet is computed.
            possible values: 6 is the max, depends on len(data)
        mother_wavelet (str) : type of wavelet used for the analysis.
            possible values: "db2", "db4", "haar", see pywt.wavelist(kind="discrete")

    Returns:
        features (pd.DataFrame()) : df with row of yearly features for each identifier.

    """
    features = (
        data.groupby(grouper)
        .apply(
            compute_features_yearly,
            combine_fill_method=combine_fill_method,
            list_featuretypes=list_featuretypes,
            observation_length=observation_length,
            fourier_n_largest_frequencies=fourier_n_largest_frequencies,
            wavelet_depth=wavelet_depth,
            mother_wavelet=mother_wavelet,
            normalize=normalize,
        )
        .reset_index(level=1, drop=True)
    )
    return features




#%% quarterly
    


def compute_features_quarterly(
    data,
    combine_fill_method,
    list_featuretypes,
    observation_length,
    fourier_n_largest_frequencies,
    wavelet_depth,
    mother_wavelet,
    normalize,
):
    """
    This function computes different types of features for one identifier.
    It does this monthly for a specified length of the data. The feature creation
    can be tweaked through several variables.

    list_featuretypes:
        "B" for Basic - min, max, mean, kurt ,skew, std, sum.
        "F" for Fourier - n largest frequencies and their values.
        "W" for Wavelet - all approximation and details coefficients at each depth.
        "W_B" for Wavelet Basic - takes "B"/Basic (min, max, etc) at each depth.

    Args:
        data (pd.DataFrame()) : data from one identifier for which to make features.
        combine_fill_method (str) : 'balance' or 'transaction'.
        observation_length (int) : amount of recent months you want for the analysis.
        list_featuretypes (list) : list of feature types to be computed.
        fourier_n_largest_frequencies (int) : amount of fourier features.
        wavelet_depth (int) : level of depth up to which the wavelet is computed.
            possible values: 3 is the max, depends on len(used_data)
        mother_wavelet (str) : type of wavelet used for the analysis.
            possible values: "db2", "db4", "haar", see pywt.wavelist(kind="discrete")

    Returns:
        features (pd.DataFrame()) : row of monthly features for one identifier.

    """
    # drop identifier column
    data = data.drop(data.columns[0], axis=1)

    # select only relevant period and fill the empty date
    prepared_data = prepare_data_quarterly(data, combine_fill_method, observation_length)

    start_date = prepared_data.iloc[0, 0]

    # create features per month
    features = pd.DataFrame()
    for quarter in range(0, observation_length):
        data_quarter = prepared_data[
            (prepared_data.iloc[:, 0] >= start_date + relativedelta(months=3*quarter))
            & (prepared_data.iloc[:, 0] < start_date + relativedelta(months=3*quarter + 3))
        ]
        used_data = data_quarter.iloc[:,1]
        if normalize == True:
            used_data = (used_data-used_data.min())/((used_data.max()+1)-used_data.min())
        
        quarterly_features = compute_list_featuretypes(
            used_data,
            list_featuretypes,
            fourier_n_largest_frequencies,
            wavelet_depth,
            mother_wavelet,
        )
        # name columns
        quarterly_features.columns = [
            data.columns[1][:2]
            + " Q_"
            + str(quarter + 1)
            + "/"
            + str(observation_length)
            + " "
            + col
            for col in quarterly_features.columns
        ]
        features = pd.concat([features, quarterly_features], axis=1)
    return features




def feature_creation_quarterly(
    data,
    grouper,
    combine_fill_method,
    normalize,
    list_featuretypes=["B"],
    observation_length=4,
    fourier_n_largest_frequencies=10,
    wavelet_depth=4,
    mother_wavelet="db2",
):
    """
    This function splits the data per identifier and performs the monthly feature
    creation.

    list_featuretypes:
        "B" for Basic - min, max, mean, kurt ,skew, std, sum.
        "F" for Fourier - n largest frequencies and their values.
        "W" for Wavelet - all approximation and details coefficients at each depth.
        "W_B" for Wavelet Basic - takes "B"/Basic (min, max, etc) at each depth.

    Args:
        data (pd.DataFrame()) : data from one identifier for which to make features.
        combine_fill_method (str) : 'balance' or 'transaction'.
        observation_length (int) : amount of recent months you want for the analysis.
        list_featuretypes (list) : list of feature types to be computed.
        fourier_n_largest_frequencies (int) : amount of fourier features.
        wavelet_depth (int) : level of depth up to which the wavelet is computed.
            possible values: 3 is the max, depends on len(data)
        mother_wavelet (str) : type of wavelet used for the analysis.
            possible values: "db2", "db4", "haar", see pywt.wavelist(kind="discrete")
    Returns:
        features (pd.DataFrame()) : df with row of monthly features for each identifier.

    """
    features = (
        data.groupby(grouper)
        .apply(
            compute_features_quarterly,
            combine_fill_method=combine_fill_method,
            list_featuretypes=list_featuretypes,
            observation_length=observation_length,
            fourier_n_largest_frequencies=fourier_n_largest_frequencies,
            wavelet_depth=wavelet_depth,
            mother_wavelet=mother_wavelet,
            normalize=normalize,
        )
        .reset_index(level=1, drop=True)
    )
    return features

