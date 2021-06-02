from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import os
os.chdir("/Users/Jan/Desktop/Thesis/Thesis-FE-SP")

from spoef.utils import combine_features_dfs, count_na
from spoef.features import create_all_features, feature_creation_yearly



def grid_search_LGBM(data, test_size=0.4):
    """
    PERSONAL FUNCTION, not part of open source:
    This function performs a grid search for the LightGBM model.

    Args:
        data (pd.DataFrame()) : data of features to train lgbm model on
        test_size (0 < float < 1) : test size for AUC determination

    Returns:
        best_LGBM (lightgbm.LGBMClassifier) : the model with best performing params

    """
    Y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, Y, test_size=test_size, random_state=0
    )

    list_n_est = [10, 30, 100]
    list_max_depth = [2, 4, 8]
    list_num_leaves = [6, 10]
    list_learn_rate = [0.1, 0.05]

    # list_reg_alpha = [0.0, 10]
    # list_reg_lambda = [0.0, 10]

    length_search = (
        len(list_n_est)
        * len(list_max_depth)
        * len(list_num_leaves)
        * len(list_learn_rate)
    )
    i = 0

    max_auc = 0
    auc_list = []
    AUC_score_avg_list = []

    for n_est in list_n_est:
        for max_depth in list_max_depth:
            for num_leaves in list_num_leaves:
                for learn_rate in list_learn_rate:
                    i = i + 1
                    
                    lgbm = LGBMClassifier(
                        objective="multiclass",
                        n_estimators=n_est,
                        max_depth=max_depth,
                        num_leaves=num_leaves,
                        learning_rate=learn_rate,
                        random_state=0,
                    )                    
                    
                    skf = StratifiedKFold(n_splits=5)
                    
                    for train_index, test_index in skf.split(X,Y):
                        X_train, X_valid = X[train_index], X[test_index]
                        y_train, y_valid = Y[train_index], Y[test_index]              
                        

                        lgbm.fit(X_train, y_train)
                        
                        current_params = [n_est, max_depth, num_leaves, learn_rate]
                        AUC_score = roc_auc_score(
                            y_valid, lgbm.predict_proba(X_valid), multi_class="ovo"
                        )
                        current_conf_matrix = confusion_matrix(
                            y_valid, lgbm.predict(X_valid)
                        )
                        # print(current_params, ":")
                        # print("\n", current_conf_matr)
                        print("AUC:", AUC_score)
                        # print("\n\n")
                        print(
                            "\r",
                            "\rLoading, please wait: "
                            + "%.1f" % (100 * i / length_search)
                            + "%",
                            end="",
                        )
                        auc_list.append(AUC_score)
                    AUC_score_avg = sum(auc_list)/len(auc_list)
                    AUC_score_avg_list.append(AUC_score_avg)
                    if AUC_score_avg > max_auc:
                        max_auc = AUC_score
                        best_params = current_params
                        best_conf_matrix = current_conf_matrix
                        best_LGBM = lgbm

    print("\n\nMax AUC: " + str(max_auc) + " at " + str(best_params) + "\n")
    print(str(best_conf_matrix))
    plt.hist(AUC_score_avg_list)
    plt.show()
    print("\nMean AUC: " + str(sum(AUC_score_avg_list)/len(AUC_score_avg_list)))
    return best_LGBM, AUC_score_avg_list


def grid_search_RF(data, test_size=0.4):
    """
    PERSONAL FUNCTION, not part of open source:
    This function performs a grid search for the RandomForest model.

    Args:
        data (pd.DataFrame()) : data of features to train rf model on
        test_size (0 < float < 1) : test size for AUC determination

    Returns:
        best_RF (RandomForestClassifier) : the model with best performing params

    """
    Y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, Y, test_size=test_size, random_state=0
    )

    list_n_trees = [100, 300]
    list_max_depth = [10, 50]

    length_search = len(list_n_trees) * len(list_max_depth)
    i = 0

    max_auc = 0
    auc_list = []

    for n_trees in list_n_trees:
        for max_depth in list_max_depth:
            i = i + 1
            rf = RandomForestClassifier(
                n_estimators=n_trees, max_depth=max_depth, random_state=0
            )
            rf.fit(X_train, y_train)
            current_params = [n_trees, max_depth]
            AUC_score = roc_auc_score(
                y_valid, rf.predict_proba(X_valid), multi_class="ovo"
            )
            current_conf_matrix = confusion_matrix(y_valid, rf.predict(X_valid))
            # print(current_params, ":")
            # print("\n", current_conf_matr)
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
                best_params = current_params
                best_conf_matrix = current_conf_matrix
                best_RF = rf
    print("\n\nMax AUC: " + str(max_auc) + " at " + str(best_params) + "\n")
    print(str(best_conf_matrix))
    plt.hist(auc_list)
    plt.show()
    return best_RF


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


def search_mother_wavelet(data, status, best_lgbm_all, list_mother_wavelets, test_size):
    """
    PERSONAL FUNCTION, not part of open source:
    This function uses different mother wavelets for creating features. It then
    inputs those features in the LGBM model that got the best performance on
    the data with all features. Those features were created with the "db2"
    mother wavelet.

    Args:
        data (pd.DataFrame()) : data for which to make features.
        status : the list of identifiers and their label
        best_LGBM (lightgbm.LGBMClassifier) : the model with best performing params
        list_mother_wavelets : list of mother wavelets to try
        test_size (0 < float < 1) : test size for AUC determination

    Returns:
        auc_list : list of AUC's for each mother wavelet

    """
    list_featuretypes = ["W_B", "W"]
    max_auc = 0
    auc_list = []
    for mother_wavelet in list_mother_wavelets:
        # mother_features = create_all_features(data, list_featuretypes, mother_wavelet)
        mother_features = create_yearly_features_mother(
            data, list_featuretypes, mother_wavelet
        )
        data = combine_features_dfs([status, mother_features])
        Y = data.iloc[:, 0].values
        X = data.iloc[:, 1:].values

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, Y, test_size=test_size, random_state=0
        )

        best_lgbm_all.fit(X_train, y_train)
        AUC_score = roc_auc_score(
            y_valid, best_lgbm_all.predict_proba(X_valid), multi_class="ovo"
        )

        print(mother_wavelet, AUC_score)

        auc_list.append(AUC_score)

        if AUC_score > max_auc:
            max_auc = AUC_score
            best_mother_wavelet = mother_wavelet
    print("\n\nMax AUC: " + str(max_auc) + " at " + str(best_mother_wavelet) + "\n")
    print(str(auc_list))
    plt.hist(auc_list)
    plt.show()
    return auc_list
