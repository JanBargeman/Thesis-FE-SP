import pandas as pd
from dateutil.relativedelta import relativedelta
import os

from sklearn.decomposition import PCA, FastICA

os.chdir("/Users/Jan/Desktop/Thesis/Thesis-FE-SP")

from spoef.utils import (
    prepare_data_monthly,
    prepare_data_yearly,
    prepare_data_overall,
    count_na,
    combine_features_dfs,
)
from spoef.feature_generation import compute_list_featuretypes



#%% For testing functions

data = pd.read_csv("personal/data/data.csv")
data.date = pd.to_datetime(data.date, format="%y%m%d")

#%% make test data
data = data.iloc[0:2000,:]
# data_acc = data[data.account_id == 1787]
# data_used = data_acc[["date","balance"]]
# data = data[data.account_id == 276]
# data = data[data.account_id == 1843]
#%%

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
    transaction_features_monthly = feature_creation_monthly_transformed(
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

    # print("yearly:", int(timeit.default_timer() - current), "seconds")
    # current = timeit.default_timer()  # 228

    transaction_features_overall = feature_creation_overall_transformed(
        data[["account_id", "date", "transaction", "balance"]],
        "account_id",
        "transaction",
        list_featuretypes,
        mother_wavelet=mother_wavelet,
    )

    # print("overall:", int(timeit.default_timer() - current), "seconds")  # 533

    list_features_dfs = [
        transaction_features_monthly,
        transaction_features_yearly,
        transaction_features_overall,
    ]
    count_na(list_features_dfs)

    all_features = combine_features_dfs(list_features_dfs)
    
    if transform_type == 'PCA':
        all_features.columns = ["PCA " + col for col in all_features.columns]
    elif transform_type == 'ICA':
        all_features.columns = ["ICA " + col for col in all_features.columns]

    return all_features


def feature_creation_monthly_transformed(
    data,
    grouper,
    combine_fill_method,
    list_featuretypes=["B"],
    observation_length=12,
    fourier_n_largest_frequencies=10,
    wavelet_depth=3,
    mother_wavelet="db2",
):

    features = (
        data.groupby(grouper)
        .apply(
            compute_features_monthly_transformed,
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


def feature_creation_overall_transformed(
    data,
    grouper,
    combine_fill_method,
    list_featuretypes=["B"],
    fourier_n_largest_frequencies=50,
    wavelet_depth=5,
    mother_wavelet="db2",
):
    features = (
        data.groupby(grouper)
        .apply(
            compute_features_overall_transformed,
            combine_fill_method=combine_fill_method,
            list_featuretypes=list_featuretypes,
            fourier_n_largest_frequencies=fourier_n_largest_frequencies,
            wavelet_depth=wavelet_depth,
            mother_wavelet=mother_wavelet,
        )
        .reset_index(level=1, drop=True)
    )
    return features


def compute_features_monthly_transformed(
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
    trans_data = prepare_data_monthly(data[["date", "transaction"]], "transaction", observation_length)
    bal_data = prepare_data_monthly(data[["date", "balance"]], "balance", observation_length)
    prepared_data = trans_data.merge(bal_data, on="date")
    
    start_date = trans_data.iloc[0, 0]

    # create features per month
    features = pd.DataFrame()
    for month in range(0, observation_length):
        data_month = prepared_data[
            (prepared_data.iloc[:, 0] >= start_date + relativedelta(months=month))
            & (prepared_data.iloc[:, 0] < start_date + relativedelta(months=month + 1))
        ]
        data_transformed = pd.DataFrame(transformer.fit_transform(data_month.iloc[:,[1,2]])).iloc[:,0]
        monthly_features = compute_list_featuretypes(
            data_transformed,
            list_featuretypes,
            fourier_n_largest_frequencies,
            wavelet_depth,
            mother_wavelet,
        )
        # name columns
        monthly_features.columns = [
            data.columns[1][:2]
            + " M "
            + str(month + 1)
            + "/"
            + str(observation_length)
            + " "
            + col
            for col in monthly_features.columns
        ]
        features = pd.concat([features, monthly_features], axis=1)
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
            data.columns[1][:2]
            + " Y "
            + str(year + 1)
            + "/"
            + str(observation_length)
            + " "
            + col
            for col in yearly_features.columns
        ]
        features = pd.concat([features, yearly_features], axis=1)
    return features


def compute_features_overall_transformed(
    data,
    combine_fill_method,
    list_featuretypes,
    fourier_n_largest_frequencies,
    wavelet_depth,
    mother_wavelet,
):

    # drop identifier column
    data = data.drop(data.columns[0], axis=1)

    if "W" in list_featuretypes:  # W does not work on overall data
        list_featuretypes.remove("W")

    # select only relevant period and fill the empty date
    trans_data = prepare_data_overall(data[["date", "transaction"]], "transaction")
    bal_data = prepare_data_overall(data[["date", "balance"]], "balance")
    prepared_data = trans_data.merge(bal_data, on="date")

    data_transformed = pd.DataFrame(transformer.fit_transform(prepared_data.iloc[:,[1,2]])).iloc[:,0]

    # create features overall
    features = compute_list_featuretypes(
        data_transformed,
        list_featuretypes,
        fourier_n_largest_frequencies,
        wavelet_depth,
        mother_wavelet,
    )

    # name columns
    features.columns = [data.columns[1][:2] + " O " + col for col in features.columns]
    return features



#%% PCA

PCA_features = create_all_features_transformed(data, 'PCA', ["B", "F", "W", "W_B"], "db2")

#%% Writeout PCA_features
PCA_features.to_csv("personal/PCA_features.csv")
#%% Read in PCA_features
PCA_features = pd.read_csv("personal/PCA_features.csv", index_col="account_id")




#%% ICA

ICA_features = create_all_features_transformed(data, 'ICA', ["B", "F", "W", "W_B"], "db2")

#%% Writeout ICA_features
ICA_features.to_csv("personal/ICA_features.csv")
#%% Read in ICA_features
ICA_features = pd.read_csv("personal/ICA_features.csv", index_col="account_id")