import pandas as pd

from spoef.feature_generation import feature_creation_quarterly, feature_creation_yearly
from spoef.utils import combine_features_dfs

#%% Generating the data
data = pd.DataFrame([
    ['John', 1, '2021-01-03', 1000, 1000],
    ['John', 1, '2021-02-03', 1000, 2000],
    ['John', 1, '2021-03-08', -3000, -1000],
    ['Jane', 0, '2021-01-03', 1000, 1000],
    ['Jane', 0, '2021-02-03', 5000, 6000],
    ['Jane', 0, '2021-03-03', 2000, 8000],
    ],
    columns=['account_id', 'status', 'date', 'transaction', 'balance']
    )
# Make the date into datetime object
data.date = pd.to_datetime(data.date, format="%Y-%m-%d")
# Find unique accounts and their status
status = data[["account_id", "status"]].drop_duplicates().set_index("account_id")


#%% Setting up which features to generate and with which mother wavelet
list_featuretypes = ["Basic", "FourierComplete", "FourierNLargest", "WaveletComplete", "WaveletBasic"]
mother_wavelet = "db2"

#%% Generating features over 1 quarter

# For the transactions
transaction_features_quarterly = feature_creation_quarterly(
    data=data[["account_id", "date", "transaction"]],
    grouper="account_id",
    combine_fill_method="transaction",
    list_featuretypes=list_featuretypes,
    mother_wavelet=mother_wavelet,
    observation_length=1
)

# For the balances
balance_features_quarterly = feature_creation_quarterly(
    data=data[["account_id", "date", "balance"]],
    grouper="account_id",
    combine_fill_method="balance",
    list_featuretypes=list_featuretypes,
    mother_wavelet=mother_wavelet,
    observation_length=1
)



#%% Combining into one dataframe
all_features = combine_features_dfs([
    status, 
    transaction_features_quarterly, 
    balance_features_quarterly,
    ])





#%% EXTRA:

#%% Also possible for yearly
transaction_features_yearly = feature_creation_yearly(
    data=data[["account_id", "date", "transaction"]],
    grouper="account_id",
    combine_fill_method="transaction",
    list_featuretypes=list_featuretypes,
    mother_wavelet=mother_wavelet,
    observation_length=1
)

balance_features_yearly = feature_creation_yearly(
    data=data[["account_id", "date", "balance"]],
    grouper="account_id",
    combine_fill_method="balance",
    list_featuretypes=list_featuretypes,
    mother_wavelet=mother_wavelet,
    observation_length=1
)

all_features = combine_features_dfs([
    status, 
    transaction_features_quarterly, 
    transaction_features_yearly, 
    balance_features_quarterly,
    balance_features_yearly
    ])