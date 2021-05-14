import os
import pandas as pd
from dateutil.relativedelta import relativedelta
import timeit
import scipy.signal  # .stft #.argrelmax  for finding max in plots or smth
import scipy.fft
import numpy as np
import matplotlib.pyplot as plt
import pywt
from functools import reduce
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from sklearn import decomposition

import pretty_errors

os.chdir("/Users/Jan/Desktop/Thesis/Thesis-FE-SP")

# scipy.detrending removal of (changing) mean
# scipy signal butter bandpass filter


#%% data import
# loan = pd.read_csv("personal/data/loan.txt", delimiter=";")
# trans = pd.read_csv("personal/data/trans.txt", delimiter=";")
# trans = trans.rename(columns={"amount":"transaction"})

# #%% data preprocess
# dataset = loan[["account_id", "status"]].merge(
#     trans[["account_id", "date", "type", "transaction", "balance"]]
# )

# dataset.loc[dataset.type == "VYDAJ", "transaction"] = -dataset.loc[
#     dataset.type == "VYDAJ", "transaction"
# ]
# dataset = dataset.drop(columns=["type"])
# dataset = dataset[["account_id", "date", "transaction", "balance", "status"]]
# dataset.to_csv("personal/data/dataset.csv", index=False)

dataset = pd.read_csv("personal/data/dataset.csv")
dataset.date = pd.to_datetime(dataset.date, format="%y%m%d")

dataset_orig = dataset.copy()

#%% make test dataset
dataset = dataset.iloc[0:2000,:]
# dataset = dataset[dataset.account_id == 1787]
# dataset = dataset[dataset.account_id == 276]
# dataset = dataset[dataset.account_id == 1843]


#%% general functions

def count_na(list_of_dfs):
    for df in list_of_dfs:
        print(df.isna().sum().sum())      
    return

def compute_features(data, feature_type, fourier_transform_n_largest_features, wavelet_transform_depth):
    "channels the 'feature_type' such that desired features are computed"
    if feature_type == "SP_all":
        features = computeSP_all(data, fourier_transform_n_largest_features, wavelet_transform_depth)
    elif feature_type == "basic":
        features = compute_basic(data)
    elif feature_type == "FT":
        features = computeSP_FT(data, fourier_transform_n_largest_features)
    elif feature_type == "WT":
        features = computeSP_WT(data, wavelet_transform_depth)
    elif feature_type == "WTB":
        features = computeSP_WTB(data, wavelet_transform_depth)
    elif feature_type == "FTWTB":
        features = computeSP_FTWTB(data, fourier_transform_n_largest_features, wavelet_transform_depth)
    elif feature_type == "ICA":
        features = compute_ICA(data)
    else:
        raise ValueError(
            'feature_type must contain one of "FT", "WT", "WTB", "FTWTB", "SP_all" or "basic"'
        )
    return features


def combine_days(data):
    "combine transactions and balances that occur on the same day"
    date = data.date[0]
    account_id = data.account_id[0]
    transaction = data.transaction.sum()
    balance = data.balance.iloc[-1]
    status = str(data.status[0])
    day_new = pd.DataFrame(
        [[date, account_id, transaction, balance, status]],
        columns=["date", "account_id", "transaction", "balance", "status"],
    )
    return day_new


def fill_balances_and_transactions(data, start_date, end_date, method_for_filling_balances, hours=24):
    "fill the dataset with respective values"
    dates = pd.DataFrame(pd.date_range(start_date, end_date, freq="D", name="date"))
    data_new = dates.merge(data, on="date", how="inner")

    # combine leap-year-day (29 feb) with 28 feb
    data_new.date[data_new.date.astype(str).str.endswith("02-29")] = data_new.date[
        data_new.date.astype(str).str.endswith("02-29")
    ] - pd.Timedelta(days=1)
    # combine multiple transactions that occur on same day
    data_new = (
        data_new.groupby(data_new.date.dt.to_period("D"))
        .apply(combine_days)
        .reset_index(drop=True)
    )

    dates = dates[~dates.date.astype(str).str.endswith("02-29")]  # drop 29th of feb
    data_new = dates.merge(data_new, on="date", how="outer")

    # fillna the dataframe
    data_new.balance = data_new.balance.fillna(
        method="ffill"
    )  # balance is forwardfilled
    if (
        method_for_filling_balances == "bfill"
    ):  # and afterwards depends on whether it's complete observation length
        data_new.balance = data_new.balance.fillna(method="bfill")
    elif method_for_filling_balances == 0:
        data_new.balance = data_new.balance.fillna(0)
    data_new.transaction = data_new.transaction.fillna(0)  # transaction is filled with 0's
    data_new.account_id = data_new.account_id.fillna(
        int(data_new.account_id.mean())
    )  # account_id with account_id
    data_new.status = data_new.status.fillna(method="ffill")  # status is forwardfilled
    data_new.status = data_new.status.fillna(method="bfill")  # and afterwards backward
    return data_new


#%% compute features on specific time basis


def monthly(dataset, feature_type, observation_length=12, fourier_transform_n_largest_features=10, wavelet_transform_depth=3):
    "grabs 'observation_length' months of the data, fills it and computes 'feature_type' features per month"
    first_date = dataset.date.iloc[0]
    last_date = dataset.date.iloc[-1]
    if first_date < last_date - relativedelta(months=observation_length):
        start_date = last_date.to_period("M").to_timestamp() - relativedelta(
            months=observation_length - 1
        )  # dropping first month of year due to varying month lengths
        end_date = (
            last_date.to_period("M").to_timestamp()
            + relativedelta(months=1)
            - relativedelta(days=1)
        )
        method_for_filling_balances = "bfill"
    else:
        print("not full obs length:", dataset.account_id.iloc[0])
        start_date = first_date.to_period("M").to_timestamp() + relativedelta(
            months=1
        )  # dropping first month of year due to varying month lengths
        end_date = (
            last_date.to_period("M").to_timestamp()
            + relativedelta(months=1)
            - relativedelta(days=1)
        )
        method_for_filling_balances = 0

    # select only start_date <-> end_date for later use groupby
    # features = dataset.groupby(dataset.date.dt.to_period('M')).apply(compute_features, feature_type) # should process start/end date
    dataset = fill_balances_and_transactions(dataset, start_date, end_date, method_for_filling_balances)

    monthly_features = pd.DataFrame()
    for month in range(0, observation_length):
        dataMonth = dataset[
            (dataset.date >= start_date + relativedelta(months=month))
            & (dataset.date < start_date + relativedelta(months=month + 1))
        ]
        features = compute_features(dataMonth, feature_type, fourier_transform_n_largest_features, wavelet_transform_depth)
        features.columns = ["M" + str(month+1) + "/" + str(observation_length) + " " + col for col in features.columns]
        monthly_features = pd.concat([monthly_features, features], axis=1)

    # monthly_features = monthly_features.reindex(sorted(monthly_features.columns),axis=1)
    
    return monthly_features


def yearly(dataset, feature_type, observation_length=2, fourier_transform_n_largest_features=20, wavelet_transform_depth=6):
    "grabs 'observation_length' years of the data, fills it and computes 'feature_type' features per year"
    first_date = dataset.date.iloc[0]
    last_date = dataset.date.iloc[-1]
    if first_date < last_date - relativedelta(years=observation_length):
        start_date = last_date.to_period("M").to_timestamp() - relativedelta(
            years=observation_length
        )  # dropping first month of year due to varying month lengths
        end_date = (
            (last_date.to_period("M")).to_timestamp()
            + relativedelta(years=1)
            - relativedelta(days=1)
        )
        method_for_filling_balances = "bfill"
    else:
        # print("suggested monthly:", dataset.account_id.iloc[0])
        start_date = first_date.to_period("M").to_timestamp()
        end_date = (
            (last_date.to_period("M")).to_timestamp()
            + relativedelta(years=1)
            - relativedelta(days=1)
        )
        method_for_filling_balances = 0

    # select only start_date <-> end_date for later use groupby
    # features = dataset.groupby(dataset.date.dt.to_period('M')).apply(compute_features, feature_type) # should process start/end date
    dataset = fill_balances_and_transactions(dataset, start_date, end_date, method_for_filling_balances)

    yearly_features = pd.DataFrame()
    for year in range(0, observation_length):
        dataYear = dataset[
            (dataset.date >= start_date + relativedelta(years=year))
            & (dataset.date < start_date + relativedelta(years=year + 1))
        ]
        features = compute_features(dataYear, feature_type, fourier_transform_n_largest_features, wavelet_transform_depth)
        features.columns = ["Y " + str(year+1) + "/" + str(observation_length) + col for col in features.columns]
        yearly_features = pd.concat([yearly_features, features], axis=1)

    return yearly_features


def overall(dataset, feature_type, fourier_transform_n_largest_features=20, wavelet_transform_depth=6):
    "grabs the entire dataset, fills it and computes 'feature_type' features over the entire length"
    first_date = dataset.date.iloc[0]
    last_date = dataset.date.iloc[-1]
    start_date = first_date.to_period("M").to_timestamp()
    end_date = last_date.to_period("M").to_timestamp()
    method_for_filling_balances = 0

    dataset_overall = fill_balances_and_transactions(dataset, start_date, end_date, method_for_filling_balances)

    overall_features = compute_features(dataset_overall, feature_type, fourier_transform_n_largest_features, wavelet_transform_depth)
    overall_features.columns = ["O " + col for col in overall_features.columns]

    return overall_features


#%% feature engineering (basic)


def compute_basic(data):
    "computes the basic features for the 'transaction' column and 'balakce' column"
    features_transaction = compute_basic_features(data["transaction"])
    features_transaction.columns = ["B transaction " + str(col) for col in features_transaction.columns] #remove str()?
    featuresBalance = compute_basic_features(data["balance"])
    featuresBalance.columns = ["B balance " + str(col) for col in featuresBalance.columns]
    features = pd.concat([features_transaction, featuresBalance], axis=1)
    return features

def compute_basic_features(data):  
    "compute basic features of input data: min max avg skw krt std"
    # fsum = dataMonth.transaction.sum()   
    col_names = ['min', 'max', 'mean', 'skew', 'kurt', 'std']
    features = pd.DataFrame([[data.min(), data.max(), data.mean(), data.skew(), data.kurt(), data.std()]], columns=col_names) 
    return features


#%% feature engineering (SP)


def computeSP_all(data, fourier_transform_n_largest_features, wavelet_transform_depth):
    "compute signal processing features, FT, WT1 and WT2"
    fourier_transaction = compute_fourier(data["transaction"], fourier_transform_n_largest_features)
    fourier_transaction.columns = ["FT amnt " + str(col) for col in fourier_transaction.columns]
    fourier_balance = compute_fourier(data["balance"], fourier_transform_n_largest_features)
    fourier_balance.columns = ["FT bal " + str(col) for col in fourier_balance.columns]
    # wavelet_transaction = compute_wavelet(data["transaction"], wavelet_transform_depth)
    # wavelet_transaction.columns = ["WT amnt " + str(col) for col in wavelet_transaction.columns]
    # wavelet_balance = compute_wavelet(data["balance"], wavelet_transform_depth)
    # wavelet_balance.columns = ["WT bal " + str(col) for col in wavelet_balance.columns]
    wavelet_transactionB = compute_wavelet_basic(data["transaction"], wavelet_transform_depth)
    wavelet_transactionB.columns = ["WTB amnt " + str(col) for col in wavelet_transactionB.columns]
    wavelet_balanceB = compute_wavelet_basic(data["balance"], wavelet_transform_depth)
    wavelet_balanceB.columns = ["WTB bal " + str(col) for col in wavelet_balanceB.columns]
    features = pd.concat([
        fourier_transaction, 
        fourier_balance, 
        # wavelet_transaction, 
        # wavelet_balance, 
        wavelet_transactionB, 
        wavelet_balanceB], axis=1)
    return features


def computeSP_FT(data, fourier_transform_n_largest_features):
    "compute signal processing features, FT"
    fourier_transaction = compute_fourier(data["transaction"], fourier_transform_n_largest_features)
    fourier_balance = compute_fourier(data["balance"], fourier_transform_n_largest_features)
    features = pd.concat([
        fourier_transaction, 
        fourier_balance], axis=1)    
    return features


def computeSP_FTWTB(data, fourier_transform_n_largest_features, wavelet_transform_depth):
    "compute signal processing features, FT and WT2"
    fourier_transaction = compute_fourier(data["transaction"], fourier_transform_n_largest_features)
    fourier_balance = compute_fourier(data["balance"], fourier_transform_n_largest_features)
    wavelet_transactionB = compute_wavelet_basic(data["transaction"], wavelet_transform_depth)
    wavelet_balanceB = compute_wavelet_basic(data["balance"], wavelet_transform_depth)
    features = pd.concat([
        fourier_amount, 
        fourier_balance, 
        wavelet_transactionB, 
        wavelet_balanceB], axis=1)
    return features


def computeSP_WT(data, wavelet_transform_depth):
    "compute signal processing features, WT1"
    wavelet_transaction = compute_wavelet(data["transaction"], wavelet_transform_depth)
    wavelet_balance = compute_wavelet(data["balance"], wavelet_transform_depth)
    features = pd.concat([
        wavelet_transaction, 
        wavelet_balance], axis=1)
    return features


def computeSP_WTB(data, wavelet_transform_depth):
    "compute signal processing features, WT2"
    wavelet_transactionB = compute_wavelet_basic(data["transaction"], wavelet_transform_depth)
    wavelet_balanceB = compute_wavelet_basic(data["balance"], wavelet_transform_depth)
    features = pd.concat([
        wavelet_transactionB, 
        wavelet_balanceB], axis=1)
    return features


def compute_fourier(data, fourier_transform_n_largest_features):
    "compute fourier transform of data and return n largest frequencies and transactions"
    fft = scipy.fft.fft(data.values)
    fft_abs = abs(fft[range(int(len(data) / 2))])

    largestIndexes = np.argsort(-fft_abs)[:fourier_transform_n_largest_features]
    largestValues = fft_abs[largestIndexes]
    largestValues = [int(a) for a in largestValues]

    features = [largestIndexes.tolist(), largestValues]
    features = [item for sublist in features for item in sublist]  # flatten list
    col_names = [str(i+1) + "/" + str(fourier_transform_n_largest_features) for i in range(len(features))]
    features = pd.DataFrame([features], columns=col_names)
    return features


def compute_wavelet(data, depth):
    "compute wavelet transform of data and return detail coefficients at each decomposition level"
    wavelet = pywt.wavedec(data, "db2", level=depth)
    features = [item for sublist in wavelet for item in sublist]  # flatten list
    col_names = [str(i+1) for i in range(len(features))]
    features = pd.DataFrame([features], columns=col_names)
    return features


def compute_wavelet_basic(data, depth):
    features = pd.DataFrame()
    for i in range(depth - 1):
        data, coeffs = pywt.dwt(data, "db2")
        featuresAtDepth = compute_basic_features(pd.Series(data))
        featuresAtDepth.columns = ["depth " + str(i) + " " + str(col) for col in featuresAtDepth.columns]
        features = pd.concat([features, featuresAtDepth], axis = 1)        
    return features




#%%
# def feature_creation_monthly(dataset,feature_type, months):
#     features = dataset.groupby("account_id").apply(monthly, feature_type=feature_type, observation_length=months).reset_index(level=1, drop=True)
#     return features
      
# def feature_creation_yearly(dataset, feature_type, years):
#     features =  dataset.groupby("account_id").apply(yearly, feature_type=feature_type, observation_length=years).reset_index(level=1, drop=True)
#     return features

# def feature_creation_overall(dataset, feature_type):
#     features =  dataset.groupby("account_id").apply(overall, feature_type=feature_type).reset_index(level=1, drop=True)
#     return features



# monthly_featuresB = feature_creation_monthly(dataset, "basic", 12)     
# monthly_featuresFT = feature_creation_monthly(dataset, "FT", 12)     
# monthly_featuresWT = feature_creation_monthly(dataset, "WT", 12)     
# monthly_featuresWTB = feature_creation_monthly(dataset, "WTB", 12)     



# timespan ="monthly"
# features = dataset.groupby("account_id").apply(exec(timespan), feature_type="basic")
# locals()["myfunction"]()
# features = dataset.groupby("account_id").apply(locals()["monthly"](), feature_type="basic")

#%% ICA TEST
    
transformer = decomposition.FastICA(n_components=7, random_state=0)
X = dataset[dataset.account_id==1787]

global ICA_global = decomposition.FastICA(n_components=2, random_state=0)
global PCA_global = decomposition.PCA(n_components=2,random_state=0)

def compute_ICA(data, fourier_transform_n_largest_features, wavelet_transform_depth):
    "compute ICA of dataset"
    ICA_data = compute_ICA_features(data[["transaction","balance"]])
    
    
    
    features = pd.concat([
        ICA_transaction, 
        ICA_balance], axis=1)
    return features

def transform_PCA(data):   
    return PCA_global.fit_transform(data)

def transform_ICA(data):   
    return ICA_global.fit_transform(data)

#%%
def comp_feat(list_of_feat_types):
    if "WT" in list_of_feat_types:
        print("yes")
        
    if "FT" in list_of_feat_types:
        print("yes")
    return 

list_of_feat_types = ["B", "FT", "WT", "WT_B"]

#%%
    
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
X, _ = load_digits(return_X_y=True)
transformer = FastICA(n_components=7,random_state=0)
X_transformed = transformer.fit_transform(X)
X_transformed.shape

def create_overall_dataset(dataset):  
    first_date = dataset.date.iloc[0]
    last_date = dataset.date.iloc[-1]
    start_date = first_date.to_period("M").to_timestamp()
    end_date = last_date.to_period("M").to_timestamp()
    method_for_filling_balances = 0
    dataset_overall = fill_balances_and_transactions(dataset, start_date, end_date, method_for_filling_balances)
    return dataset_overall





#%%
y_test = dataset[dataset.account_id==1787]
y_test_fill = create_dataset(y_test)
ica = FastICA(n_components=2, random_state=0)
y_transformed = ica.fit_transform(y_test_fill[["transaction","balance"]])

#%%
plt.plot(X_transformed[:,6])
plt.show()


#%% MAIN



current = timeit.default_timer()

monthly_featuresB = (
    dataset.groupby("account_id")
    .apply(monthly, feature_type="basic", observation_length=12)
    .reset_index(level=1, drop=True)
)
print(timeit.default_timer() - current); current = timeit.default_timer()

monthly_featuresSP = (
    dataset.groupby("account_id")
    .apply(monthly, feature_type="SP_all", observation_length=12)
    .reset_index(level=1, drop=True)
)
print(timeit.default_timer() - current); current = timeit.default_timer()

yearly_featuresB = (
    dataset.groupby("account_id")
    .apply(yearly, feature_type="basic", observation_length=2)
    .reset_index(level=1, drop=True)
)
print(timeit.default_timer() - current); current = timeit.default_timer()

yearly_featuresSP = (
    dataset.groupby("account_id")
    .apply(yearly, feature_type="SP_all", observation_length=2)
    .reset_index(level=1, drop=True)
)
print(timeit.default_timer() - current); current = timeit.default_timer()

overall_featuresB = (
    dataset.groupby("account_id")
    .apply(overall, feature_type="basic")
    .reset_index(level=1, drop=True)
)
print(timeit.default_timer() - current); current = timeit.default_timer()

overall_featuresSP = (
    dataset.groupby("account_id")
    .apply(overall, feature_type="FTWTB")
    .reset_index(level=1, drop=True)
)
print(timeit.default_timer() - current); current = timeit.default_timer()

count_na(
    [
        monthly_featuresB,
        monthly_featuresSP,
        yearly_featuresB,
        yearly_featuresSP,
        overall_featuresB,
        overall_featuresSP,
    ]
)

 #%%     Writing all files to a results folder (if it doesn't exist, manually create please)

# typeOfRun="FT"
# typeOfRun="WT"
typeOfRun="WTB"
# typeOfRun="FTWTWTB"


my_list = ["monthly_featuresB", "monthly_featuresSP", "yearly_featuresB", "yearly_featuresSP", "overall_featuresB", "overall_featuresSP"]
for item in my_list:
    eval("%s" %item).to_csv("personal/results/{folder}/{filename}.csv".format(folder=typeOfRun,filename=item))
    print("Writing %s" %item)

#%%

monthly_featuresB = pd.read_csv("personal/results/{folder}/monthly_featuresB.csv".format(folder=typeOfRun), index_col="account_id")
monthly_featuresSP = pd.read_csv(
    "personal/results/{folder}/monthly_featuresSP.csv".format(folder=typeOfRun), index_col="account_id"
)
yearly_featuresB = pd.read_csv("personal/results/{folder}/yearly_featuresB.csv".format(folder=typeOfRun), index_col="account_id")
yearly_featuresSP = pd.read_csv("personal/results/{folder}/yearly_featuresSP.csv".format(folder=typeOfRun), index_col="account_id")
overall_featuresB = pd.read_csv("personal/results/{folder}/overall_featuresB.csv".format(folder=typeOfRun), index_col="account_id")
overall_featuresSP = pd.read_csv(
    "personal/results/{folder}/overall_featuresSP.csv".format(folder=typeOfRun), index_col="account_id"
)


#%% combine SP/basic with status and train/test split


def combine(list_of_dfs):
    combined_dfs = reduce(
        lambda left, right: pd.merge(
            left, right, left_index=True, right_index=True, how="inner"
        ),
        list_of_dfs,
    )
    return combined_dfs


status = (
    dataset_orig[["account_id", "status"]].drop_duplicates().set_index("account_id")
)

l_data_all = [
    status,
    monthly_featuresB,
    monthly_featuresSP,
    yearly_featuresB,
    yearly_featuresSP,
    overall_featuresB,
    overall_featuresSP,
]
l_data_B = [status, monthly_featuresB, yearly_featuresB, overall_featuresB]
l_data_SP = [status, monthly_featuresSP, yearly_featuresSP, overall_featuresSP]
data_all = combine(l_data_all)
data_B = combine(l_data_B)
data_SP = combine(l_data_SP)

# df_scaled = fScaling(df_total)      # Scaling: StandardScaler     -- Deze is beter
Y_all = data_all.iloc[:, 0].values
X_all = data_all.iloc[:, 1:].values
Y_B = data_B.iloc[:, 0].values
X_B = data_B.iloc[:, 1:].values
Y_SP = data_SP.iloc[:, 0].values
X_SP = data_SP.iloc[:, 1:].values


t_size = 0.4

X_train_all, X_valid_all, y_train_all, y_valid_all = train_test_split(
    X_all, Y_all, test_size=t_size, random_state=0
)
X_train_B, X_valid_B, y_train_B, y_valid_B = train_test_split(
    X_B, Y_B, test_size=t_size, random_state=0
)
X_train_SP, X_valid_SP, y_train_SP, y_valid_SP = train_test_split(
    X_SP, Y_SP, test_size=t_size, random_state=0
)


#%% train & compare models

n_trees = 300
max_depth = None

clf_all = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth)
clf_all.fit(X_train_all, y_train_all)
print("\nall:\n", confusion_matrix(y_valid_all, clf_all.predict(X_valid_all)))
print("acc:", accuracy_score(y_valid_all, clf_all.predict(X_valid_all)))
print("AUC:", roc_auc_score(y_valid_all, clf_all.predict_proba(X_valid_all), multi_class="ovo"))


clf_B = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth)
clf_B.fit(X_train_B, y_train_B)
print("\nB:\n", confusion_matrix(y_valid_B, clf_B.predict(X_valid_B)))
print("acc:", accuracy_score(y_valid_B, clf_B.predict(X_valid_B)))
print("AUC:", roc_auc_score(y_valid_B, clf_B.predict_proba(X_valid_B), multi_class="ovo"))


clf_SP = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth)
clf_SP.fit(X_train_SP, y_train_SP)
print("\nSP:\n", confusion_matrix(y_valid_SP, clf_SP.predict(X_valid_SP)))
print("acc:", accuracy_score(y_valid_SP, clf_SP.predict(X_valid_SP)))
print("AUC:", roc_auc_score(y_valid_SP,clf_SP.predict_proba(X_valid_SP), multi_class="ovo"))


#%%
# problems:

#   verschillende lengtes van observaties, dus 4 maanden of 12 maanden

# TODO:
#   columnnames
#   fft power spectrum?
#   AUC plot ROC om eindes te vergelijken, area of interest
#   computational complexity scaling (On^2?)
#   smaller functions + some comments
#   try logistic ligthGBMregressorclassfiier


#%%


plt.hist(data_all.iloc[:, 0])




# %% sample pwyt

x = [3, 7, 1, 1, -2, 5, 4, 6]
cA, cD = pywt.dwt(x, "db2")
print(pywt.idwt(cA, cD, "db2"))

cA, cD = pywt.wavedec(x, "db2", level=4)


#%% WAVELET TEST

# datatest = dataset[dataset.account_id==1787]
# first_date = datatest.date.iloc[0]
# last_date = datatest.date.iloc[-1]
# start_date = first_date.to_period('M').to_timestamp()
# end_date = last_date.to_period('M').to_timestamp()
# method_for_filling_balances = 0
# datatest = fill_balances_and_transactions(datatest, start_date, end_date, method_for_filling_balances)

# month = 30
# dataMonth = datatest[(datatest.date >= start_date + relativedelta(months=month)) & (datatest.date < start_date + relativedelta(months=month+1))]


# #%%

# data_for_analysis = dataMonth.transaction
# cA, cD = pywt.dwt(data_for_analysis, 'db2') #max_level = 9 voor 2000 obs, 3 voor 1 maand (30)
# # cA = approximation coeff --> lowpass filter
# # cD = detail coeff --> highpass filter


# pywt.dwt_max_level(365,"db2")

# a_list_of_coefs = pywt.wavedec(data_for_analysis,'db2',level=3)

# (datas,coeffs) = pywt.dwt(data_for_analysis, 'db2')

# wtrec = pywt.idwt(cA, cD, 'db2')

# #%%
# figure, axis = plotter.subplots(2, 1)
# plotter.subplots_adjust(hspace=1)

# axis[0].plot(datatest.date, datatest.transaction)
# axis[1].plot(datatest.date, wtrec)

#%% filters, denoisers


# def waveletDenoise(data, threshold=0.63, wavelet="db2", mode="periodization"):
#     "removing noisy high frequencies from the input data by applying a wavelet threshold"
#     threshold = threshold * np.nanmax(data.iloc[:, 1].values)
#     coeff = pywt.wavedec(data.iloc[:, 1].values, wavelet, mode=mode)
#     coeff[1:] = (pywt.threshold(i, value=threshold, mode="soft") for i in coeff[1:])
#     data.iloc[:, 1] = pywt.waverec(coeff, wavelet, mode="periodization")
#     return data


# data = dataset_overall[['date','transaction']]
# test = waveletDenoise(data, 0.002) #doet het niet echt


# figure, axis = plotter.subplots(2, 1)
# plotter.subplots_adjust(hspace=1)
# axis[0].plot(data.date, test.transaction)
# axis[1].plot(data.date, data.transaction)
