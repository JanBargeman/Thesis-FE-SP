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

os.chdir("/Users/Jan/Desktop/Thesis/Thesis-FE-SP")

# scipy.detrending removal of (changing) mean
# scipy signal butter bandpass filter


#%% data import
loan = pd.read_csv("personal/data/loan.txt", delimiter=";")
trans = pd.read_csv("personal/data/trans.txt", delimiter=";")

#%% data preprocess
dataset = loan[["account_id", "status"]].merge(
    trans[["account_id", "date", "type", "amount", "balance"]]
)

dataset.loc[dataset.type == "VYDAJ", "amount"] = -dataset.loc[
    dataset.type == "VYDAJ", "amount"
]
dataset = dataset.drop(columns=["type"])
dataset.date = pd.to_datetime(dataset.date, format="%y%m%d")
dataset = dataset[["account_id", "date", "amount", "balance", "status"]]
dataset_orig = dataset.copy()

#%% make test dataset
dataset = dataset.iloc[0:2000,:]
dataset = dataset[dataset.account_id == 1787]
dataset = dataset[dataset.account_id == 276]


#%% general functions

data = dataset
date_column="date"

def get_last_n_months(data, date_column="date", n):
    
    first_date = data[date_column].iloc[0]
    last_date = data[date_column].iloc[-1]
    if first_date < last_date - relativedelta(months=n):
        start_date = (
            last_date.to_period("M").to_timestamp() 
            - relativedelta(months=n-1)
        )
        end_date = (
            last_date.to_period("M").to_timestamp()
            + relativedelta(months=1)
            - relativedelta(days=1)
        )
    else:
        raise ValueError("observation length too short")
        
    data_new = data[data[date_column] > start_date]
    return data_new

data = data_new

def merge_with_full_observation_length(data, date_column="date", n):
    last_date = data[date_column].iloc[-1]
    start_date = (
        last_date.to_period("M").to_timestamp() 
        - relativedelta(months=n-1)
    )
    end_date = (
        last_date.to_period("M").to_timestamp()
        + relativedelta(months=1)
        - relativedelta(days=1)
    )
    
    dates = pd.DataFrame(pd.date_range(start_date, end_date, freq="D", name=date_column))
    data_new = dates.merge(data, on=date_column, how="inner")
    dates = dates[~dates[date_column].astype(str).str.endswith("02-29")]  # drop 29th of feb

    # combine leap-year-day (29 feb) with 28 feb
    data_new[date_column][data_new[date_column].astype(str).str.endswith("02-29")] = data_new[date_column][
        data_new[date_column].astype(str).str.endswith("02-29")
    ] - pd.Timedelta(days=1)
    # combine multiple transactions that occur on same day
    data_new = (
        data_new.groupby(data_new[date_column].dt.to_period("D"))
        .apply(combine_multiple_per_day,date_column=date_column)
        .reset_index(drop=True)
    )

    data_new = dates.merge(data_new, on=[date_column], how="outer")
    return data_new
    
def combine_multiple_per_day(data):
    "combine transactions and balances that occur on the same day"
    date = data.date[0]
    account_id = data.account_id[0]
    amount = data.amount.sum()
    balance = data.balance.iloc[-1]
    status = str(data.status[0])
    day_new = pd.DataFrame(
        [[date, account_id, amount, balance, status]],
        columns=["date", "account_id", "amount", "balance", "status"],
    )
    return day_new


def correctly_name_columns(df, date_column="date", account_id_column="account_id", amount_column="amount",
                           balance_column="balance", status_column="status"):
    df.rename(columns={date_column: "date", account_id_column: "account_id", amount_column: "amount",
                           balance_column: "balance", status_column: "status"})
    return renamed_df



def fill_balance(data, balance_column='balance'):
    return


def fillBalancesAndTransactions(data, startDate, endDate, methodForFillingBalances, hours=24):
    "fill the dataset with respective values"
    dates = pd.DataFrame(pd.date_range(startDate, endDate, freq="D", name="date"))
    data_new = dates.merge(data, on="date", how="inner")

    # combine leap-year-day (29 feb) with 28 feb
    data_new.date[data_new.date.astype(str).str.endswith("02-29")] = data_new.date[
        data_new.date.astype(str).str.endswith("02-29")
    ] - pd.Timedelta(days=1)
    # combine multiple transactions that occur on same day
    data_new = (
        data_new.groupby(data_new.date.dt.to_period("D"))
        .apply(combineDays)
        .reset_index(drop=True)
    )

    dates = dates[~dates.date.astype(str).str.endswith("02-29")]  # drop 29th of feb
    data_new = dates.merge(data_new, on="date", how="outer")

    # fillna the dataframe
    data_new.balance = data_new.balance.fillna(
        method="ffill"
    )  # balance is forwardfilled
    if (
        methodForFillingBalances == "bfill"
    ):  # and afterwards depends on whether it's complete observation length
        data_new.balance = data_new.balance.fillna(method="bfill")
    elif methodForFillingBalances == 0:
        data_new.balance = data_new.balance.fillna(0)
    data_new.amount = data_new.amount.fillna(0)  # amount is filled with 0's
    data_new.account_id = data_new.account_id.fillna(
        int(data_new.account_id.mean())
    )  # account_id with account_id
    data_new.status = data_new.status.fillna(method="ffill")  # status is forwardfilled
    data_new.status = data_new.status.fillna(method="bfill")  # and afterwards backward
    return data_new









#%% EXTRA
    

#%%

def split_data_and_fill(data, date_column = time_interval="monthly", months=0, years=0):
    """"
    This function splits the data into time_intervals for a specified period.
    
    Args:
        data: pd.DataFrame()
        time_interval: string like "monthly", "yearly" or "overall"
        
    """"
        
    first_date = data[date_column].iloc[0]
    last_date = data[date_column].iloc[-1]
    if first_date < last_date - relativedelta(months=months,years=years):
        start_date = last_date.to_period("M").to_timestamp() - relativedelta(
            months=months - 1
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
    # features = dataset.groupby(dataset.date.dt.to_period('M')).apply(computeFeat, feature_type) # should process start/end date
    data_filled = fillBalancesAndTransactions(dataset, start_date, end_date, method_for_filling_balances)
    
    return data_filled
#%%
    
test = split_data_and_fill(dataset, "monthly", 12)
    
    
    
    
    
#%%


for functionName in ["splitMonthlyAndFill", "splitYearlyAndFill", "overallAndFill"]:
    
    exec("{functionName}(data,feature_type={feature_type}, observation_length={observation_length})".format(functionName=functionName, feature_type=feature_type, observation_length=observation_length))




































































































