import pandas as pd
from dateutil.relativedelta import relativedelta
from functools import reduce
import os

os.chdir("/Users/Jan/Desktop/Thesis/Thesis-FE-SP")

#%%
def preprocess_data(loan, trans):

    dataset = loan[["account_id", "status"]].merge(
        trans[["account_id", "date", "type", "amount", "balance"]]
    )
    
    dataset.loc[dataset.type == "VYDAJ", "amount"] = -dataset.loc[
        dataset.type == "VYDAJ", "amount"
    ]
    dataset = dataset.drop(columns=["type"])
    dataset.date = pd.to_datetime(dataset.date, format="%y%m%d")
    dataset = dataset[["account_id", "date", "amount", "balance", "status"]]    
    dataset.columns = ["account_id", "date", "transaction", "balance", "status"]  

    return dataset


def shap_elim_to_reduce(start_data, shap_elim, num_features):
    set_of_feats = shap_elim.get_reduced_features_set(num_features=num_features)
    return get_reduced_data(start_data, set_of_feats)



def select_non_default_subset_data(data, multiplier):
    data_sel = data[['account_id','status']].drop_duplicates()
       
    data_def = data_sel[data_sel.status==1]
    data_non_def = data_sel[data_sel.status==0]
    data_subset_non_def = data_non_def.sample(n=multiplier*len(data_def), random_state=multiplier, axis=0)
    data_subset = pd.concat([data_def, data_subset_non_def], axis=0)
    data_subset = data_subset.merge(data, on=['account_id', 'status'], how='inner')
    return data_subset


def get_reduced_data(data, set_of_feats):
    data = data.copy()
    if 'status' not in set_of_feats:
        set_of_feats.insert(0,'status')
    return data.loc[:,set_of_feats]

# deze
def count_occurences_features(pd_feat_names, print_head=None):
    if isinstance(pd_feat_names, pd.DataFrame()):
        pd_feat_names = pd_feat_names.columns
    pd_split = pd.Series(pd_feat_names).str.split(" ", expand=True)    
    col_names = ['transformation', 'transaction/balance/transform', 'time_window', 'feature_type', 'detail 1', 'detail 2']    
    dataframe = pd.DataFrame()
    
    for i in range(len(pd_split.columns)):
        count = pd_split[pd_split.columns[i]].value_counts()
        count = count.reset_index()
        count = count.iloc[:,0] + ": " + count.iloc[:,1].astype(str)
        count.name = col_names[i]           
        dataframe = pd.concat([dataframe,count],axis=1)
    
    if print_head != None:    
        pd.set_option('display.max_columns', None)
        print(dataframe.head(print_head))
        pd.reset_option('display.max_columns')

    return dataframe


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


