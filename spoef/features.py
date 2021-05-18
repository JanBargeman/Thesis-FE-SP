import pandas as pd
from dateutil.relativedelta import relativedelta
import scipy.signal  # .stft #.argrelmax  for finding max in plots or smth
import scipy.fft
import numpy as np
import pywt
import timeit

from spoef.utils import (
    prepare_data_monthly,
    prepare_data_yearly,
    prepare_data_overall,
    count_na,
    combine_features_dfs,
)

#%% For testing functions

# data = pd.read_csv("personal/data/data.csv")
# data.date = pd.to_datetime(data.date, format="%y%m%d")

# #%% make test data
# data = data.iloc[0:2000,:]
# data_acc = data[data.account_id == 1787]
# data_used = data_acc[["date","balance"]]
# # data = data[data.account_id == 276]
# # data = data[data.account_id == 1843]


#%%


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
    overall data.

    list_featuretypes:
        "B" for Basic - min, max, mean, kurt ,skew, std.
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
    features_basic = pd.DataFrame()
    features_fourier = pd.DataFrame()
    features_wavelet = pd.DataFrame()
    features_wavelet_basic = pd.DataFrame()
    if "B" in list_featuretypes:
        features_basic = compute_basic(data)
    if "F" in list_featuretypes:
        features_fourier = compute_fourier(data, fourier_n_largest_frequencies)
    if "W" in list_featuretypes:
        features_wavelet = compute_wavelet(data, wavelet_depth, mother_wavelet)
    if "W_B" in list_featuretypes:
        features_wavelet_basic = compute_wavelet_basic(
            data, wavelet_depth, mother_wavelet
        )
    features = pd.concat(
        [features_basic, features_fourier, features_wavelet, features_wavelet_basic],
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

    Returns:
        features (pd.DataFrame()) : (1 x 2n) row of largest frequencies and values.

    """
    # Fast Fourier Transform
    fft = scipy.fft.fft(data.values)
    fft_abs = abs(fft[range(int(len(data) / 2))])

    # Select largest indexes (=frequencies) and their values
    larges_indexes = np.argsort(-fft_abs)[:fourier_n_largest_frequencies]
    largest_values = fft_abs[larges_indexes]
    largest_values = [int(a) for a in largest_values]

    # Name the columns
    features = [*larges_indexes.tolist(), *largest_values]
    col_names_index = [
        "fft index " + str(i + 1) + "/" + str(fourier_n_largest_frequencies)
        for i in range(int(len(features) / 2))
    ]
    col_names_size = [
        "fft size " + str(i + 1) + "/" + str(fourier_n_largest_frequencies)
        for i in range(int(len(features) / 2))
    ]
    col_names = [*col_names_index, *col_names_size]
    features = pd.DataFrame([features], columns=col_names)
    return features


def compute_basic(data):
    """
    This function creates basic features.

    "B" for Basic - min, max, mean, kurt ,skew, std.

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
        mother_wavelet (str) : type of wavelet used for the analysis.

    Returns:
        features (pd.DataFrame()) : row of wavelet features.

    """
    if (
        len(data) < 35 and len(data) > 23
    ):  # due to varying month lengths only first 28 days are used ...
        data = data[:28]

    wavelet = pywt.wavedec(data, wavelet=mother_wavelet, level=wavelet_depth)
    features = [item for sublist in wavelet for item in sublist]  # flatten list

    col_names = ["wavelet " + str(i + 1) for i in range(len(features))]
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
        mother_wavelet (str) : type of wavelet used for the analysis.

    Returns:
        features (pd.DataFrame()) : (2 x 7 x wavelet_depth) row of wavelet features.

    """
    data_wavelet = data
    features = pd.DataFrame()
    for i in range(wavelet_depth):
        data_wavelet, coeffs = pywt.dwt(data_wavelet, wavelet=mother_wavelet)
        features_at_depth = compute_basic(pd.Series(data_wavelet))
        features_at_depth.columns = [
            "wav_B depth " + str(i + 1) + " " + str(col)
            for col in features_at_depth.columns
        ]
        features_at_depth_high = compute_basic(pd.Series(coeffs))
        features_at_depth_high.columns = [
            "wav_B_high depth " + str(i + 1) + " " + str(col)
            for col in features_at_depth_high.columns
        ]
        features = pd.concat(
            [features, features_at_depth, features_at_depth_high], axis=1
        )
    return features


#%%


def compute_features_monthly(
    data,
    combine_fill_method,
    list_featuretypes,
    observation_length,
    fourier_n_largest_frequencies,
    wavelet_depth,
    mother_wavelet,
):
    """
    This function computes different types of features for one identifier.
    It does this monthly for a specified length of the data. The feature creation
    can be tweaked through several variables.

    list_featuretypes:
        "B" for Basic - min, max, mean, kurt ,skew, std.
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
        mother_wavelet (str) : type of wavelet used for the analysis.

    Returns:
        features (pd.DataFrame()) : row of monthly features for one identifier.

    """
    # drop identifier column
    data = data.drop(data.columns[0], axis=1)

    # select only relevant period and fill the empty date
    prepared_data = prepare_data_monthly(data, combine_fill_method, observation_length)

    start_date = prepared_data.iloc[0, 0]

    # create features per month
    features = pd.DataFrame()
    for month in range(0, observation_length):
        data_month = prepared_data[
            (prepared_data.iloc[:, 0] >= start_date + relativedelta(months=month))
            & (prepared_data.iloc[:, 0] < start_date + relativedelta(months=month + 1))
        ]
        monthly_features = compute_list_featuretypes(
            data_month.iloc[:, 1],
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


def compute_features_yearly(
    data,
    combine_fill_method,
    list_featuretypes,
    observation_length,
    fourier_n_largest_frequencies,
    wavelet_depth,
    mother_wavelet,
):
    """
    This function computes different types of features for one identifier.
    It does this yearly for a specified length of the data. The feature creation
    can be tweaked through several variables.

    list_featuretypes:
        "B" for Basic - min, max, mean, kurt ,skew, std.
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
        mother_wavelet (str) : type of wavelet used for the analysis.

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
        yearly_features = compute_list_featuretypes(
            data_year.iloc[:, 1],
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


def compute_features_overall(
    data,
    combine_fill_method,
    list_featuretypes,
    fourier_n_largest_frequencies,
    wavelet_depth,
    mother_wavelet,
):
    """
    This function computes different types of features for one identifier.
    It does this for the entire length of the data. The feature creation
    can be tweaked through several variables.

    Please note that "W" is not applicable for overall feature creation. It will
    be automatically removed from the list_featuretypes.

    list_featuretypes:
        "B" for Basic - min, max, mean, kurt ,skew, std.
        "F" for Fourier - n largest frequencies and their values.
        "W" for Wavelet - is NOT APPLICABLE for overall
        "W_B" for Wavelet Basic - takes "B"/Basic (min, max, etc) at each depth.

    Args:
        data (pd.DataFrame()) : data from one identifier for which to make features.
        combine_fill_method (str) : 'balance' or 'transaction'.
        list_featuretypes (list) : list of feature types to be computed.
        fourier_n_largest_frequencies (int) : amount of fourier features.
        wavelet_depth (int) : level of depth up to which the wavelet is computed.
        mother_wavelet (str) : type of wavelet used for the analysis.

    Returns:
        features (pd.DataFrame()) : row of overall features for one identifier.

    """
    # drop identifier column
    data = data.drop(data.columns[0], axis=1)

    if "W" in list_featuretypes:  # W does not work on overall data
        list_featuretypes.remove("W")

    # select only relevant period and fill the empty date
    prepared_data = prepare_data_overall(data, combine_fill_method)

    # create features overall
    features = compute_list_featuretypes(
        prepared_data.iloc[:, 1],
        list_featuretypes,
        fourier_n_largest_frequencies,
        wavelet_depth,
        mother_wavelet,
    )

    # name columns
    features.columns = [data.columns[1][:2] + " O " + col for col in features.columns]
    return features


def create_all_features(data, list_featuretypes, mother_wavelet="db2"):
    """
    PERSONAL FUNCTION, not part of open source:
    This function creates all features for transactions and balances.
    It also times how long it takes for the monthly, yearly and overall creation.
    It also checks whether there are any NaN's in the result and then combines
    it into one large dataframe.
    
    list_featuretypes:
    "B" for Basic - min, max, mean, kurt ,skew, std.
    "F" for Fourier - n largest frequencies and their values.
    "W" for Wavelet - is NOT APPLICABLE for overall
    "W_B" for Wavelet Basic - takes "B"/Basic (min, max, etc) at each depth.

    Args:
        data (pd.DataFrame()) : data for which to make features.
        list_featuretypes (list) : list of feature types to be computed.
        mother_wavelet (str) : type of wavelet used for the analysis.

    Returns:
        features (pd.DataFrame()) : dataframe with all features for all identifiers.

    """

    current = timeit.default_timer()
    transaction_features_monthly = feature_creation_monthly(
        data[["account_id", "date", "transaction"]],
        "account_id",
        "transaction",
        list_featuretypes,
        mother_wavelet=mother_wavelet,
    )
    balance_features_monthly = feature_creation_monthly(
        data[["account_id", "date", "balance"]],
        "account_id",
        "balance",
        list_featuretypes,
        mother_wavelet=mother_wavelet,
    )
    print("monthly:", int(timeit.default_timer() - current), "seconds")
    current = timeit.default_timer()  # 527

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
    print("yearly:", int(timeit.default_timer() - current), "seconds")
    current = timeit.default_timer()  # 228

    transaction_features_overall = feature_creation_overall(
        data[["account_id", "date", "transaction"]],
        "account_id",
        "transaction",
        list_featuretypes,
        mother_wavelet=mother_wavelet,
    )
    balance_features_overall = feature_creation_overall(
        data[["account_id", "date", "balance"]],
        "account_id",
        "balance",
        list_featuretypes,
        mother_wavelet=mother_wavelet,
    )
    print("overall:", int(timeit.default_timer() - current), "seconds")  # 533

    list_features_dfs = [
        transaction_features_monthly,
        balance_features_monthly,
        transaction_features_yearly,
        balance_features_yearly,
        transaction_features_overall,
        balance_features_overall,
    ]
    count_na(list_features_dfs)

    all_features = combine_features_dfs(list_features_dfs)

    return all_features


#%%


def feature_creation_monthly(
    data,
    grouper,
    combine_fill_method,
    list_featuretypes=["B"],
    observation_length=12,
    fourier_n_largest_frequencies=10,
    wavelet_depth=3,
    mother_wavelet="db2",
):
    """
    This function splits the data per identifier and performs the monthly feature
    creation.

    list_featuretypes:
        "B" for Basic - min, max, mean, kurt ,skew, std.
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
        mother_wavelet (str) : type of wavelet used for the analysis.
    Returns:
        features (pd.DataFrame()) : df with row of monthly features for each identifier.

    """
    features = (
        data.groupby(grouper)
        .apply(
            compute_features_monthly,
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


def feature_creation_yearly(
    data,
    grouper,
    combine_fill_method,
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
        "B" for Basic - min, max, mean, kurt ,skew, std.
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
        mother_wavelet (str) : type of wavelet used for the analysis.

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
        )
        .reset_index(level=1, drop=True)
    )
    return features


def feature_creation_overall(
    data,
    grouper,
    combine_fill_method,
    list_featuretypes=["B"],
    fourier_n_largest_frequencies=50,
    wavelet_depth=6,
    mother_wavelet="db2",
):
    """
    This function splits the data per identifier and performs the overall feature
    creation.

    Please note that "W" is not applicable for overall feature creation. It will
    be automatically removed from the list_featuretypes.

    list_featuretypes:
        "B" for Basic - min, max, mean, kurt ,skew, std.
        "F" for Fourier - n largest frequencies and their values.
        "W" for Wavelet - is NOT APPLICABLE for overall
        "W_B" for Wavelet Basic - takes "B"/Basic (min, max, etc) at each depth.

    Args:
        data (pd.DataFrame()) : data from one identifier for which to make features.
        combine_fill_method (str) : 'balance' or 'transaction'.
        list_featuretypes (list) : list of feature types to be computed.
        fourier_n_largest_frequencies (int) : amount of fourier features.
        wavelet_depth (int) : level of depth up to which the wavelet is computed.
        mother_wavelet (str) : type of wavelet used for the analysis.

    Returns:
        features (pd.DataFrame()) : df with row of overall features for each identifier.

    """
    features = (
        data.groupby(grouper)
        .apply(
            compute_features_overall,
            combine_fill_method=combine_fill_method,
            list_featuretypes=list_featuretypes,
            fourier_n_largest_frequencies=fourier_n_largest_frequencies,
            wavelet_depth=wavelet_depth,
            mother_wavelet=mother_wavelet,
        )
        .reset_index(level=1, drop=True)
    )
    return features
