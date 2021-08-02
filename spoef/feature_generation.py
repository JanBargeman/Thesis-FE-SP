import pandas as pd
from dateutil.relativedelta import relativedelta
import scipy.signal
import scipy.fft
import numpy as np
import pywt
import timeit
import os
os.chdir("/Users/Jan/Desktop/Thesis/Thesis-FE-SP")

from spoef.utils import (
    prepare_data_yearly,
    prepare_data_quarterly,
    combine_features_dfs,
)

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
    overall data. This is because "W" depends on len(data), which varies for overall.

    list_featuretypes:
        "Basic" - min, max, mean, kurt ,skew, std, sum.
        "FourierComplete" - all frequencies amplitudes and phases.
        "FourierNLargest" - n largest frequencies and their values.
        "WaveletComplete" - all approximation and details coefficients at each depth.
        "WaveletBasic" - takes "B"/Basic (min, max, etc) at each depth.

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
        
    allowed_components = ["Basic", "FourierNLargest", "WaveletComplete", "WaveletBasic", "FourierComplete"]

    for argument in list_featuretypes:
        if argument not in allowed_components:
            raise ValueError(f"argument must be one of {allowed_components}")
    
    features_basic = pd.DataFrame()
    features_fourier = pd.DataFrame()
    features_wavelet = pd.DataFrame()
    features_wavelet_basic = pd.DataFrame()
    features_fft2 = pd.DataFrame()
    if "Basic" in list_featuretypes:
        features_basic = compute_basic(data)
    if "FourierNLargest" in list_featuretypes:
        features_fourier = compute_fourier_n_largest(data, fourier_n_largest_frequencies)
    if "FourierComplete" in list_featuretypes:
        features_fft2 = compute_fourier_complete(data)
    if "WaveletComplete" in list_featuretypes:
        features_wavelet = compute_wavelet_complete(data, wavelet_depth, mother_wavelet)
    if "WaveletBasic" in list_featuretypes:
        features_wavelet_basic = compute_wavelet_basic(
            data, wavelet_depth, mother_wavelet
        )
    features = pd.concat(
        [features_basic, features_fourier, features_fft2, features_wavelet, features_wavelet_basic],
        axis=1,
    )
    return features


def compute_basic(data):
    """
    This function creates basic features.

    "Basic" - min, max, mean, kurt ,skew, std, sum.

    Args:
        data (pd.DataFrame()) : one column from which to make basic features.

    Returns:
        features (pd.DataFrame()) : (1 x 7) row of basic features.

    """
    col_names = ["min", "max", "mean", "skew", "kurt", "std", "sum"]
    col_names = ["Basic " + str(col) for col in col_names]
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


def compute_fourier_n_largest(data, fourier_n_largest_frequencies):
    """
    This function takes the Fast Fourier Transform and returns the n largest
    frequencies and their values.

    "FourierNLargest" - n largest frequencies and their values.

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
        "FourierNLargest freq_" + str(i + 1) + "/" + str(fourier_n_largest_frequencies)
        for i in range(int(len(features) / 2))
    ]
    col_names_size = [
        "FourierNLargest ampl_" + str(i + 1) + "/" + str(fourier_n_largest_frequencies)
        for i in range(int(len(features) / 2))
    ]
    col_names = [*col_names_index, *col_names_size]
    features = pd.DataFrame([features], columns=col_names)
    return features



def compute_fourier_complete(data):
    """
    This function takes the Fast Fourier Transform and returns the amplitudes 
    and phases of each frequency.

    "FourierComplete" - all frequencies amplitudes and phases.

    Args:
        data (pd.DataFrame()) : one column from which to make Fourier features.

    Returns:
        features (pd.DataFrame()) : (1 x (2 x len(data))) row of amplitudes and phases.

    """
    if (
        len(data) < 95 and len(data) > 85
    ):  # due to varying quarter lengths only first 88 days are used ...
        data = data[:89]
    
    # Fast Fourier Transform
    fft = scipy.fft.fft(data.values)
    fft_sel = fft[range(int(len(data) / 2))]
    
    fft_real = [abs(np.real(a)) for a in fft_sel]
    fft_imag = [np.imag(a) for a in fft_sel]
    


    # Name the columns
    features = [*fft_real, *fft_imag]
    col_names_real = [
        "FourierComplete ampl_" + str(i + 1) + "/" + str(len(data)/2)
        for i in range(int(len(features) / 2))
    ]
    col_names_imag = [
        "FourierComplete phase_" + str(i + 1) + "/" + str(len(data)/2)
        for i in range(int(len(features) / 2))
    ]
    col_names = [*col_names_real, *col_names_imag]
    features = pd.DataFrame([features], columns=col_names)
    return features





def compute_wavelet_complete(data, wavelet_depth, mother_wavelet):
    """
    This function takes the Wavelet Transform and returns all approximation
    and details coefficients at each depth.

    "WaveletComplete" - all approximation and details coefficients at each depth.

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
        len(data) < 95 and len(data) > 85
    ):  # due to varying quarter lengths only first 88 days are used ...
        data = data[:88]

    wavelet = pywt.wavedec(data, wavelet=mother_wavelet, level=wavelet_depth)
    
    features = [item for sublist in wavelet for item in sublist]  # flatten list

    col_names = [f'WaveletComplete depth_{wavelet_depth-sublist[0]}/{wavelet_depth}_item_{i+1}' for sublist in enumerate(wavelet) for i in range(len(sublist[1]))]  # flatten list
    
    features = pd.DataFrame([features], columns=col_names)
        
    return features


def compute_wavelet_basic(data, wavelet_depth, mother_wavelet):
    """
    This function takes the Wavelet Transform and at each depth makes basic
    features for the approximation and detail coefficients.

    "WaveletBasic" - takes "Basic" (min, max, etc) at each depth.

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
            "WaveletBasic_approx depth_" + str(i) + "_" + str(col)
            for col in features_at_depth.columns
        ]
        features_at_depth_high = compute_basic(pd.Series(coeffs))
        features_at_depth_high.columns = [
            "WaveletBasic_detail depth_" + str(i) + " " + str(col)
            for col in features_at_depth_high.columns
        ]
        features = pd.concat(
            [features, features_at_depth, features_at_depth_high], axis=1
        )
    return features


#%%
def create_all_features(data, list_featuretypes, mother_wavelet="db2", normalize=False, show_time=False):
    """
    PERSONAL FUNCTION, not part of open source:
    This function creates all features for transactions and balances.
    It also times how long it takes for the monthly, yearly and overall generation.
    It also checks whether there are any NaN's in the result and then combines
    it into one large dataframe.
    
    list_featuretypes:
        "Basic" - min, max, mean, kurt ,skew, std, sum.
        "FourierComplete" - all frequencies amplitudes and phases.
        "FourierNLargest" - n largest frequencies and their values.
        "WaveletComplete" - is NOT APPLICABLE for overall
        "WaveletBasic" - takes "B"/Basic (min, max, etc) at each depth.

    Args:
        data (pd.DataFrame()) : data for which to make features.
        list_featuretypes (list) : list of feature types to be computed.
        mother_wavelet (str) : type of wavelet used for the analysis.
            possible values: "db2", "db4", "haar", see pywt.wavelist(kind="discrete")
        normalize (bool) : normalize data in time window
        show_time (bool) : show benchmarking feature generation times
    Returns:
        features (pd.DataFrame()) : dataframe with all features for all identifiers.

    """
    list_featuretypes = list_featuretypes.copy()
    
    current = timeit.default_timer()  
    transaction_features_quarterly = feature_generation(
        data[["account_id", "date", "transaction"]],
        "account_id",
        "transaction",
        'quarter',
        normalize,
        list_featuretypes,
        mother_wavelet=mother_wavelet,
        observation_length=4,
    )
    balance_features_quarterly = feature_generation(
        data[["account_id", "date", "balance"]],
        "account_id",
        "balance",
        'quarter',
        normalize,
        list_featuretypes,
        mother_wavelet=mother_wavelet,
        observation_length=4,
    )
    if show_time:
        print("quarterly:", int(timeit.default_timer() - current), "seconds")
        current = timeit.default_timer()    
    
    transaction_features_yearly = feature_generation(
        data[["account_id", "date", "transaction"]],
        "account_id",
        "transaction",
        'year',
        normalize,
        list_featuretypes,
        fourier_n_largest_frequencies=30,
        wavelet_depth=6,
        mother_wavelet=mother_wavelet,
    )
    balance_features_yearly = feature_generation(
        data[["account_id", "date", "balance"]],
        "account_id",
        "balance",
        'year',
        normalize,
        list_featuretypes,
        fourier_n_largest_frequencies=30,
        wavelet_depth=6,
        mother_wavelet=mother_wavelet,
    )
    if show_time:
        print("yearly:", int(timeit.default_timer() - current), "seconds")

    list_features_dfs = [
        transaction_features_quarterly,
        balance_features_quarterly,
        transaction_features_yearly,
        balance_features_yearly,
    ]
    
    all_features = combine_features_dfs(list_features_dfs)
    
    if normalize == True:
        all_features.columns = ["norm " + col for col in all_features.columns]
    else:
        all_features.columns = ["reg " + col for col in all_features.columns]
    
    return all_features



def feature_generation(
    data,
    grouper,
    combine_fill_method,
    time_window,
    normalize=False,
    list_featuretypes=["Basic"],
    observation_length=1,
    fourier_n_largest_frequencies=10,
    wavelet_depth=4,
    mother_wavelet="db2",
):
    """
    This function splits the data per identifier and performs the monthly feature
    generation.

    list_featuretypes:
        "Basic" - min, max, mean, kurt ,skew, std, sum.
        "FourierComplete" - all frequencies amplitudes and phases.
        "FourierNLargest" - n largest frequencies and their values.
        "WaveletComplete" - all approximation and details coefficients at each depth.
        "WaveletBasic" - takes "B"/Basic (min, max, etc) at each depth.

    Args:
        data (pd.DataFrame()) : data from one identifier for which to make features.
        combine_fill_method (str) : 'balance' or 'transaction'.
        observation_length (int) : amount of recent months you want for the analysis.
        normalize (bool) : normalize data in time window
        list_featuretypes (list) : list of feature types to be computed.
        fourier_n_largest_frequencies (int) : amount of fourier features.
        wavelet_depth (int) : level of depth up to which the wavelet is computed.
            possible values: 3 is the max, depends on len(data)
        mother_wavelet (str) : type of wavelet used for the analysis.
            possible values: "db2", "db4", "haar", see pywt.wavelist(kind="discrete")
    Returns:
        features (pd.DataFrame()) : df with row of monthly features for each identifier.

    """
    if time_window == 'quarter':
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
    elif time_window == 'year':
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
    else:
        raise ValueError('Please enter one of ["quarter", "year"]')
    
    # Check if NaNs are generated
    na_list = []
    na_list.append(features.isna().sum().sum())
    if sum(na_list) != 0:
        raise ValueError("There are NaNs generated in the following columns:", na_list)
    
    return features


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
    It does this monthly for a specified length of the data. The feature generation
    can be tweaked through several variables.

    list_featuretypes:
        "Basic" - min, max, mean, kurt ,skew, std, sum.
        "FourierComplete" - all frequencies amplitudes and phases.
        "FourierNLargest" - n largest frequencies and their values.
        "WaveletComplete" - all approximation and details coefficients at each depth.
        "WaveletBasic" - takes "B"/Basic (min, max, etc) at each depth.

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
        normalize (bool) : normalize data in time window

    Returns:
        features (pd.DataFrame()) : row of quarterly features for one identifier.

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
    It does this yearly for a specified length of the data. The feature generation
    can be tweaked through several variables.

    list_featuretypes:
        "Basic" - min, max, mean, kurt ,skew, std, sum.
        "FourierComplete" - all frequencies amplitudes and phases.
        "FourierNLargest" - n largest frequencies and their values.
        "WaveletComplete" - all approximation and details coefficients at each depth.
        "WaveletBasic" - takes "B"/Basic (min, max, etc) at each depth.

    Args:
        data (pd.DataFrame()) : data from one identifier for which to make features.
        combine_fill_method (str) : 'balance' or 'transaction'.
        list_featuretypes (list) : list of feature types to be computed.
        observation_length (int) : amount of recent months you want for the analysis.
        fourier_n_largest_frequencies (int) : amount of fourier features.
        wavelet_depth (int) : level of depth up to which the wavelet is computed.
            possible values: 6 is the max, depends on len(used_data)
        mother_wavelet (str) : type of wavelet used for the analysis.
            possible values: "db2", "db4", "haar", see pywt.wavelist(kind="discrete")
        normalize (bool) : normalize data in time window

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
