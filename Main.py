import os
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
from datetime import datetime
import time
import timeit
import scipy.signal #.stft #.argrelmax  for finding max in plots or smth
import scipy.fft
import numpy as np
import matplotlib.pyplot as plt
import pywt
import matplotlib.pyplot as plotter
from functools import reduce
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

os.chdir('/Users/Jan/Desktop/Thesis/Thesis-FE-SP')

# scipy.detrending removal of (changing) mean
# scipy signal butter bandpass filter



#%% data import
loan = pd.read_csv('0data/loan.txt', delimiter = ";")
trans = pd.read_csv('0data/trans.txt', delimiter = ";")

#%% data preprocess
dataset = loan[["account_id", "status"]].merge(trans[["account_id", "date", "type", "amount", "balance"]])

dataset.loc[dataset.type == "VYDAJ", 'amount'] = -dataset.loc[dataset.type == "VYDAJ", 'amount']
dataset = dataset.drop(columns=["type"])
dataset.date = pd.to_datetime(dataset.date, format='%y%m%d')
dataset = dataset[["account_id", "date", "amount", "balance", "status"]]
dataset_orig = dataset.copy()

#%% make test dataset
# dataset = dataset.iloc[0:2000,:]
# dataset = dataset[dataset.account_id == 1787]
dataset = dataset[dataset.account_id == 276]


#%% general functions

def computeFeat(data, featType, ft_nfeat, wt_depth):
    "channels the 'featType' such that desired features are computed"    
    if featType == "SP_all": 
        features = computeSP_all(data, ft_nfeat, wt_depth)
    elif featType == "basic": 
        features = computeBasic(data)          
    elif featType == "SP_FT": 
        features = computeSP_FT(data, ft_nfeat)
    elif featType == "SP_WT1": 
        features = computeSP_WT1(data, wt_depth)
    elif featType == "SP_WT2": 
        features = computeSP_WT2(data, wt_depth)
    elif featType == "SP_FTWT2": 
        features = computeSP_FTWT2(data, ft_nfeat, wt_depth)  
    else:
        raise ValueError('featType must contain one of "FT", "WT1", "WT2", or "SP_all" or "basic"')
    return features

def combineDays(data):
    "combine transactions and balances that occur on the same day"
    date = data.date[0]
    account_id = data.account_id[0]
    amount = data.amount.sum()
    balance = data.balance.iloc[-1]
    status = str(data.status[0])
    day_new = pd.DataFrame([[date,account_id,amount,balance,status]], columns=['date','account_id','amount','balance','status'])  
    return day_new

def fillBalTran(data, startDate, endDate, fillBalMeth, hours = 24):
    "fill the dataset with respective values"
    dates = pd.DataFrame(pd.date_range(startDate, endDate, freq='D', name="date"))    
    data_new = dates.merge(data, on="date", how="inner")
    
    # combine leap-year-day (29 feb) with 28 feb
    data_new.date[data_new.date.astype(str).str.endswith('02-29')] = data_new.date[data_new.date.astype(str).str.endswith('02-29')]-pd.Timedelta(days=1)
    # combine multiple transactions that occur on same day
    data_new = data_new.groupby(data_new.date.dt.to_period('D')).apply(combineDays).reset_index(drop=True)

    dates = dates[~dates.date.astype(str).str.endswith('02-29')]    # drop 29th of feb
    data_new = dates.merge(data_new, on="date", how="outer")    
    
    # fillna the dataframe
    data_new.balance = data_new.balance.fillna(method='ffill')  # balance is forwardfilled 
    if fillBalMeth == 'bfill':  # and afterwards depends on whether it's complete observation length
        data_new.balance = data_new.balance.fillna(method='bfill') 
    elif fillBalMeth == 0:
        data_new.balance = data_new.balance.fillna(0)
    data_new.amount = data_new.amount.fillna(0) # amount is filled with 0's
    data_new.account_id = data_new.account_id.fillna(int(data_new.account_id.mean())) # account_id with account_id
    data_new.status = data_new.status.fillna(method='ffill') # status is forwardfilled
    data_new.status = data_new.status.fillna(method='bfill') # and afterwards backward  
    return data_new

#%% compute features on specific time basis
    
def monthly(dataset, featType, obsLen = 12, ft_nfeat = 10, wt_depth = 3):
    "grabs 'obsLen' months of the data, fills it and computes 'featType' features per month"
    firstDate = dataset.date.iloc[0]
    lastDate = dataset.date.iloc[-1]
    if firstDate < lastDate - relativedelta(months = obsLen):
        startDate = lastDate.to_period('M').to_timestamp() - relativedelta(months = obsLen-1) #dropping first month of year due to varying month lengths
        endDate = lastDate.to_period('M').to_timestamp() + relativedelta(months=1) - relativedelta(days=1)
        fillBalMeth = 'bfill'
    else:
        print("not full obs length:", dataset.account_id.iloc[0])
        startDate = firstDate.to_period('M').to_timestamp() + relativedelta(months=1) #dropping first month of year due to varying month lengths
        endDate = lastDate.to_period('M').to_timestamp() + relativedelta(months=1) - relativedelta(days=1)
        fillBalMeth = 0
    
    # select only start_date <-> end_date for later use groupby
    # features = dataset.groupby(dataset.date.dt.to_period('M')).apply(computeFeat, featType) # should process start/end date 
    dataset = fillBalTran(dataset, startDate, endDate, fillBalMeth)

    monthlyFeatures = pd.DataFrame()
    for month in range(0, obsLen):
        dataMonth = dataset[(dataset.date >= startDate + relativedelta(months=month)) & (dataset.date < startDate + relativedelta(months=month+1))]
        features = pd.DataFrame(computeFeat(dataMonth, featType, ft_nfeat, wt_depth)).T
        monthlyFeatures = pd.concat([monthlyFeatures,features], axis = 1)        
    
    monthlyFeatures.columns = np.arange(len(monthlyFeatures.columns))    
    
    return monthlyFeatures

def yearly(dataset, featType, obsLen = 2, ft_nfeat = 20, wt_depth = 6):
    "grabs 'obsLen' years of the data, fills it and computes 'featType' features per year"
    firstDate = dataset.date.iloc[0]
    lastDate = dataset.date.iloc[-1]
    if firstDate < lastDate - relativedelta(years = obsLen):
        startDate = lastDate.to_period('M').to_timestamp() - relativedelta(years = obsLen) #dropping first month of year due to varying month lengths
        endDate = (lastDate.to_period('M')).to_timestamp() + relativedelta(years=1) - relativedelta(days=1)
        fillBalMeth = 'bfill'
    else:
        # print("suggested monthly:", dataset.account_id.iloc[0])
        startDate = firstDate.to_period('M').to_timestamp()
        endDate = (lastDate.to_period('M')).to_timestamp() + relativedelta(years=1) - relativedelta(days=1)
        fillBalMeth = 0
    
    # select only start_date <-> end_date for later use groupby
    # features = dataset.groupby(dataset.date.dt.to_period('M')).apply(computeFeat, featType) # should process start/end date     
    dataset = fillBalTran(dataset, startDate, endDate, fillBalMeth)
 
    yearlyFeatures = pd.DataFrame()
    for year in range(0, obsLen):
        dataYear = dataset[(dataset.date >= startDate + relativedelta(years=year)) & (dataset.date < startDate + relativedelta(years=year+1))]       
        features = pd.DataFrame(computeFeat(dataYear, featType, ft_nfeat, wt_depth)).T
        yearlyFeatures = pd.concat([yearlyFeatures,features], axis = 1)    

    return yearlyFeatures

def overall(dataset, featType, ft_nfeat = 20, wt_depth = 6):
    "grabs the entire dataset, fills it and computes 'featType' features over the entire length"
    firstDate = dataset.date.iloc[0]
    lastDate = dataset.date.iloc[-1]
    startDate = firstDate.to_period('M').to_timestamp()    
    endDate = lastDate.to_period('M').to_timestamp()
    fillBalMeth = 0
    
    datasetOverall = fillBalTran(dataset, startDate, endDate, fillBalMeth)
 
    overallFeatures = pd.DataFrame(computeFeat(datasetOverall, featType, ft_nfeat, wt_depth)).T

    return overallFeatures

#%% feature engineering (basic)

def computeBasic(data):
    "computes the basic features for the 'amount' column and 'balakce' column"
    featuresAmount = computeBasicFeat(data['amount'])
    featuresBalance = computeBasicFeat(data['balance'])
    features = [*featuresAmount, *featuresBalance]
    return features

def computeBasicFeat(data):  
    "compute basic features of input data: min max avg skw krt std"
    # fsum = dataMonth.amount.sum()    
    features = [data.min(), data.max(), data.mean(), data.skew(), data.kurt(), data.std()]   
    return features

#%% feature engineering (SP)

def computeSP_all(data, ft_nfeat, wt_depth):
    "compute signal processing features, FT, WT1 and WT2"
    fourierAmount = computeFourier(data['amount'], ft_nfeat)
    fourierBalance = computeFourier(data['balance'], ft_nfeat)    
    waveletAmount = computeWavelet(data['amount'], wt_depth)
    waveletBalance = computeWavelet(data['balance'], wt_depth)
    waveletAmountB = computeWaveletB(data['amount'], wt_depth)
    waveletBalanceB = computeWaveletB(data['balance'], wt_depth)
    features = [*fourierAmount, *fourierBalance, *waveletAmount, *waveletBalance, *waveletAmountB, *waveletBalanceB]   
    return features

def computeSP_FT(data, ft_nfeat):
    "compute signal processing features, FT"
    fourierAmount = computeFourier(data['amount'], ft_nfeat)
    fourierBalance = computeFourier(data['balance'], ft_nfeat)
    features = [*fourierAmount, *fourierBalance]
    return features

def computeSP_FTWT2(data, ft_nfeat, wt_depth):
    "compute signal processing features, FT and WT2"
    fourierAmount = computeFourier(data['amount'], ft_nfeat)
    fourierBalance = computeFourier(data['balance'], ft_nfeat)
    waveletAmountB = computeWaveletB(data['amount'], wt_depth)
    waveletBalanceB = computeWaveletB(data['balance'], wt_depth)
    features = [*fourierAmount, *fourierBalance, *waveletAmountB, *waveletBalanceB]   
    return features

def computeSP_WT1(data, wt_depth):
    "compute signal processing features, WT1"
    waveletAmount = computeWavelet(data['amount'], wt_depth)
    waveletBalance = computeWavelet(data['balance'], wt_depth)
    features = [*waveletAmount, *waveletBalance]   
    return features

def computeSP_WT2(data, wt_depth):
    "compute signal processing features, WT2"
    waveletAmountB = computeWaveletB(data['amount'], wt_depth)
    waveletBalanceB = computeWaveletB(data['balance'], wt_depth)
    features = [*waveletAmountB, *waveletBalanceB]   
    return features

def computeFourier(data, ft_nfeat):
    "compute fourier transform of data and return 10 largest frequencies and amounts"
    fft = scipy.fft.fft(data.values)
    fft_abs = abs(fft[range(int(len(data)/2))])
        
    largestIndexes = np.argsort(-fft_abs)[:ft_nfeat]
    largestValues = fft_abs[largestIndexes]
    largestValues = [int(a) for a in largestValues]
    
    features = [largestIndexes.tolist(), largestValues]
    features = [item for sublist in features for item in sublist]   # flatten list      
    return features

def computeWavelet(data, depth):
    "compute wavelet transform of data and return detail coefficients at each decomposition level"
    wavelet = pywt.wavedec(data, 'db2', level=depth)
    features = [item for sublist in wavelet for item in sublist]   # flatten list      
    return features

def computeWaveletB(data, depth):
    features = []
    for i in range(depth-1):
        data, coeffs = pywt.dwt(data, 'db2')
        featuresAtDepth = computeBasicFeat(pd.Series(data))
        features = [*features, *featuresAtDepth]     
    return features

    

#%% filters, denoisers

def waveletDenoise(data, thresh = 0.63, wavelet='db2'):
    "removing noisy high frequencies from the input data by applying a wavelet threshold"
    thresh = thresh*np.nanmax(data.iloc[:,1].values)
    coeff = pywt.wavedec(data.iloc[:,1].values, wavelet, mode="periodization")
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft") for i in coeff[1:])
    data.iloc[:,1] = pywt.waverec(coeff, wavelet, mode="periodization")       
    return data
# data = datasetOverall[['date','amount']]
# test = waveletDenoise(data, 0.002) #doet het niet echt


# figure, axis = plotter.subplots(2, 1)
# plotter.subplots_adjust(hspace=1) 
# axis[0].plot(data.date, test.amount)
# axis[1].plot(data.date, data.amount)


def countna(listofdfs):
    for item in listofdfs:
        print(item.isna().sum().sum())


#%% MAIN

current = timeit.default_timer()

monthlyFeaturesB = dataset.groupby("account_id").apply(monthly, featType="basic", obsLen=12).reset_index(level=1,drop=True)
print(timeit.default_timer() - current)     # 2 min
current = timeit.default_timer()

monthlyFeaturesSP = dataset.groupby("account_id").apply(monthly, featType="SP_all", obsLen=12).reset_index(level=1,drop=True)
print(timeit.default_timer() - current)     # FT 2 min // +WT1   // +WT2 2.5 min
current = timeit.default_timer()

yearlyFeaturesB = dataset.groupby("account_id").apply(yearly, featType="basic", obsLen=2).reset_index(level=1,drop=True)
print(timeit.default_timer() - current)     # 3 min
current = timeit.default_timer()

yearlyFeaturesSP = dataset.groupby("account_id").apply(yearly, featType="SP_all", obsLen=2).reset_index(level=1,drop=True)
print(timeit.default_timer() - current)     # FT 3.5 min // +WT1   // +WT2 3.5
current = timeit.default_timer()

overallFeaturesB = dataset.groupby("account_id").apply(overall, featType="basic").reset_index(level=1,drop=True)
print(timeit.default_timer() - current)     # 5 min
current = timeit.default_timer()

overallFeaturesSP = dataset.groupby("account_id").apply(overall, featType="SP_FTWT2").reset_index(level=1,drop=True)
print(timeit.default_timer() - current)     # FT 5 min // +WT1 0   // +WT2
current = timeit.default_timer()

countna([monthlyFeaturesB, monthlyFeaturesSP, yearlyFeaturesB, yearlyFeaturesSP, overallFeaturesB, overallFeaturesSP])
    
#%%     Writing all files to a 0results folder (if it doesn't exist, manually create please)

# my_list = ["monthlyFeaturesB", "monthlyFeaturesSP", "yearlyFeaturesB", "yearlyFeaturesSP", "overallFeaturesB", "overallFeaturesSP"]
# for item in my_list:
#     eval("%s" %item).to_csv("0results/%s.csv" %item)
#     print("Writing %s" %item)

#%%
monthlyFeaturesB = pd.read_csv('0results/monthlyFeaturesB.csv',index_col='account_id')
monthlyFeaturesSP = pd.read_csv('0results/monthlyFeaturesSP.csv',index_col='account_id')
yearlyFeaturesB = pd.read_csv('0results/yearlyFeaturesB.csv',index_col='account_id')
yearlyFeaturesSP = pd.read_csv('0results/yearlyFeaturesSP.csv',index_col='account_id')
overallFeaturesB = pd.read_csv('0results/overallFeaturesB.csv',index_col='account_id')
overallFeaturesSP = pd.read_csv('0results/overallFeaturesSP.csv',index_col='account_id')


#%% combine SP/basic with status and train/test split

def combine(list_of_dfs):
    combined_dfs = reduce(lambda  left,right: pd.merge(left,right,left_index=True, right_index=True, how='inner'), list_of_dfs)
    return combined_dfs

status = dataset_orig[['account_id','status']].drop_duplicates().set_index('account_id')

l_data_all = [status, monthlyFeaturesB, monthlyFeaturesSP, yearlyFeaturesB, yearlyFeaturesSP, overallFeaturesB, overallFeaturesSP]
l_data_B = [status, monthlyFeaturesB, yearlyFeaturesB, overallFeaturesB]
l_data_SP = [status, monthlyFeaturesSP, yearlyFeaturesSP, overallFeaturesSP]
data_all = combine(l_data_all)
data_B = combine(l_data_B)
data_SP = combine(l_data_SP)

# df_scaled = fScaling(df_total)      # Scaling: StandardScaler     -- Deze is beter
Y_all = data_all.iloc[:,0].values
X_all = data_all.iloc[:,1:].values
Y_B = data_B.iloc[:,0].values
X_B = data_B.iloc[:,1:].values
Y_SP = data_SP.iloc[:,0].values
X_SP = data_SP.iloc[:,1:].values


t_size = 0.4

X_train_all, X_valid_all, y_train_all, y_valid_all = train_test_split(X_all,Y_all , test_size=t_size, random_state=0)
X_train_B, X_valid_B, y_train_B, y_valid_B = train_test_split(X_B,Y_B , test_size=t_size, random_state=0)
X_train_SP, X_valid_SP, y_train_SP, y_valid_SP = train_test_split(X_SP,Y_SP , test_size=t_size, random_state=0)




#%% train models

n_trees = 300
max_depth = None

clf_all = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth)
clf_all.fit(X_train_all, y_train_all)
print("\nall:\n", confusion_matrix(y_valid_all, clf_all.predict(X_valid_all)))
print("acc:", accuracy_score(y_valid_all, clf_all.predict(X_valid_all)))

clf_B = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth)
clf_B.fit(X_train_B, y_train_B)
print("\nB:\n", confusion_matrix(y_valid_B, clf_B.predict(X_valid_B)))
print("acc:", accuracy_score(y_valid_B, clf_B.predict(X_valid_B)))


clf_SP = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth)
clf_SP.fit(X_train_SP, y_train_SP)
print("\nSP:\n", confusion_matrix(y_valid_SP, clf_SP.predict(X_valid_SP)))
print("acc:", accuracy_score(y_valid_SP, clf_SP.predict(X_valid_SP)))



#%% compare models



#%%
# problems:

#   verschillende lengtes van observaties, dus 4 maanden of 12 maanden
#   in monthly komen soms verschillende maanden bij elkaar in de monthlyfeaturesSP kolom

# TODO:
#   columnnames
#   fft power spectrum?
#   computational complexity scaling (On^2?)
#   smaller functions + some comments
#   try logistic regression/ligthGBMregressorclassfiier


#%%











plt.hist(data_all.iloc[:,0])







#%% meh
data1 = pd.read_csv('0data/PS_20174392719_1491204439457_log.csv')
data2 = pd.read_excel('0data/bank.xlsx')

# %% sample pwyt

x = [3, 7, 1, 1, -2, 5, 4, 6]
cA, cD = pywt.dwt(x, 'db2')
print(pywt.idwt(cA, cD, 'db2'))

cA, cD = pywt.wavedec(x, 'db2', level=4)




#%% WAVELET TEST

# datatest = dataset[dataset.account_id==1787]
# firstDate = datatest.date.iloc[0]
# lastDate = datatest.date.iloc[-1]
# startDate = firstDate.to_period('M').to_timestamp()    
# endDate = lastDate.to_period('M').to_timestamp()
# fillBalMeth = 0
# datatest = fillBalTran(datatest, startDate, endDate, fillBalMeth)

# month = 30
# dataMonth = datatest[(datatest.date >= startDate + relativedelta(months=month)) & (datatest.date < startDate + relativedelta(months=month+1))]


# #%%

# data_for_analysis = dataMonth.amount
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

# axis[0].plot(datatest.date, datatest.amount)
# axis[1].plot(datatest.date, wtrec)




































































































