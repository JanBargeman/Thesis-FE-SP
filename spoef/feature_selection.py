import pandas as pd
import os
os.chdir("/Users/Jan/Desktop/Thesis/Thesis-FE-SP")




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
def asses_5x2cv(dataset1, dataset2, model):
    return 


def assess_McNemar(dataset1, dataset2, model):
    return


def feat_importance_RF(dataset, model):
    return

def feat_importance_LGBM(dataset, model):
    return

