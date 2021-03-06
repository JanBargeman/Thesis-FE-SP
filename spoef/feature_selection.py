from sklearn.metrics import roc_auc_score, auc
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold
import statistics
import math
import scipy.stats
from sklearn.model_selection import train_test_split
import os
from probatus.feature_elimination import ShapRFECV
import matplotlib.pyplot as plt
import numpy as np
from numpy import interp

from numpy.random import seed

os.chdir("/Users/Jan/Desktop/Thesis/Thesis-FE-SP")


#%%

def shap_fs(data, classifier, step=0.2, cv=5, scoring='roc_auc', n_iter=5):
    
    shap_elim = ShapRFECV(classifier, step=step, cv=cv, scoring=scoring, n_jobs=1)
    
    y = data.iloc[:, 0]
    X = data.iloc[:, 1:]
    
    shap_elim.fit_compute(X,y, check_additivity=False)
    
    return shap_elim

def assess_5x2cv(dataset1, dataset2, model1, model2, results_location=None, filename=None, color1='#ff7f0e', color2='#2279b5'):
    
    mean1, stdev1, base_fpr1, mean_tprs1, mean_auc1, std_auc1, tprs_lower1, tprs_upper1 = perform_5x2cv(dataset1, model1)
    mean2, stdev2, base_fpr2, mean_tprs2, mean_auc2, std_auc2, tprs_lower2, tprs_upper2 = perform_5x2cv(dataset2, model2)

    plt.figure(figsize=(12, 8))
    plt.plot(base_fpr1, mean_tprs1, color1, alpha = 1, label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc1, std_auc1),)
    plt.plot(base_fpr2, mean_tprs2, color2, alpha = 1, label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc2, std_auc2),)
    plt.fill_between(base_fpr1, tprs_lower1, tprs_upper1, color = color1, alpha = 0.3)
    plt.fill_between(base_fpr2, tprs_lower2, tprs_upper2, color = color2, alpha = 0.3)
    plt.plot([0, 1], [0, 1], linestyle = '--', lw = 2, color = 'black', label = 'Luck', alpha= 0.8)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.title('Receiver operating characteristic (ROC) curve')
    #plt.axes().set_aspect('equal', 'datalim')
    
    if filename != None:
        plt.savefig(f"{results_location}/figures/{filename}",dpi=200)
        
    plt.show()

    
    print(f"\nFirst: {mean1} ({stdev1})")
    print(f"Second: {mean2} ({stdev2})")
    
    try:
        Z = (mean1 - mean2) / math.sqrt(stdev1*stdev1/math.sqrt(5) + stdev2*stdev2/math.sqrt(5))
    except:
        print("Performance equal, div by zero error.")
        return
    
    print("Z-value: " + "%.4f" %(Z))
    
    if Z > -2 and Z < 2:
        print("The performance is equal.")
    elif Z >= 2:
        print("The first set performs significantly better.")
    elif Z <= -2:
        print("The second set performs significantly better.")
    
    print("p-value: " + "%.6f" %(scipy.stats.norm.sf(abs(Z))*2))
    
    return 

#%%
# assess_5x2cv(fs_data_B_LGBM_final, fs_data_SP_LGBM_final, lgbm_B, lgbm_SP) 

#%%
def perform_5x2cv(data, model):

    X = data.iloc[:, 1:].values
    Y = data.iloc[:, 0].values
    
    AUC_score_avg_list = []
    
    base_fpr = np.linspace(0, 1, 101)
    tprs = []
    aucs = []
    
    seed(0)
    
    for i in range(5):
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=i)
        auc_list = []
        for train_index, test_index in skf.split(X,Y):
            X_train, X_valid = X[train_index], X[test_index]
            y_train, y_valid = Y[train_index], Y[test_index]
            model.fit(X_train, y_train)
            
            auc_list.append(roc_auc_score(
                y_valid, model.predict_proba(X_valid)[:,1]
            ))
            
            # stacked ROC curves
            fpr, tpr, threshold = metrics.roc_curve(y_valid, model.predict_proba(X_valid)[:,1])
            roc_auc = metrics.auc(fpr, tpr)
            aucs.append(roc_auc)
            tpr = interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)
            
        AUC_score_avg_list.append(sum(auc_list)/len(auc_list))
        # print("avg", AUC_score_avg_list)
    
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    
    mean_auc = auc(base_fpr, mean_tprs)
    std_auc = np.std(aucs)
    
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    
    mean = float("%.4f" %statistics.mean(AUC_score_avg_list))
    stdev = float("%.6f" %statistics.stdev(AUC_score_avg_list))
    
    # print("Mean (std): " + str(mean) + " (" + str(stdev) + ")")
    
    return mean, stdev, base_fpr, mean_tprs, mean_auc, std_auc, tprs_lower, tprs_upper

#%%
def assess_McNemar(dataset1, dataset2, model1, model2, test_size=0.4):
    
    Y1 = dataset1.iloc[:, 0].values
    X1 = dataset1.iloc[:, 1:].values

    X_train1, X_valid1, y_train1, y_valid1 = train_test_split(
        X1, Y1, test_size=test_size, random_state=0
    )

    Y2 = dataset2.iloc[:, 0].values
    X2 = dataset2.iloc[:, 1:].values

    X_train2, X_valid2, y_train2, y_valid2 = train_test_split(
        X2, Y2, test_size=test_size, random_state=0
    )
    
    # model1.fit(X_train1, y_train1)
    pred1 = model1.predict(X_valid1)
    
    # model2.fit(X_train2, y_train2)
    pred2 = model2.predict(X_valid2)
    
    a,b,c,d = calc_contingency_table(y_valid1, pred1, pred2)
    
    try: 
        McNemar = (b-c)*(b-c) / (b+c)
    except:
        print("\nDivision by zero, the data sets perform equally well\n")
        return
    
    print('\n')
    if (b+c) < 25:
        if b > c:
            print("The first set performs better. Evaluated with binomial distribution.")
            print("p-value: " + str("%.4f" %scipy.stats.binom.cdf(c, b+c, 0.5)))
        elif b < c:
            print("The second set performs better. Evaluated with binomial distribution.")
            print("p-value: " + str("%.4f" %scipy.stats.binom.cdf(b, b+c, 0.5)))
        else:
            print("The sets perform equally well. Evaluated with binomial distribution.")
            
    else:    
        if McNemar < 3.841:
            print("The data sets perform equally well.", McNemar, "< 3.841")
        else:
            if b > c:
                print("The first set performs significantly better.")
                print("p-value: " + str("%.4f" %(1-scipy.stats.chi2.cdf(McNemar,1))))    
            elif b < c:
                print("The second set performs significantly better.")
                print("p-value: " + str("%.4f" %(1-scipy.stats.chi2.cdf(McNemar,1))))    
    return         
        
def calc_contingency_table(valid, pred1, pred2):
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range(len(pred1)):
        if valid[i] == 1 and pred1[i] == 1 and pred2[i] == 1:
            a = a + 1
        elif valid[i] == 0 and pred1[i] == 0 and pred2[i] == 0:
            a = a + 1
        elif valid[i] == 1 and pred1[i] == 1 and pred2[i] == 0:
            b = b + 1
        elif valid[i] == 0 and pred1[i] == 0 and pred2[i] == 1:
            b = b + 1
        elif valid[i] == 1 and pred1[i] == 0 and pred2[i] == 1:
            c = c + 1
        elif valid[i] == 0 and pred1[i] == 1 and pred2[i] == 0:
            c = c + 1
        elif valid[i] == 1 and pred1[i] == 0 and pred2[i] == 0:
            d = d + 1
        elif valid[i] == 0 and pred1[i] == 1 and pred2[i] == 1:
            d = d + 1
        else:
            raise ValueError("Contingency table error.")

    return a,b,c,d   



def return_without_column_types(data, string_list, index_list):
    data = data.copy()
    for string, index in zip(string_list, index_list):
        data = data[["status"]+[col for col in data.columns[1:] if not col.split(" ")[index].startswith(string)]]
    return data

def return_with_only_column_types(data, string_list, index_list):
    data = data.copy()
    for string, index in zip(string_list, index_list):
        data = data[["status"]+[col for col in data.columns[1:] if col.split(" ")[index].startswith(string)]]
    return data