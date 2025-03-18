from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

from complexity import Complexity
import numpy as np
from os import listdir
from os.path import isfile, join


from imblearn.under_sampling import RandomUnderSampler, OneSidedSelection,RepeatedEditedNearestNeighbours
from imblearn.over_sampling import SMOTE,BorderlineSMOTE
from imblearn.combine import SMOTEENN,SMOTETomek


import pandas as pd
import smote_variants


def oversample(method,classifier,X_train,y_train,X_test,y_test,f1_average):
        sm = method() 
        X_res, y_res = sm.fit_resample(X_train, y_train)
        classifier.fit(X_res, y_res)
        y_prob = classifier.predict_proba(X_test)
        y_pred = (y_prob[:,1] > 0.5).astype(int)        
        f1 = f1_score(y_test, y_pred, zero_division=0,average=f1_average,pos_label=1)
        return f1


#Load files with the datasets
folder = "../dataset/alg_sel/"
onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
onlyfiles.sort(reverse=True)


#Measure the complexity
onb_dic = {}
dataset_dic = {}
for file in onlyfiles:
    complexity = Complexity(folder+file,distance_func="default",file_type="arff")
    onb_val = complexity.ONB(imb=True)[1]
    onb_dic[file] = onb_val
    dataset_dic[file] = [complexity.X, complexity.y]


df = pd.DataFrame(onb_dic.items(), columns=['dataset', 'ONB'])

#Select Classifier
f1_average = 'binary'
knn = KNeighborsClassifier(n_neighbors=5)


#Select the values with high and low complexity
difs_avg = []
good_bad_vals_df = df[(df['ONB']>0.7) | (df['ONB']<0.3)]


#Balance the datasets - run N times
max_versions = 10
for dataset in good_bad_vals_df['dataset']:
    
    smote_dif = []
    rus_dif = []
    reen_dif = []
    enn_dif = []
    border_dif = []
    gp_dif = []
    mw_dif = []
    
    ONB_val = good_bad_vals_df[good_bad_vals_df['dataset']==dataset]['ONB'].iloc[0]

    for version in range(max_versions):
        
        X = dataset_dic[dataset][0]
        y = dataset_dic[dataset][1]
        
        X_train, X_test, y_train, y_test = train_test_split(X,y , random_state=104, test_size=0.25,shuffle=True,stratify=y) 
        

        knn.fit(X_train, y_train)
        y_prob = knn.predict_proba(X_test)
        y_pred = (y_prob[:,1] > 0.5).astype(int)        
        f1_orig = f1_score(y_test, y_pred, zero_division=0,average=f1_average,pos_label=1)

        f1_smote = oversample(SMOTE,knn,X_train,y_train,X_test,y_test,f1_average)
        smote_dif.append(f1_smote - f1_orig)

        f1_rus = oversample(RandomUnderSampler,knn,X_train,y_train,X_test,y_test,f1_average)
        rus_dif.append(f1_rus - f1_orig)
        
        f1_reen = oversample(RepeatedEditedNearestNeighbours,knn,X_train,y_train,X_test,y_test,f1_average)
        reen_dif.append(f1_reen - f1_orig)

        f1_smoteenn = oversample(SMOTEENN,knn,X_train,y_train,X_test,y_test,f1_average)
        enn_dif.append(f1_smoteenn - f1_orig)
    
        f1_smoteborder = oversample(BorderlineSMOTE,knn,X_train,y_train,X_test,y_test,f1_average)
        border_dif.append(f1_smoteborder - f1_orig)

        f1_graph = oversample(smote_variants.SL_graph_SMOTE,knn,X_train,y_train,X_test,y_test,f1_average)
        gp_dif.append(f1_graph - f1_orig)




        f1_mw = oversample(smote_variants.MWMOTE,knn,X_train,y_train,X_test,y_test,f1_average)
        mw_dif.append(f1_mw - f1_orig)

    #Average the value for every run
    difs_avg.append([ONB_val,np.mean(smote_dif),np.mean(rus_dif),np.mean(reen_dif),np.mean(enn_dif),np.mean(border_dif),np.mean(gp_dif),np.mean(mw_dif)])



difs_df = pd.DataFrame(np.array(difs_avg),columns=['ONB','SMOTE','RUS','REEN','EEN','Borderline','GRAPH','MWMOTE'])

#Divide low and high complexity datasets
low_metric_df = difs_df[difs_df['ONB']<0.3]
high_metric_df = difs_df[difs_df['ONB']>0.7]

print(low_metric_df[['SMOTE','RUS','REEN','EEN','Borderline','GRAPH','MWMOTE']].mean())
print(high_metric_df[['SMOTE','RUS','REEN','EEN','Borderline','GRAPH','MWMOTE']].mean())
