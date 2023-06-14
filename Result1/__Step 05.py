#!/usr/bin/env python
# coding: utf-8





###### STEP 5.1: RANDOM FOREST analysis without oversampling and selecting hospitals to use

ALLPARAM=['CD34','CD20','CD10','CD19','CD45','CD13','CD33','cyCD3','cyMPO','CD22','IGM','CD38','cyTDT','CD3','CD66','CD58']
ALLPARAM.sort()
listacomun=ALLPARAM
######
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import confusion_matrix
import os

sourcerelapse='/Users/salvador/Documents/shapecancerrelapse/RIPS DATA (from Step 3-4 and 6-7)/All combinations (Step 3-4)/RelapseAnalysis'
sourcenonrelapse='/Users/salvador/Documents/shapecancerrelapse/RIPS DATA (from Step 3-4 and 6-7)/All combinations (Step 3-4)/NonRelapseAnalysis'


listparam=os.listdir(sourcerelapse)
listparam.sort()
listparam=[x for x in listparam if all([y in listacomun for y in x.split('-')])]
###
NORMED=False
###
names=listparam
names.sort()
df=pd.DataFrame(columns=['AUC','TPR','TNR','PPV','NPV','FPR','FNR','FDR','ACC'],index=names)
for l in range(0,len(listparam)):
    if listparam[l][0]!='.':
        print('\n----------')
        print(listparam[l])
        listpac=os.listdir(sourcerelapse+'/'+listparam[l])
        listpac.sort()
        for j in range(0,len(listpac)):
            if listpac[j][0]!='.':
                if listpac[j].split('.')[-2][-4:]!='norm':
                    
                    datarelapse=pd.read_csv(sourcerelapse+'/'+listparam[l]+'/'+listpac[j])
                    datanonrelapse=pd.read_csv(sourcenonrelapse+'/'+listparam[l]+'/'+listpac[j])
                    ##Possibilities of including or excluding hospital data
                    # Delete comments if needed
                    #deletehosp1='HMU'
                    #deletehosp2='HVR'
                    #datarelapse=datarelapse[~datarelapse.iloc[:,0].astype(str).str.startswith(deletehosp1)]
                    #datarelapse=datarelapse.reset_index(drop=True)
                    #datarelapse=datarelapse[~datarelapse.iloc[:,0].astype(str).str.startswith(deletehosp2)]
                    #datarelapse=datarelapse.reset_index(drop=True)
                    #datanonrelapse=datanonrelapse[~datanonrelapse.iloc[:,0].astype(str).str.startswith(deletehosp1)]
                    #datanonrelapse=datanonrelapse.reset_index(drop=True)
                    #datanonrelapse=datanonrelapse[~datanonrelapse.iloc[:,0].astype(str).str.startswith(deletehosp2)]
                    #datanonrelapse=datanonrelapse.reset_index(drop=True)

                    datanonrelapse['Relapse']=0
                    datarelapse['Relapse']=1
                    maxit=10
                    fpr=[]
                    tpr=[]
                    thresholds=[]
                    rocscore=[]
                    ooberror=[]
                    featuresimp=[]
                    confmatrices=[]
                    predsgeneral=[]
                    testsgeneral=[]


                    ######
                    TN=np.zeros(maxit)
                    TN[:]=np.nan
                    TP=np.zeros(maxit)
                    TP[:]=np.nan
                    FP=np.zeros(maxit)
                    FP[:]=np.nan
                    FN=np.zeros(maxit)
                    FN[:]=np.nan
                    #
                    TPR=np.zeros(maxit)
                    TPR[:]=np.nan
                    TNR=np.zeros(maxit)
                    TNR[:]=np.nan
                    PPV=np.zeros(maxit)
                    PPV[:]=np.nan
                    NPV=np.zeros(maxit)
                    NPV[:]=np.nan
                    FPR=np.zeros(maxit)
                    FPR[:]=np.nan
                    FNR=np.zeros(maxit)
                    FNR[:]=np.nan
                    FDR=np.zeros(maxit)
                    FDR[:]=np.nan
                    ACC=np.zeros(maxit)
                    ACC[:]=np.nan

                    ######


                    for r in range(0,maxit):
                     
                        samples=np.array(datarelapse.sample(frac=0.6).index)
                        datarelapse['is_train']=0
                        datarelapse.loc[samples,'is_train']=1
                        samples=np.array(datanonrelapse.sample(frac=0.6).index)
                        datanonrelapse['is_train']=0
                        datanonrelapse.loc[samples,'is_train']=1
                        RANDOM_STATE=123
                        ensemble_clfs = [
                            ("RandomForestClassifier, max_features=None",
                                RandomForestClassifier(warm_start=True, max_features=None,
                                                       oob_score=True,
                                                       random_state=RANDOM_STATE))
                        ]
                        min_estimators = 15
                        max_estimators = 175
                        trainrelapse,testrelapse  = datarelapse[datarelapse['is_train']==True],datarelapse[datarelapse['is_train']==False]
                        trainnonrelapse,testnonrelapse  = datanonrelapse[datanonrelapse['is_train']==True],datanonrelapse[datanonrelapse['is_train']==False]
                        train=pd.concat([trainnonrelapse,trainrelapse]).reset_index(drop=True)
                        test=pd.concat([testnonrelapse,testrelapse]).reset_index(drop=True)
                        features=datarelapse.columns[2:len(datarelapse.columns)-2]
                        y=train['Relapse']
                        names=['Nonrelapse','Relapse']
                        error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

                        # Range of number of estimator values
                        min_estimators = 20
                        max_estimators = 100

                        #for label, clf in ensemble_clfs:
                        #    for i in range(min_estimators, max_estimators + 1):
                        #        clf.set_params(n_estimators=i)
                        #        clf.fit(train[features], y)
#
                        #        # Record the OOB error
                        #        oob_error = 1 - clf.oob_score_
                        #        #error_rate[label].append((i, oob_error))
                        clf=RandomForestClassifier(oob_score=True,
                                                       random_state=RANDOM_STATE)
                        clf.set_params(n_estimators=100,max_depth=100)
                        clf.fit(train[features], y)
                    
                        preds=np.array(names)[clf.predict(test[features])]
                        preds[preds=='Nonrelapse']=0
                        preds[preds=='Relapse']=1

                        confusionmatrix=pd.crosstab(test['Relapse'], preds, rownames=['Relapsed patients?'], colnames=['Predicted Diagnosis'])

                        confmatrices.append(confusionmatrix.values)
                        feataux=list(zip(train[features], clf.feature_importances_))
                        featuresimp.append(feataux)
                        probas_ = clf.predict_proba(test[features])
                        fpraux, tpraux, thresholdsaux = roc_curve(test['Relapse'], 1-probas_[:, 0])
                        if sum(test['Relapse'].ravel())!=0:
                            rocscoreaux=roc_auc_score(test['Relapse'], 1-probas_[:, 0])
                            rocscore.append(rocscoreaux)
                        predsgeneral.append(list(map(int,preds)))
                        testsgeneral.append(test['Relapse'].values)
                        fpr.append(fpraux)
                        tpr.append(tpraux)

                        thresholds.append(thresholdsaux)
                        
                        #ooberror.append(np.mean(ys))
                        #print('%s'%r)

                        TN[r], FP[r], FN[r], TP[r]=confusion_matrix(testsgeneral[r],predsgeneral[r]).ravel()
                        # Sensitivity, hit rate, recall, or true positive rate
                        TPR[r] = TP[r]/(TP[r]+FN[r])
                        # Specificity or true negative rate
                        TNR[r] = TN[r]/(TN[r]+FP[r])
                        # Precision or positive predictive value
                        if (TP[r]+FP[r])!= 0:
                            PPV[r] = TP[r]/(TP[r]+FP[r])
                        else:
                            PPV[r] = 0
                        # Negative predictive value
                        NPV[r] = TN[r]/(TN[r]+FN[r])
                        # Fall out or false positive rate
                        FPR[r] = FP[r]/(FP[r]+TN[r])
                        # False negative rate
                        FNR[r] = FN[r]/(TP[r]+FN[r])
                        # False discovery rate
                        if (TP[r]+FP[r])!= 0:
                            FDR[r] = FP[r]/(TP[r]+FP[r])
                        else:
                            FDR[r] =0
                        # Overall accuracy
                        ACC[r] = (TP[r]+TN[r])/(TP[r]+FP[r]+FN[r]+TN[r])
                    df.iloc[l]=[round(item,2) for item in list(map(np.nanmean,[rocscore,TPR,TNR,PPV,NPV,FPR,FNR,FDR,ACC]))]
                    print('Done. Performed '+str(l+1)+'/'+str(len(listparam)))

df.to_csv('/'.join(sourcerelapse.split('/')[0:-1])+'/'+'Analysis_Simple.csv')


###### STEP 5.2: RANDOM FOREST analysis with oversampling


import os
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import confusion_matrix
import os
import numpy as np
from sklearn.model_selection import train_test_split


from sklearn.model_selection import RepeatedStratifiedKFold






sourcerelapse='/Users/salvador/Documents/shapecancerrelapse/RIPS DATA (from Step 3-4 and 6-7)/All combinations (Step 3-4)/RelapseAnalysis'
sourcenonrelapse='/Users/salvador/Documents/shapecancerrelapse/RIPS DATA (from Step 3-4 and 6-7)/All combinations (Step 3-4)/NonRelapseAnalysis'


ALLPARAM=['CD34','CD20','CD10','CD19','CD45','CD13','CD33','cyCD3','cyMPO','CD22','IGM','CD38','cyTDT','CD3','CD66','CD58']
ALLPARAM.sort()
listacomun=ALLPARAM
listparam=os.listdir(sourcerelapse)
listparam.sort()
listparam=[x for x in listparam if all([y in listacomun for y in x.split('-')])]
NORMED=False
names=listparam
names.sort()
df=pd.DataFrame(columns=['AUC','OOB','TPR','TNR','PPV','NPV','FPR','FNR','FDR','ACC'],index=names)
for l in range(0,len(listparam)):#len(listparam)#For each parameter combination do the following analysis
    if listparam[l][0]!='.':
        print('\n----------')
        print(listparam[l])
        listpac=os.listdir(sourcerelapse+'/'+listparam[l])
        listpac.sort()
        for j in range(0,len(listpac)):
            if listpac[j][0]!='.':
                if listpac[j].split('.')[-2][-4:]!='norm':
                    datarelapse=pd.read_csv(sourcerelapse+'/'+listparam[l]+'/'+listpac[j])
                    datanonrelapse=pd.read_csv(sourcenonrelapse+'/'+listparam[l]+'/'+listpac[j])
                    datanonrelapse['Relapse']=0
                    datarelapse['Relapse']=1
                    data=pd.concat([datarelapse,datanonrelapse]).reset_index(drop=True)
                    features=data.columns[2:len(data.columns)-1]
                    fpr=[]
                    tpr=[]
                    thresholds=[]
                    rocscore=[]
                    ooberror=[]
                    featuresimp=[]
                    confmatrices=[]
                    predsgeneral=[]
                    testsgeneral=[]



                    ######
                    RANDOM_STATE=123
                    ros = RandomOverSampler(random_state=l)
                    X_resampled, y_resampled = ros.fit_resample(data, data['Relapse'])
                    rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=10,random_state=j)
                    maxit=rskf.get_n_splits(X_resampled,y_resampled)
                    
                    ######
                    TN=np.zeros(maxit)
                    TN[:]=np.nan
                    TP=np.zeros(maxit)
                    TP[:]=np.nan
                    FP=np.zeros(maxit)
                    FP[:]=np.nan
                    FN=np.zeros(maxit)
                    FN[:]=np.nan
                    #
                    TPR=np.zeros(maxit)
                    TPR[:]=np.nan
                    TNR=np.zeros(maxit)
                    TNR[:]=np.nan
                    PPV=np.zeros(maxit)
                    PPV[:]=np.nan
                    NPV=np.zeros(maxit)
                    NPV[:]=np.nan
                    FPR=np.zeros(maxit)
                    FPR[:]=np.nan
                    FNR=np.zeros(maxit)
                    FNR[:]=np.nan
                    FDR=np.zeros(maxit)
                    FDR[:]=np.nan
                    ACC=np.zeros(maxit)
                    ACC[:]=np.nan
                    
                    
                    
                    for r, (train_index, test_index) in enumerate(rskf.split(X_resampled,y_resampled)):
                        
                        # Comment these lines to use OVERSAMPLING or not
                        ros = RandomOverSampler(random_state=r)
                        X_resampled, y_resampled = ros.fit_resample(data, data['Relapse'])
                        train=X_resampled.iloc[train_index].reset_index(drop=True)
                        test=X_resampled.iloc[test_index].reset_index(drop=True)
                        test=X_resampled.iloc[test_index]
                        
                        X_train=train[features]
                        y_train=train["Relapse"]
                        X_test=test[features]
                        y_test=test["Relapse"]
                        
                        clf=RandomForestClassifier(max_depth=3,oob_score=True,random_state=RANDOM_STATE)
                        clf.set_params(n_estimators=20)
                        clf.fit(X_train,y_train)
                        preds=clf.predict(X_test[features])

                        confusionmatrix=pd.crosstab(y_test, preds, rownames=['Relapsed patients?'], colnames=['Predicted Diagnosis'])

                        confmatrices.append(confusionmatrix.values)
                        feataux=list(zip(X_train[features], clf.feature_importances_))
                        featuresimp.append(feataux)
                        probas_ = clf.predict_proba(X_test[features])
                        fpraux, tpraux, thresholdsaux = roc_curve(y_test, 1-probas_[:, 0])
                        if sum(y_test.ravel())!=0:
                            rocscoreaux=roc_auc_score(y_test, 1-probas_[:, 0])
                            rocscore.append(rocscoreaux)
                        predsgeneral.append(list(map(int,preds)))
                        testsgeneral.append(y_test.values)
                        fpr.append(fpraux)
                        tpr.append(tpraux)

                        thresholds.append(thresholdsaux)
                        oob_error = 1 - clf.oob_score_
                        
                        ooberror.append(oob_error)
                        #print('%s'%r)

                        TN[r], FP[r], FN[r], TP[r]=confusion_matrix(testsgeneral[r],predsgeneral[r]).ravel()
                        # Sensitivity, hit rate, recall, or true positive rate
                        TPR[r] = TP[r]/(TP[r]+FN[r])
                        # Specificity or true negative rate
                        TNR[r] = TN[r]/(TN[r]+FP[r])
                        # Precision or positive predictive value
                        PPV[r] = TP[r]/(TP[r]+FP[r])
                        # Negative predictive value
                        NPV[r] = TN[r]/(TN[r]+FN[r])
                        # Fall out or false positive rate
                        FPR[r] = FP[r]/(FP[r]+TN[r])
                        # False negative rate
                        FNR[r] = FN[r]/(TP[r]+FN[r])
                        # False discovery rate
                        FDR[r] = FP[r]/(TP[r]+FP[r])
                        # Overall accuracy
                        ACC[r] = (TP[r]+TN[r])/(TP[r]+FP[r]+FN[r]+TN[r])
                    df.iloc[l]=[round(item,2) for item in list(map(np.nanmean,[rocscore,ooberror,TPR,TNR,PPV,NPV,FPR,FNR,FDR,ACC]))]
                    print('Done. Performed '+str(l+1)+'/'+str(len(listparam)))

df.to_csv('/'.join(sourcerelapse.split('/')[0:-1])+'/'+'Analysis_Oversmapling.csv')



#### FOR PLOTTING RESULTS

pd.read_csv('/'.join(sourcerelapse.split('/')[0:-1])+'/'+'Analysis.csv') # Or Analysis_Oversampling

aucmean=[[]]*len(ALLPARAM)
for i in range(0,len(ALLPARAM)):
    aucaux=np.zeros(len(ALLPARAM)-1)
    k=0
    for j in range(0,len(df)):
        lista=df.iloc[j,0].split('-')
        if ALLPARAM[i] in lista and lista[1-lista.index(ALLPARAM[i])] in ALLPARAM:
            aucaux[k]=df.iloc[j,1]
            k=k+1
    aucmean[i]=[aucaux]
accmean=[[]]*len(ALLPARAM)
for i in range(0,len(ALLPARAM)):
    accaux=np.zeros(len(ALLPARAM)-1)
    k=0
    for j in range(0,len(df)):
        lista=df.iloc[j,0].split('-')
        if ALLPARAM[i] in lista and lista[1-lista.index(ALLPARAM[i])] in ALLPARAM:
            accaux[k]=df.iloc[j,-1]
            k=k+1
    accmean[i]=[accaux]
    
aucs=np.array(list(map(np.mean,aucmean)))
plt.figure(figsize=(15, 3),dpi=600)
plt.bar(ALLPARAM,aucs)
plt.axhline(y=0.5, color='r', linestyle=':')
plt.ylim([0.5,1])
plt.xticks(rotation=45,size=20)
plt.yticks(size=20)
