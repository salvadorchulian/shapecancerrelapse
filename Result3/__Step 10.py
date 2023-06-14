#!/usr/bin/env python
# coding: utf-8




############### STEP 10: Methods (SVM, LR) for classification of PIs ###############



import os 
from persim import PersImage
import matplotlib.pyplot as plt

import time as time
from imblearn.over_sampling import RandomOverSampler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import svm


base='/home/HospitalXLandmarks_CD10203845RIPS/'

dimensions=[0,1,2]
pixel=[5,10,25,50,100]
spreads=[0.01,0.05]

tablaLR=[[]]*len(dimensions)
tablaSVM=[[]]*len(dimensions)
scorelist=[]
stdlist=[]
auclist=[]
f1list=[]
recalllist=[]
precisionlist=[]



for nitem in range(0,len(dimensions)):
    DIMENSION=dimensions[nitem]
    tablaLR[nitem]=[[] for i in range(len(pixel))]
    tablaSVM[nitem]=[[] for i in range(len(pixel))]
    
    for npix in range(0,len(pixel)):
        pix=pixel[npix]
        pixels=[pix,pix]
        pixI=pixel.index(pix)
        tablaLR[nitem][pixI]=[[] for i in range(len(spreads))]
        tablaSVM[nitem][pixI]=[[] for i in range(len(spreads))]
        
        for nspread in range(0,len(spreads)):
            spread=spreads[nspread]
            imgs_array=[]
            sprI=spreads.index(spread)
            tablaLR[nitem][pixI][sprI]=[[] for i in range(3)]
            tablaSVM[nitem][pixI][sprI]=[[] for i in range(8)]
            
            timeINI=time.time()
            folder='PersistenceImages'+str(DIMENSION)
            dirR=base+'/'+folder+'/'+str(pixels[0])+'_'+str(spread)+'/'+'Relapse'
            dirNR=base+'/'+folder+'/'+str(pixels[0])+'_'+str(spread)+'/'+'NonRelapse'

            numberR=len(os.listdir(dirR))
            numberNR=len(os.listdir(dirNR))

            listdirR=os.listdir(dirR)
            listdirR.sort()
            listdirR=[x for x in listdirR if not x.startswith('.')]
            listdirNR=os.listdir(dirNR)
            listdirNR.sort()
            listdirNR=[x for x in listdirNR if not x.startswith('.')]
            ImgsR=[[]]*numberR
            ImgsNR=[[]]*numberNR
            print(dirR)
            for i in range(0,numberR):
                ImgsR[i]=np.loadtxt(dirR+'/'+listdirR[i],delimiter=' ')
            for i in range(0,numberNR):
                ImgsNR[i]=np.loadtxt(dirNR+'/'+listdirNR[i],delimiter=' ')
            
            imgs_array=[img.flatten() for img in np.concatenate([ImgsNR,ImgsR])]
            labels=np.concatenate([np.zeros(len(ImgsNR)),np.ones(len(ImgsR))])
            print(len(imgs_array[0]))
            #### We can consider or not oversamplig by means of this comments
            ros = RandomOverSampler(random_state=0)
            X_resampled, y_resampled = ros.fit_resample(imgs_array, labels)
            imgs_array=X_resampled
            labels=y_resampled
            
            
            print('###########################')
            print('DIMENSION')
            print(DIMENSION)
            
            print('Pixels')
            print(pix)
            
            print('Spread')
            print(spread)
            C,gamma=[10,0.001]
            classifier = svm.SVC(gamma=gamma,C=C,kernel='linear',random_state=None)
            
            # If LR wants to be used, use this code
            # classifier= LogisticRegression(solver='lbfgs',max_iter=100)
            

            CV=6;

            scores = cross_val_score(classifier,imgs_array, labels, cv=CV)
            #metrics.roc_auc_score(y_test,y_test_pred)
            #metrics.recall_score(y_test,y_pred,zero_division=0)
            #metrics.f1_score(y_test,y_pred,zero_division=0)
            #metrics.precision_score(y_test,y_pred,zero_division=0)
            predictions = cross_val_predict(classifier,imgs_array, labels, cv=CV)
            accuracy = metrics.r2_score(labels, predictions)
            print("Score "+str(np.mean(scores)))
            print("Accuracy "+str(accuracy))

            tablaSVM[nitem][pixI][sprI][0]=np.mean(cross_val_score(classifier,imgs_array, labels, cv=CV,scoring='roc_auc'))
            tablaSVM[nitem][pixI][sprI][1]=scores.mean()
            tablaSVM[nitem][pixI][sprI][2]=scores.std()
            tablaSVM[nitem][pixI][sprI][3]=metrics.confusion_matrix(labels, predictions)
            tablaSVM[nitem][pixI][sprI][4]=[C,gamma]
            tablaSVM[nitem][pixI][sprI][5]=np.mean(cross_val_score(classifier,imgs_array, labels, cv=CV,scoring='recall'))
            tablaSVM[nitem][pixI][sprI][6]=np.mean(cross_val_score(classifier,imgs_array, labels, cv=CV,scoring='f1'))
            tablaSVM[nitem][pixI][sprI][7]=np.mean(cross_val_score(classifier,imgs_array, labels, cv=CV,scoring='precision'))
            
            #accuracylist.append(accuracy)
            #scorelist.append(np.mean(scores))
            print(len(imgs_array[0]))
            print('Done in '+str(np.round(time.time()-timeINI,3))+' secs.')
            
import pandas as pd
print('SVM K-FOLD')
df=[]
for j in range(0,len(dimensions)):
    DIMENSION=dimensions[j]
    print('DIMENSION '+str(DIMENSION))
    for i in range(0,len(pixel)):
        
        print('--Pixels '+str(pixel[i]))
        dfaux=pd.DataFrame(tablaSVM[j][i],spreads,['AUC','Accuracy','Std','Conf_matrix','C, gamma','Recall','f1','precision'])
        dfaux['Dimension']=DIMENSION
        dfaux['Pixels']=pixel[i]
        if len(df)==0:
            df=dfaux
        else:
            df=pd.concat([df,dfaux])
        print(df)
    print('##########')
