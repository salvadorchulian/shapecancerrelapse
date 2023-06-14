#!/usr/bin/env python
# coding: utf-8


############### STEP 11: Performing SVM for classification of PIs with or without upper sampling and glueing the PIs together ###############



import time
import os 
from persim import PersImage
import matplotlib.pyplot as plt

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
from sklearn.model_selection import KFold # import KFold
from sklearn import svm
from imblearn.over_sampling import RandomOverSampler

base='/home/HospitalXLandmarks_CD10203845RIPS/4D Analysis/PersistenceImages'


dimensions=[0,1,2]
pixel=[5,10,25,50,100]
spreads=[0.01,0.05]

tablaLR=[[]]*len(dimensions)
tablaSVM=[[]]*len(dimensions)
tablaLR_LOOCV=[[]]*len(dimensions)
tablaSVM_LOOCV=[[]]*len(dimensions)
accuracylist=[]
scorelist=[]


for nitem in range(0,len(dimensions)):
    DIMENSION=dimensions[nitem]
    tablaLR[nitem]=[[] for i in range(len(pixel))]
    tablaSVM[nitem]=[[] for i in range(len(pixel))]
    tablaLR_LOOCV[nitem]=[[] for i in range(len(pixel))]
    tablaSVM_LOOCV[nitem]=[[] for i in range(len(pixel))]
    
    for npix in range(0,len(pixel)):
        pix=pixel[npix]
        pixels=[pix,pix]
        pixI=pixel.index(pix)
        tablaLR[nitem][pixI]=[[] for i in range(len(spreads))]
        tablaSVM[nitem][pixI]=[[] for i in range(len(spreads))]
        tablaLR_LOOCV[nitem][pixI]=[[] for i in range(len(spreads))]
        tablaSVM_LOOCV[nitem][pixI]=[[] for i in range(len(spreads))]
        
        for nspread in range(0,len(spreads)):
            spread=spreads[nspread]
            imgs_array=[]
            sprI=spreads.index(spread)
            tablaLR[nitem][pixI][sprI]=[[] for i in range(3)]
            tablaSVM[nitem][pixI][sprI]=[[] for i in range(4)]
            tablaLR_LOOCV[nitem][pixI][sprI]=[[] for i in range(3)]
            tablaSVM_LOOCV[nitem][pixI][sprI]=[[] for i in range(4)]  
            
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
            #ros = RandomOverSampler(random_state=0)
            #X_resampled, y_resampled = ros.fit_resample(imgs_array, labels)
            #imgs_array=X_resampled
            #labels=y_resampled
            seed=42
            X_train, X_test, y_train, y_test = train_test_split(imgs_array,labels,test_size=0.4, random_state = 42)
            
            
            
            print('###########################')
            print('DIMENSION')
            print(DIMENSION)
            
            print('Pixels')
            print(pix)
            
            print('Spread')
            print(spread)
            
            C_range = np.logspace(-2, 10, 13)
            gamma_range = np.logspace(-9, 3, 13)
            param_grid = dict(gamma=gamma_range, C=C_range)
            cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
            grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
            grid.fit(X_train, y_train)
            
            C_2d_range = [1e-2, 1, 1e2]
            gamma_2d_range = [1e-1, 1, 1e1]
            classifiers = []
            for C in C_2d_range:
                for gamma in gamma_2d_range:
                    clf = SVC(C=C, gamma=gamma)
                    clf.fit(X_train, y_train)
                    classifiers.append((C, gamma, clf))
            
            scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                                 len(gamma_range))


            C,gamma=list(grid.best_params_.values())
            
            classifier = svm.SVC(gamma=gamma,C=C,kernel='linear',random_state=None)
            #fit to the training data
            classifier.fit(X_train,y_train)
            # now to Now predict the value of the digit on the test data
            y_pred = classifier.predict(X_test)

            CV=6;

            scores = cross_val_score(classifier,X_test, y_test, cv=CV)
            predictions = cross_val_predict(classifier,X_test, y_test, cv=CV)
            accuracy = metrics.r2_score(y_test, predictions)
            print("Score "+str(np.mean(scores)))
            print("Accuracy "+str(accuracy))

            tablaSVM[nitem][pixI][sprI][0]=np.mean(scores)
            tablaSVM[nitem][pixI][sprI][1]=accuracy
            tablaSVM[nitem][pixI][sprI][2]=metrics.confusion_matrix(y_test, predictions)
            tablaSVM[nitem][pixI][sprI][3]=[C,gamma]
            accuracylist.append(accuracy)
            scorelist.append(np.mean(scores))
            print(len(imgs_array[0]))
            print('Done in '+str(np.round(time.time()-timeINI,3))+' secs.')





import pandas as pd
print('SVM K-FOLD')
for j in range(0,len(dimensions)):
    DIMENSION=dimensions[j]
    print('DIMENSION '+str(DIMENSION))
    for i in range(0,len(pixel)):
        
        print('--Pixels '+str(pixel[i]))
        df=pd.DataFrame(tablaSVM[j][i],spreads,['Score','Accuracy','Conf_matrix','C, gamma'])
        print(df)
    print('##########')



# Be sure to previously load PIs in ImgsNR and ImgsR as vectors of each PIs


labelsOriginal=np.concatenate([np.zeros(len(ImgsNR)),np.ones(len(ImgsR))])


################# HERE WE LOAD ALL IMAGES ONE AFTER THE OTHER FOR EACH PATIENT IN ALL DIMENSIONS

import time
import os 
from persim import PersImage
import matplotlib.pyplot as plt

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
from imblearn.over_sampling import RandomOverSampler


base='/home/HospitalXLandmarks_CD10203845RIPS/4D Analysis/PersistenceImages'



dimensions=[0,1,2]
pixel=[5,10,25,50,100]
spreads=[0.01,0.05]


tablaLR=[[]]*len(dimensions)
tablaSVM=[[]]*len(dimensions)
tablaLR_LOOCV=[[]]*len(dimensions)
tablaSVM_LOOCV=[[]]*len(dimensions)


datosgeneral=[]
for nitem in range(0,len(dimensions)):
    DIMENSION=dimensions[nitem]
    pixaux=[]
    for npix in range(0,len(pixel)):
        pix=pixel[npix]
        pixels=[pix,pix]
        pixI=pixel.index(pix)
        spreadaux=[]
        for nspread in range(0,len(spreads)):
            spread=spreads[nspread]
            imgs_array=[]
            sprI=spreads.index(spread)    
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
            for i in range(0,numberR):
                ImgsR[i]=np.loadtxt(dirR+'/'+listdirR[i],delimiter=' ')
            for i in range(0,numberNR):
                ImgsNR[i]=np.loadtxt(dirNR+'/'+listdirNR[i],delimiter=' ')

            
            imgs_array=[img.flatten() for img in np.concatenate([ImgsNR,ImgsR])]
            labels=np.concatenate([np.zeros(len(ImgsNR)),np.ones(len(ImgsR))])
            

            
            spreadaux.append([imgs_array,labels])
        pixaux.append(spreadaux)
    datosgeneral.append(pixaux)


# Here we concatenate the images obtained
nNR=len(listdirNR)
nR=len(listdirR)
n=nNR+nR
dataNR=[[[[]]*nNR]*len(spreads)]*len(pixel)
mat012=[]
for j in range(0,len(spreads)):
    spreadAUX=[]
    for i in range(0,len(pixel)):
        pixelAUX=[]
        for m in range(0,n):
            aux=np.concatenate((datosgeneral[0][i][j][0][m],datosgeneral[1][i][j][0][m],datosgeneral[2][i][j][0][m]))
            pixelAUX.append(aux)
        spreadAUX.append(pixelAUX)
        
    mat012.append(spreadAUX)



# Here we perform classification with those 'glued' images
nNR=len(listdirNR)
nR=len(listdirR)
n=nNR+nR
dataNR=[[[[]]*nNR]*len(spreads)]*len(pixel)
mat012=[]
for j in range(0,len(spreads)):
    spreadAUX=[]
    for i in range(0,len(pixel)):
        pixelAUX=[]
        for m in range(0,n):
            aux=np.concatenate((datosgeneral[0][i][j][0][m],datosgeneral[1][i][j][0][m],datosgeneral[2][i][j][0][m]))
            pixelAUX.append(aux)
        spreadAUX.append(pixelAUX)
    mat012.append(spreadAUX)
    
tablaSVM012=[[[] for j in range(len(pixel))] for i in range(len(spreads))]
for j in range(0,len(spreads)):
    for i in range (0,len(pixel)):
        timeINI=time.time()
        X_train, X_test, y_train, y_test = train_test_split(mat012[j][i],labels,test_size=0.4, random_state = 42)
        
        spread=spreads[j]
        pix=pixel[i]
        
        pixI=i
        sprI=j
        tablaSVM012[sprI][pixI]=[[] for i in range(4)]
        
        print('###########################')
        print('DIMENSION')
        print(DIMENSION)
        
        print('Pixels')
        print(pix)
        
        print('Spread')
        print(spread)
        
        
        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
        grid.fit(X_train, y_train)
        
        C_2d_range = [1e-2, 1, 1e2]
        gamma_2d_range = [1e-1, 1, 1e1]
        classifiers = []
        for C in C_2d_range:
            for gamma in gamma_2d_range:
                clf = SVC(C=C, gamma=gamma)
                clf.fit(X_train, y_train)
                classifiers.append((C, gamma, clf))
        
        scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                             len(gamma_range))
        C,gamma=list(grid.best_params_.values())
        
        classifier = svm.SVC(gamma=gamma,C=C,kernel='linear',random_state=None)
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        CV=6;
        scores = cross_val_score(classifier,X_test, y_test, cv=CV)
        predictions = cross_val_predict(classifier,X_test, y_test, cv=CV)
        accuracy = metrics.r2_score(y_test, predictions)  
        tablaSVM012[sprI][pixI][0]=np.mean(scores)
        tablaSVM012[sprI][pixI][1]=accuracy
        tablaSVM012[sprI][pixI][2]=metrics.confusion_matrix(y_test, predictions)
        tablaSVM012[sprI][pixI][3]=[C,gamma]
        
        print('Done in '+str(np.round(time.time()-timeINI,3))+' secs.')



# We can show the results with this last code

import pandas as pd
print('SVM K-FOLD')
for j in range(0,len(spreads)):
    print('----Spread '+str(spreads[j]))
    for i in range(0,len(pixel)):

        print('--Pixels '+str(pixel[i]))
        df=pd.DataFrame(tablaSVM012[j][i],['Score','Accuracy','Conf_matrix','C, gamma'])
        print(df)
print('##########')



# Here we perform classification with those 'glued' images with oversampling
nNR=len(listdirNR)
nR=len(listdirR)
n=nNR+nR
dataNR=[[[[]]*nNR]*len(spreads)]*len(pixel)
mat012=[]
for j in range(0,len(spreads)):
    spreadAUX=[]
    for i in range(0,len(pixel)):
        pixelAUX=[]
        for m in range(0,n):
            aux=np.concatenate((datosgeneral[0][i][j][0][m],datosgeneral[1][i][j][0][m],datosgeneral[2][i][j][0][m]))
            pixelAUX.append(aux)
        spreadAUX.append(pixelAUX)
    mat012.append(spreadAUX)
    
tablaSVM012=[[[] for j in range(len(pixel))] for i in range(len(spreads))]





for j in range(0,len(spreads)):
    for i in range (0,len(pixel)):
        timeINI=time.time()
        
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(mat012[j][i], labelsOriginal)
        MATRIX=X_resampled
        labels=y_resampled
        
        X_train, X_test, y_train, y_test = train_test_split(MATRIX,labels,test_size=0.4, random_state = 42)
        
        spread=spreads[j]
        pix=pixel[i]
        
        pixI=i
        sprI=j
        tablaSVM012[sprI][pixI]=[[] for i in range(4)]
        
        print('###########################')
        print('DIMENSION')
        print(DIMENSION)
        
        print('Pixels')
        print(pix)
        
        print('Spread')
        print(spread)
        
        
        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
        grid.fit(X_train, y_train)
        
        C_2d_range = [1e-2, 1, 1e2]
        gamma_2d_range = [1e-1, 1, 1e1]
        classifiers = []
        for C in C_2d_range:
            for gamma in gamma_2d_range:
                clf = SVC(C=C, gamma=gamma)
                clf.fit(X_train, y_train)
                classifiers.append((C, gamma, clf))
        
        scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                             len(gamma_range))
        C,gamma=list(grid.best_params_.values())
        
        classifier = svm.SVC(gamma=gamma,C=C,kernel='linear',random_state=None)
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        CV=6;
        scores = cross_val_score(classifier,X_test, y_test, cv=CV)
        predictions = cross_val_predict(classifier,X_test, y_test, cv=CV)
        accuracy = metrics.r2_score(y_test, predictions)  
        tablaSVM012[sprI][pixI][0]=np.mean(scores)
        tablaSVM012[sprI][pixI][1]=accuracy
        tablaSVM012[sprI][pixI][2]=metrics.confusion_matrix(y_test, predictions)
        tablaSVM012[sprI][pixI][3]=[C,gamma]
        
        print('Done in '+str(np.round(time.time()-timeINI,3))+' secs.')



# We can show the results with this last code
import pandas as pd
print('SVM K-FOLD')
for j in range(0,len(spreads)):
    print('----Spread '+str(spreads[j]))
    for i in range(0,len(pixel)):

        print('--Pixels '+str(pixel[i]))
        df=pd.DataFrame(tablaSVM012[j][i],['Score','Accuracy','Conf_matrix','C, gamma'])
        print(df)
print('##########')

