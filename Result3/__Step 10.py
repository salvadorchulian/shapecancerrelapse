#!/usr/bin/env python
# coding: utf-8




############### STEP 10: Combining several Methods (SVM, LR, LOOCV) for classification of PIs ############### 



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



class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

base='/home/HospitalXLandmarks_CD10203845RIPS/'



dimensions=[2]
pixel=[5,10,25,50,100]
spreads=[0.01,0.05]

tablaLR=[[]]*len(dimensions)
tablaSVM=[[]]*len(dimensions)
tablaLR_LOOCV=[[]]*len(dimensions)
tablaSVM_LOOCV=[[]]*len(dimensions)
for nitem in range(0,len(dimensions)):
    DIMENSION=dimensions[nitem]
    tablaLR[nitem]=[[]]*len(pixel)
    tablaSVM[nitem]=[[]]*len(pixel)
    tablaLR_LOOCV[nitem]=[[]]*len(pixel)
    tablaSVM_LOOCV[nitem]=[[]]*len(pixel)
    for pix in pixel:
        pixels=[pix,pix]
        pixI=pixel.index(pix)
        tablaLR[nitem][pixI]=[[]]*len(spreads)
        tablaSVM[nitem][pixI]=[[]]*len(spreads)
        tablaLR_LOOCV[nitem][pixI]=[[]]*len(spreads)
        tablaSVM_LOOCV[nitem][pixI]=[[]]*len(spreads)
        for spread in spreads:
            sprI=spreads.index(spread)
            tablaLR[nitem][pixI][sprI]=[[]]*3
            tablaSVM[nitem][pixI][sprI]=[[]]*4
            tablaLR_LOOCV[nitem][pixI][sprI]=[[]]*3
            tablaSVM_LOOCV[nitem][pixI][sprI]=[[]]*4
            

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
            ImgsNRtrain=[imgnr[0] for imgnr in zip(ImgsNR,[pac[0:3]!='HMU' for pac in listdirNR]) if imgnr[1]==True]
            ImgsNRtest=[imgnr[0] for imgnr in zip(ImgsNR,[pac[0:3]=='HMU' for pac in listdirNR]) if imgnr[1]==True]
            ImgsRtrain=[imgnr[0] for imgnr in zip(ImgsR,[pac[0:3]!='HMU' for pac in listdirR]) if imgnr[1]==True]
            ImgsRtest=[imgnr[0] for imgnr in zip(ImgsR,[pac[0:3]=='HMU' for pac in listdirR]) if imgnr[1]==True]
            X_train=[img.flatten() for img in np.concatenate([ImgsNRtrain,ImgsRtrain])]
            y_train=np.concatenate([np.zeros(len(ImgsNRtrain)),np.ones(len(ImgsRtrain))])
            X_test=[img.flatten() for img in np.concatenate([ImgsNRtest,ImgsRtest])]
            y_test=np.concatenate([np.zeros(len(ImgsNRtest)),np.ones(len(ImgsRtest))])
            
            
            
            print('###########################')
            print('DIMENSION')
            print(DIMENSION)
            
            print('Pixels')
            print(pix)
            
            print('Spread')
            print(spread)
            
            lr = LogisticRegression(solver='lbfgs',max_iter=100)
            lr.fit(X_train, y_train)
            lr.fit(X_train, y_train)
            lrpred=lr.predict(X_test)
            print(metrics.confusion_matrix(y_test, lrpred))
            print('LR: '+str(lr.score(X_test,y_test)))
            
            gamma,C=[1,1]
            classifier = svm.SVC(gamma=gamma,C=C,kernel='linear',random_state=None)
            classifier.fit(X_train,y_train)
            y_pred = classifier.predict(X_test)
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

            print('C')
            print(C)
            print('gamma')
            print(gamma)

            
            classifier = svm.SVC(gamma=gamma,C=C,kernel='linear')
            classifier.fit(X_train,y_train)
            y_pred = classifier.predict(X_test)
            
            print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_pred))
            inverse_image = np.copy(classifier.coef_).reshape((pixels[0],pixels[1]))
            
            
            lr = LogisticRegression(max_iter=100)
            lr.fit(X_train, y_train)
            lrpred=lr.predict(X_test)
            
            lr.score(X_test,y_test)
            
            

            CV=3;

            
            
            scores = cross_val_score(lr,np.concatenate([X_test]), y_test, cv=CV)
            predictions = cross_val_predict(lr,np.concatenate([X_test]),y_test, cv=CV)
            print("Cross-Predicted Scores:", scores)
            print("Mean Cross-Predicted Score:%.2f" % np.mean(scores))

            
            predictions = cross_val_predict(lr, X_test, y_test, cv=CV)
            accuracy = metrics.r2_score(y_test, predictions)
            print("Cross-Predicted Accuracy:%.2f" % accuracy)
            print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictions))
            
            
            
            tablaLR[nitem][pixI][sprI][0]=np.mean(scores)
            tablaLR[nitem][pixI][sprI][1]=accuracy
            tablaLR[nitem][pixI][sprI][2]=metrics.confusion_matrix(y_test, predictions)
            
            
            
            
            
            
            
            classifier = svm.SVC(gamma=gamma,C=C,kernel='linear',random_state=None)
            classifier.fit(X_train,y_train)

            y_pred = classifier.predict(X_test)



            CV=3;

            scores = cross_val_score(classifier,X_test, y_test, cv=CV)
            print("Cross-Predicted Scores:", scores)
            print("Mean Cross-Predicted Score:%.2f" % np.mean(scores))

 
            predictions = cross_val_predict(classifier,X_test, y_test, cv=CV)
            accuracy = metrics.r2_score(y_test, predictions)
            print("Cross-Predicted Accuracy:%.2f" % accuracy)
            print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictions))
            

            
            tablaSVM[nitem][pixI][sprI][0]=np.mean(scores)
            tablaSVM[nitem][pixI][sprI][1]=accuracy
            tablaSVM[nitem][pixI][sprI][2]=metrics.confusion_matrix(y_test, predictions)
            tablaSVM[nitem][pixI][sprI][3]=[C,gamma]
            
            
            loocv = LeaveOneOut()
            model_loocv = lr
            results_loocv = cross_val_score(model_loocv,X_test,y_test, cv=loocv)
            print("Accuracy: %.2f%%" % (results_loocv.mean()*100.0))
            predictionsloo = cross_val_predict(model_loocv,X_test,y_test, cv=loocv)
            print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictionsloo))
            accuracy = metrics.r2_score(y_test, predictionsloo)
            print("Cross-Predicted Accuracy:%.2f" % accuracy)

            
            tablaLR_LOOCV[nitem][pixI][sprI][0]=np.mean(scores)
            tablaLR_LOOCV[nitem][pixI][sprI][1]=accuracy
            tablaLR_LOOCV[nitem][pixI][sprI][2]=metrics.confusion_matrix(y_test, predictionsloo)


            
            loocv = LeaveOneOut()
            model_loocv = classifier
            results_loocv = cross_val_score(model_loocv,X_test,y_test, cv=loocv)
            print("Accuracy: %.2f%%" % (results_loocv.mean()*100.0))
            predictionsloo = cross_val_predict(model_loocv,X_test,y_test, cv=loocv)
            print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictionsloo))
            accuracy = metrics.r2_score(y_test, predictionsloo)
            print("Cross-Predicted Accuracy:%.2f" % accuracy)
            
            
            tablaSVM_LOOCV[nitem][pixI][sprI][0]=np.mean(scores)
            tablaSVM_LOOCV[nitem][pixI][sprI][1]=accuracy
            tablaSVM_LOOCV[nitem][pixI][sprI][2]=metrics.confusion_matrix(y_test, predictionsloo)
            tablaSVM_LOOCV[nitem][pixI][sprI][3]=[C,gamma]




print('LR K-FOLD')
for j in range(0,len(dimensions)):
    DIMENSION=dimensions[j]
    print('DIMENSION '+str(DIMENSION))
    for i in range(0,len(pixel)):
        
        print('--Pixels '+str(pixel[i]))
        df=pd.DataFrame(tablaLR[j][i],spreads,['Score','Accuracy','Conf_matrix'])
        print(df)
    print('##########')
print('LR_LOOCV')
for j in range(0,len(dimensions)):
    DIMENSION=dimensions[j]
    print('DIMENSION '+str(DIMENSION))
    for i in range(0,len(pixel)):
        
        print('--Pixels '+str(pixel[i]))
        df=pd.DataFrame(tablaLR_LOOCV[j][i],spreads,['Score','Accuracy','Conf_matrix'])
        print(df)
    print('##########')


print('SVM K-FOLD')
for j in range(0,len(dimensions)):
    DIMENSION=dimensions[j]
    print('DIMENSION '+str(DIMENSION))
    for i in range(0,len(pixel)):
        
        print('--Pixels '+str(pixel[i]))
        df=pd.DataFrame(tablaSVM[j][i],spreads,['Score','Accuracy','Conf_matrix','C, gamma'])
        print(df)
    print('##########')
print('SVM_LOOCV')
for j in range(0,len(dimensions)):
    DIMENSION=dimensions[j]
    print('DIMENSION '+str(DIMENSION))
    for i in range(0,len(pixel)):
        
        print('--Pixels '+str(pixel[i]))
        df=pd.DataFrame(tablaSVM_LOOCV[j][i],spreads,['Score','Accuracy','Conf_matrix','C, gamma'])
        print(df)
    print('##########')
    

