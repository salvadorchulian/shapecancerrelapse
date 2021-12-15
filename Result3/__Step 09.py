#!/usr/bin/env python
# coding: utf-8


############### STEP 9: First steps for the classification of Persistence Images ############### 






# Load PIs for analysis 
DIMENSION=0
pixels=[5,5]
spread=0.01

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

PimR = PersImage(pixels=pixels, spread=spread)

PimNR = PersImage(pixels=pixels, spread=spread)





# Show mean values of PIs
PimNR.show(np.mean(ImgsNR,axis=0))
PimR.show(np.mean(ImgsR,axis=0))



# Logistic regression for the PIs loaded

imgs_array=[img.flatten() for img in np.concatenate([ImgsNR,ImgsR])]
labels=np.concatenate([np.zeros(len(ImgsNR)),np.ones(len(ImgsR))])
#Selection of PIs as train and Test
ImgsNRtrain=[imgnr[0] for imgnr in zip(ImgsNR,[pac[0:3]!='HMU' for pac in listdirNR]) if imgnr[1]==True]
ImgsNRtest=[imgnr[0] for imgnr in zip(ImgsNR,[pac[0:3]=='HMU' for pac in listdirNR]) if imgnr[1]==True]
ImgsRtrain=[imgnr[0] for imgnr in zip(ImgsR,[pac[0:3]!='HMU' for pac in listdirR]) if imgnr[1]==True]
ImgsRtest=[imgnr[0] for imgnr in zip(ImgsR,[pac[0:3]=='HMU' for pac in listdirR]) if imgnr[1]==True]
X_train=[img.flatten() for img in np.concatenate([ImgsNRtrain,ImgsRtrain])]
y_train=np.concatenate([np.zeros(len(ImgsNRtrain)),np.ones(len(ImgsRtrain))])
X_test=[img.flatten() for img in np.concatenate([ImgsNRtest,ImgsRtest])]
y_test=np.concatenate([np.zeros(len(ImgsNRtest)),np.ones(len(ImgsRtest))])


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train, y_train)
lr.fit(X_train, y_train)
lrpred=lr.predict(X_test)
print(metrics.confusion_matrix(y_test, lrpred))
print('LR: '+str(lr.score(X_test,y_test)))

gamma,C=[1,1]
classifier = svm.SVC(gamma=gamma,C=C,kernel='linear',random_state=None)
#fit to the training data
classifier.fit(X_train,y_train)
# now to Now predict the value of the digit on the test data
y_pred = classifier.predict(X_test)
print(metrics.confusion_matrix(y_test, y_pred))
print('SVM: '+str(classifier.score(X_test,y_test)))

plt.subplots(1,2)
plt.subplot(121)
PimNR.show(np.mean(ImgsNRtrain,axis=0))
plt.subplot(122)
PimR.show(np.mean(ImgsRtrain,axis=0))








#Obtention of Discriminating Areas by means of LR coefficients 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
lr = LogisticRegression()
lr.fit(X_train, y_train)
lrpred=lr.predict(X_test)
print("Classification report for classifier %s:\n%s\n"
      % (lr, metrics.classification_report(y_test, lrpred)))
print("Confusion matrix:\n%s" % confusion_matrix(y_test, lrpred))
inverse_image = np.copy(lr.coef_).reshape((pixels[0],pixels[1]))
PimR.show(inverse_image)










# We obtained C and Gamma using the code from https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV


class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X_train, y_train)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))


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


plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.show()



C,gamma=list(grid.best_params_.values())



C,gamma=list(grid.best_params_.values())




from sklearn import svm
classifier = svm.SVC(gamma=gamma,C=C,kernel='linear')
#fit to the training data
classifier.fit(X_train,y_train)
# now to Now predict the value of the digit on the test data
y_pred = classifier.predict(X_test)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, y_pred)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_pred))
inverse_image = np.copy(classifier.coef_).reshape((pixels[0],pixels[1]))
PimR.show(inverse_image)








from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold 
lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train, y_train)
lr.fit(X_train, y_train)
lrpred=lr.predict(X_test)

lr.score(X_test,y_test)

CV=2;

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
scores = cross_val_score(lr,np.concatenate([X_test]), y_test, cv=CV)
print("Cross-Predicted Scores:", scores)
print("Mean Cross-Predicted Score:%.2f" % np.mean(scores))

predictions = cross_val_predict(lr, X_test, y_test, cv=CV)
accuracy = metrics.r2_score(y_test, predictions)
print("Cross-Predicted Accuracy:%.2f" % accuracy)
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictions))




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold 
classifier = svm.SVC(gamma=gamma,C=C,kernel='linear',random_state=None)

classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

CV=3;

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
scores = cross_val_score(classifier,X_test, y_test, cv=CV)
print("Cross-Predicted Scores:", scores)
print("Mean Cross-Predicted Score:%.2f" % np.mean(scores))

predictions = cross_val_predict(classifier,X_test, y_test, cv=CV)
accuracy = metrics.r2_score(y_test, predictions)
print("Cross-Predicted Accuracy:%.2f" % accuracy)
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictions))




from sklearn.model_selection import LeaveOneOut, cross_val_score
loocv = LeaveOneOut()
model_loocv = lr
results_loocv = cross_val_score(model_loocv,X_test,y_test, cv=loocv)
print("Accuracy: %.2f%%" % (results_loocv.mean()*100.0))
predictionsloo = cross_val_predict(model_loocv,X_test,y_test, cv=loocv)
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictionsloo))
accuracy = metrics.r2_score(y_test, predictionsloo)
print("Cross-Predicted Accuracy:%.2f" % accuracy)




from sklearn.model_selection import LeaveOneOut, cross_val_score
loocv = LeaveOneOut()
model_loocv = classifier
results_loocv = cross_val_score(model_loocv,X_test,y_test, cv=loocv)
print("Accuracy: %.2f%%" % (results_loocv.mean()*100.0))
predictionsloo = cross_val_predict(model_loocv,X_test,y_test, cv=loocv)
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictionsloo))
accuracy = metrics.r2_score(y_test, predictionsloo)
print("Cross-Predicted Accuracy:%.2f" % accuracy)



