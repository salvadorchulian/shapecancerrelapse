#!/usr/bin/env python
# coding: utf-8


############### STEP 1: Reading files ############### 



#Import basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os


def readingdir(direct):
    # The input of this function is the folder with subfolders of patients and their fcs files.
    # Merging all samples and normalisation is not included in these files. 
    output='/home/outputfolder'
    os.mkdir(output)
    listdir=os.listdir(direct)
    listdir.sort()
    exports=[];
    for i in range(0,len(listdir)):
        if listdir[i][0]!='.':
            print(listdir[i])
            os.mkdir(output+'/'+listdir[i])
            listpac=os.listdir(direct+'/'+listdir[i])
            listpac.sort()
            exportspac=[];
            for j in range(0,len(listpac)):
                if listpac[j][0]!='.':
                    exportation=reading(direct+'/'+listdir[i]+'/'+listpac[j])
                    exportspac.append(exportation)
                    if len(exportation)!=0:
                        print(listpac[j])
                        os.mkdir(output+'/'+listdir[i]+'/'+listpac[j].split('.')[0]) 
                        f = open(output+'/'+listdir[i]+'/'+listpac[j].split('.')[0]+'/Tube'+str(j)+'.txt', "w")
                        np.savetxt(f, exportation.values, fmt='%f')
                        f.close
            exports.append([listdir[i],exportspac])
    return exports;
    
def reading(datafile):
    tube=flow.Tube(file=datafile)
    namemetadata=flow.operations.import_op.autodetect_name_metadata(datafile)
    import_op = flow.ImportOp(tubes = [tube],name_metadata='$PnN')
    try: 
        exp=import_op.apply()
    except KeyError:
        import_op = flow.ImportOp(tubes = [tube],name_metadata='$PnS')
        exp=import_op.apply()
    dicti=exp.metadata['fcs_metadata'][datafile]
    exportation=[];
    npar=dicti['$PAR'];
    dicti['lista']=[]
    for i in range(1,npar+1):
        if '$P'+str(i)+'S'in dicti:
            aux=dicti['$P'+str(i)+'S'];
            if aux.isdecimal() or (aux[0].isdigit() and aux.endswith(('A','B','C'))):
                dicti['lista'].append(['CD'+aux,dicti['$P'+str(i)+'N']])
            else:
                dicti['lista'].append([aux,dicti['$P'+str(i)+'N']])
        else:
            dicti['lista'].append([dicti['$P'+str(i)+'N']])

            
    ALLPARAM=['CD34','CD20','CD10','CD19','CD45','CD13','CD33','cyCD3','cyMPO','CD22','IGM','CD38','cyTDT','CD3','CD66','CD58']
    ALLPARAM.sort()
    chosenpar=ALLPARAM
    [samplepar,columnname]=choosepar(dicti['lista'],chosenpar,exp.channels)
    if len(samplepar)==len(chosenpar):
        exportation=exp.data[samplepar]
        exportation.columns=columnname
        return exportation
    else:
        return [];

def choosepar(lista,chosenpar,channels):
    columnname=[];
    samplepar=[];
    changedpar=[];
    for i in chosenpar:
        for j in range(0,len(lista)):
            markers=lista[j];
            if i in markers[0] and len(i)==len(markers[0]) and not any(i in s for s in samplepar):
                if markers[0] in channels:
                    samplepar.append(markers[0])
                else:
                    samplepar.append(markers[1])
                columnname.append(markers[0])
    return [samplepar,columnname]





readingdir('/home/inputfolder')





# Check for duplicates in data and creates a folder with those files
import os
import pandas as pd
direct='/home/outputfolder/NonRelapse' #Reading folder 
outdir='/home/outputfolder_nonduplicate/NonRelapse' #Outputfolder
os.mkdir(outdir)
listdir=os.listdir(direct)
listdir.sort()
df=[]
for i in range(0,len(listdir)):
    pacient=listdir[i]
    if pacient[0]!='.':
        print('----------')
        print(pacient)
        listpac=os.listdir(direct+'/'+pacient)
        listpac.sort()
        for j in range(0,len(listpac)):
            tube=listpac[j]
            df=pd.read_csv(direct+'/'+pacient+'/'+tube,sep=' ',header=None)
            dfnon=df[~df.duplicated(keep='first')]
            if len(df[df.duplicated(keep='first')])>=1:
                
                os.mkdir(outdir+'/'+pacient)
                print(pacient+' duplicated a lot: '+str(len(dfnon))+'/'+str(len(df)))
                dfnon.to_csv(outdir+'/'+pacient+'/'+tube+'_nondup.csv')





############### STEP 2: Obtaining LANDMARKS of each file ############### 






from __future__ import print_function
import pandas as pd
import time
from scipy.spatial import distance
import numpy as np
import operator
import progressbar
from time import sleep
import os
import errno
import re

from joblib import Parallel, delayed
import multiprocessing


def MaxMinPAC(pacient,direct,outdir):
    listpac=os.listdir(direct+'/'+pacient)
    listpac.sort()
    if pacient[0]!='.':
        print('----------')
        print(pacient)
        listpac=os.listdir(direct+'/'+pacient)
        listpac.sort()
        for j in range(0,len(listpac)):
            MaxMin(listpac[j],pacient,direct,outdir)
        
def MaxMin(tube,pacient,direct,outdir):
    extension=tube.split('.')[-1]
    if extension=='txt':
        data=pd.DataFrame(np.loadtxt(direct+'/'+pacient+'/'+tube, comments="#", delimiter=" ", unpack=False))
    elif extension=='csv':
        data=pd.read_csv(direct+'/'+pacient+'/'+tube,sep=',',header=0).drop(columns='Unnamed: 0')
    else:
        print('Problem reading.')
    n=len(data)
    m=10000 #Number of landmarks selected
    print('Landmarking '+pacient+'/'+tube)
    pacientnumber=int(re.findall("\d+", pacient)[0])
    if n<m:
        print('There are enough points:'+str(n)+'!')
    elif pacientnumber<=5:
        try:
            os.mkdir(outdir+'/'+pacient)
            bar = progressbar.ProgressBar(maxval=m,                 widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(),'.', progressbar.Timer(),'. Points:', progressbar.Counter()])
            bar.start()
            tic=time.time()
            L=data.sample(n=1)
            yindex=L.index.values[0]
            datanew=data.drop(yindex)

            D=[]
            for i in range(0,m-1):
                indexlist=[]
                maxlist=[]
                y=L.iloc[[-1]]
                D.append(pd.DataFrame(distance.cdist(y,datanew,'euclidean')[0],index=datanew.index))
                maxnew=[[item.idxmax()[0],item.max()[0]] for item in D]
                indexbig,biggest=max(enumerate(maxnew),key=operator.itemgetter(1))
                for j in range(0,len(D)):
                    D[j].loc[[biggest[0]]]=0
                L=pd.concat([L,datanew.loc[[biggest[0]]]])
                datanew=datanew.drop(biggest[0])
                bar.update(i+2)
                sleep(0.1)
            bar.finish()
            toc=time.time()
            print(str(toc-tic))
            print('Tube '+pacient+'/'+tube+' landmarked!')
            L.to_csv(outdir+'/'+pacient+'/'+tube+'_Landmarks.csv')
            datanew.to_csv(outdir+'/'+pacient+'/'+tube+'_Others.csv')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Directory for Patient '+ outdir+'/'+pacient+' already exists!')
            else:
                raise
    else:
         print('Patient '+pacient+' is being coded somewhere else.')

direct='/home/HospitalX/NonRelapse'
outdir='/home/HospitalXLandmarks/NonRelapse'
os.mkdir(outdir)
listdir=os.listdir(direct)
listdir.sort()
num_cores=multiprocessing.cpu_count()
Parallel(n_jobs=num_cores)(delayed(MaxMinPAC)(listdir[i],direct,outdir) for i in range(0,len(listdir)))





############### STEP 3: VIETORIS-RIPS in each patient for all pairwise combinations of markers ###############





import numpy as np
import pandas as pd
import time
from ripser import ripser
from persim import plot_diagrams
from numpy import loadtxt
import os
import errno
import multiprocessing
from joblib import Parallel, delayed
from itertools import combinations


def paramrips(loadfile,TODOS,combi,tube,pacient,ripsdir):
    start_time = time.time()
    ripsfile=ripser(np.array(loadfile[:,combi]))['dgms']
    tuberips=ripsdir+'/'+TODOS[combi[0]]+"-"+TODOS[combi[1]]+'/'+pacient
    os.mkdir(tuberips)
    print("--- %s seconds --- Pacient " % (time.time() - start_time)+pacient+" Markers "+TODOS[combi[0]]+"-"+TODOS[combi[1]])
    f = open(tuberips+'/'+tube+'-'+TODOS[combi[0]]+"-"+TODOS[combi[1]]+'_Rips0.txt', "w")
    np.savetxt(f, ripsfile[0], fmt='%f')
    f.close()
    f = open(tuberips+'/'+tube+'-'+TODOS[combi[0]]+"-"+TODOS[combi[1]]+'_Rips1.txt', "w")
    np.savetxt(f, ripsfile[1], fmt='%f')
    f.close();


def ripsing(tube,pacient,direct,ripsdir):
        if tube[0]!='.' and tube.split('.')[-2][-6:]!='Others':
            print(tube)
            file_name=direct+'/'+pacient+'/'+tube
            file_rips=ripsdir+'/'+pacient+'/'+tube
            file_stats = os.stat(file_name)
            extension=file_name.split('.')[-1]
            MB_file=file_stats.st_size / (1000000)
            ALLPARAM=['CD10','CD13','CD14+CD8','CD19','CD20','CD22','CD3','CD33','CD34','CD38','CD4','CD45','CD58','CD66','CD79a','CD79b','HLADR','IGM','TDTc','cMPO','cyCD3','cyIGM']
            comb=list(combinations(range(0,len(ALLPARAM)),2))
            if not os.path.exists(file_rips+'_Rips0.txt'):
                if MB_file<=1.5:
                    if extension=='txt':
                        loadfile=loadtxt(file_name, comments="#", delimiter=" ", unpack=False)
                    elif extension=='csv':
                        loadfile=np.array(pd.read_csv(file_name).drop(columns='Unnamed: 0'))
                    else:
                        print('Problem reading.')
                    for k in range(0,len(comb)):
                        paramrips(loadfile,ALLPARAM,comb[k],tube,pacient,ripsdir)
                    print("----- Ended "+pacient)
            else:
                print('Tube '+pacient+'/'+tube+' already ripsed!')
        
def general(direct,ripsdir,P):
    if P[0]!='.':
        print('----------')
        print(P)
        listpac=os.listdir(direct+'/'+P)
        listpac.sort()
        try:
            for j in range(0,len(listpac)):
                ripsing(listpac[j],P,direct,ripsdir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Directory for Patient '+ ripsdir+'/'+P+' already exists!')
            else:
                raise


direct='/home/HospitalXLandmarks/NonRelapse'
ripsdir='/home/HospitalXLandmarks_RIPS/NonRelapse'
listdir=os.listdir(direct)
listdir.sort()
num_cores=multiprocessing.cpu_count()
print('Begin RIPS')
ALLPARAM=['CD34','CD20','CD10','CD19','CD45','CD13','CD33','cyCD3','cyMPO','CD22','IGM','CD38','cyTDT','CD3','CD66','CD58']
ALLPARAM.sort()
comb=list(combinations(range(0,len(ALLPARAM)),2))

Parallel(n_jobs=num_cores)(delayed(general)(direct,ripsdir,listdir[j]) for j in range(0,len(listdir)))





############### STEP 4: Statistical analysis of pairwise combinations of all markers ############### 



from __future__ import print_function
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
def read_and_save(filedir,tube):
    if tube[0]!='.':
        file_name, file_extension = os.path.splitext(filedir+'/'+tube)
        tubename=tube.split('_')[0].split('.')[0] #TubeX "name"
        tubenamerips=tube.split('_')[-1].split('.')[0] #Feature from Tube X: Rips 0 or 1
        if file_extension != '.pdf' and tubenamerips=='Rips0':
            Rips0=np.array(pd.read_csv(filedir+'/'+'_'.join(tube.split('_')[:-1])+'_Rips0.txt', sep=" ", header=None))
            if np.isnan(Rips0[-1,1]):
                Rips0[-1,1]=np.inf
            Rips1=np.array(pd.read_csv(filedir+'/'+'_'.join(tube.split('_')[:-1])+'_Rips1.txt', sep=" ", header=None))
            if np.isnan(Rips1[-1,1]):
                Rips1[-1,1]=0
            Rips=[Rips0,Rips1]
            Biglist=[None]*2
            isinf=[False]*2
            for j in range(0,len(Rips)):
                Persistlist=[None]*len(Rips[j])
                for i in range(0,len(Rips[j])):
                    if not np.isinf(Rips[j][i][1]) and not np.isinf(Rips[j][i][0]):
                        Persistlist[i]=(Rips[j][i][1]-Rips[j][i][0])
                    else:
                        isinf[j]=True
                        Persistlist.remove(None)
                Biglist[j]=Persistlist
            return [Rips,Biglist,isinf]
        else:
            return []
direct='/home/HospitalXLandmarks_RIPS/NonRelapse'
outputdir='/home/HospitalXLandmarks_RIPS/NonRelapseAnalysis'
os.mkdir(outputdir)
listparam=os.listdir(direct)
listparam.sort()

def splitting(lista,nametubes,namepacs):
    splitter = lambda lista: [x.split('.')[0] for x in lista]
    origtubenames = list(map(splitter, nametubes))
    tubeperpac=[]
    for i in range(0,len(origtubenames)):
        tubeperpac.append([namepacs[i]]*len(origtubenames[i]))
    flat_tubes = []
    flat_pacs=[]
    for sublist in origtubenames:
        for item in sublist:
            flat_tubes.append(item)
    for sublist in tubeperpac:
        for item in sublist:
            flat_pacs.append(item)
    arraytubes=np.array(flat_tubes)
    arraypacs=np.array(flat_pacs)
    arrays=[arraypacs,arraytubes]
    return [savepacs,lista,flat_tubes,arrays]


for l in range(0,len(listparam)): #Analysis for each combination of parameters
    savetubes=[]
    if listparam[l][0]!='.':
        print('\n----------')
        print(listparam[l])#
        commondir=direct+'/'+listparam[l]
        listdir=os.listdir(commondir)
        listdir.sort()
        savetubes=[]
        savepacs=[]
        namepacs=[]
        nametubes=[]
        for i in range(0,len(listdir)): # See each patient
            savetubes=[]
            if listdir[i][0]!='.':
                filedir=commondir+'/'+listdir[i]
                listpac=os.listdir(filedir) 
                listpac.sort()
                for j in range(0,len(listpac)):# Analyse each file 
                    if listpac[j][0]!='.':
                        lista=read_and_save(filedir,listpac[j])
                        if len(lista)!=0:
                            if len(savetubes)==0:
                                savetubes=[lista]
                            else:
                                savetubes.append(lista)
                if len(nametubes)==0:
                    nametubes=[[x for x in listpac if (not x.startswith('.')) and (not x.endswith('1.txt'))]]
                else:
                    nametubes.append([x for x in listpac if (not x.startswith('.')and (not x.endswith('1.txt')))])
                if len(savepacs)==0:
                    savepacs=[savetubes]
                else:
                    savepacs.append(savetubes)
                if len(namepacs)==0:
                    namepacs=[listdir[i]]
                else:
                    namepacs.append(listdir[i])
        [savepacs,lista,flat_tubes,arrays]=splitting(lista,nametubes,namepacs)
        maxper=[[]]*2
        medper=[[]]*2
        meanper=[[]]*2
        minper=[[]]*2
        stdper=[[]]*2
        lenlist=[[]]*2
        print('\n------ Statistic Analysis ------')
        for i in range(0,len(savepacs)): #for every patient
     
            for j in range(0,len(savepacs[i])):#for every tube

                for k in range(0,2): #for every rips (there are two)
                    lista=savepacs[i][j][1][k]
                   
                    if len(maxper[k])!=0:
                        maxper[k].append(max(lista))
                        minper[k].append(np.min(lista))
                        medper[k].append(np.median(lista))
                        meanper[k].append(np.mean(lista))
                        lenlist[k].append(len(lista))
                        stdper[k].append(np.std(lista))
                    else:
                        maxper[k]=[max(lista)]
                        minper[k]=[np.min(lista)]
                        medper[k]=[np.median(lista)]
                        meanper[k]=[np.mean(lista)]
                        lenlist[k]=[len(lista)]
                        stdper[k]=[np.std(lista)]
        columnnames=['MaxPer0','MaxPer1','MinPer0','MinPer1','MedPer0','MedPer1','MeanPer0','MeanPer1','StdPer0','StdPer1','Lenlist0','Lenlist1']
        #Create Vector of Max,Min,Med,Mean and Std persistence in each dimension
        df = pd.DataFrame(np.array([maxper[0],maxper[1],minper[0],minper[1],medper[0],medper[1],meanper[0],meanper[1],stdper[0],stdper[1],lenlist[0],lenlist[1]]).transpose(),
                          index=arrays,columns=columnnames)
 
        finalout=outputdir+'/'+listparam[l]
        os.mkdir(finalout)
        df.to_csv(finalout+'/'+listparam[l]+'.csv')
        dfnorm=pd.DataFrame(np.empty((len(flat_tubes),len(columnnames))),
                          index=arrays,columns=columnnames)
        for i in range(0,len(columnnames)):
            if i%2==0: #if its even (Dim 0)
                dfnorm[columnnames[i]]=df[columnnames[i]].values/df['Lenlist0'].values
            else: #if its odd (Dim 1)
                dfnorm[columnnames[i]]=df[columnnames[i]].values/df['Lenlist1'].values
        dfnorm.to_csv(finalout+'/'+listparam[l]+'_norm.csv')
        #Create file normalised over the length of each barcode
        print("Dataframes exported")





############### STEP 5: RANDOM FOREST analysis ############### 





TODOSSEV=['CD10','CD123','CD13','CD19','CD20','CD24','CD33','CD34','CD38','CD45','CD66c','CD7','CD9','FSC.H','FSC.A','SSC.A','KAPPA','LAMBDA']
TODOSNJ=['CD10','CD13','CD15','CD19','CD20','CD22','CD24','CD33','CD34','CD38','CD45','CD58','CD66c','CD7','CD71','HLADR','IGM','cyCD3']
TODOSMU=['CD10','CD13','CD14+CD8','CD19','CD20','CD22','CD3','CD33','CD34','CD38','CD4','CD45','CD58','CD66','CD79a','CD79b','HLADR','IGM','TDTc','cMPO','cyCD3','cyIGM']
TODOSCONJ=list(set(TODOSSEV).intersection(TODOSNJ))
TODOSCONJ=list(set(TODOSCONJ).intersection(TODOSMU))
TODOSCONJ.sort()
TODOSCONJ


import os
import pandas as pd
sourcerelapse='/home/HospitalXLandmarks_RIPS/RelapseAnalysis'
sourcenonrelapse='/home/HospitalXLandmarks_RIPS/NonRelapseAnalysis'
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
for l in range(0,len(listparam)):#For each parameter combination do the following analysis
    if listparam[l][0]!='.':
        print('\n----------')
        print(listparam[l])
        listpac=os.listdir(sourcerelapse+'/'+listparam[l])
        listpac.sort()
        for j in range(0,len(listpac)):
            if listpac[j][0]!='.':
                if listpac[j].split('.')[-2][-4:]!='norm':
                    deletehosp='HMU' #Use discovery set, delete the validation set
                    datarelapse=pd.read_csv(sourcerelapse+'/'+listparam[l]+'/'+listpac[j])
                    datanonrelapse=pd.read_csv(sourcenonrelapse+'/'+listparam[l]+'/'+listpac[j])




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

sourcerelapse='/home/HospitalXLandmarks_RIPS/RelapseAnalysis'
sourcenonrelapse='/home/HospitalXLandmarks_RIPS/NonRelapseAnalysis'


listparam=os.listdir(sourcerelapse)
listparam.sort()
listparam=[x for x in listparam if all([y in listacomun for y in x.split('-')])]
###
NORMED=False
###
names=listparam
names.sort()
df=pd.DataFrame(columns=['AUC','OOB','TPR','TNR','PPV','NPV','FPR','FNR','FDR','ACC'],index=names)
for l in range(0,len(listparam)):
    if listparam[l][0]!='.':
        print('\n----------')
        print(listparam[l])
        listpac=os.listdir(sourcerelapse+'/'+listparam[l])
        listpac.sort()
        for j in range(0,len(listpac)):
            if listpac[j][0]!='.':
                if listpac[j].split('.')[-2][-4:]!='norm':
                    ##Possibilities of including or excluding hospital data
                    deletehosp1='HMU'
                    deletehosp2='HVR'
                    datarelapse=pd.read_csv(sourcerelapse+'/'+listparam[l]+'/'+listpac[j])
                    datanonrelapse=pd.read_csv(sourcenonrelapse+'/'+listparam[l]+'/'+listpac[j])
                    datarelapse=datarelapse[~datarelapse.iloc[:,0].astype(str).str.startswith(deletehosp1)]
                    datarelapse=datarelapse.reset_index(drop=True)
                    datarelapse=datarelapse[~datarelapse.iloc[:,0].astype(str).str.startswith(deletehosp2)]
                    datarelapse=datarelapse.reset_index(drop=True)
                    datanonrelapse=datanonrelapse[~datanonrelapse.iloc[:,0].astype(str).str.startswith(deletehosp1)]
                    datanonrelapse=datanonrelapse.reset_index(drop=True)
                    datanonrelapse=datanonrelapse[~datanonrelapse.iloc[:,0].astype(str).str.startswith(deletehosp2)]
                    datanonrelapse=datanonrelapse.reset_index(drop=True)

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

                        for label, clf in ensemble_clfs:
                            for i in range(min_estimators, max_estimators + 1):
                                clf.set_params(n_estimators=i)
                                clf.fit(train[features], y)

                                # Record the OOB error 
                                oob_error = 1 - clf.oob_score_
                                error_rate[label].append((i, oob_error))

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
                        
                        ooberror.append(np.mean(ys))

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

df.to_csv('/'.join(sourcerelapse.split('/')[0:-1])+'/'+'Analysis_0001_NJ.csv')



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sourcerelapse='/home/HospitalXLandmarks_RIPS/RelapseAnalysis'
sourcenonrelapse='/home/HospitalXLandmarks_RIPS/NonRelapseAnalysis'

sourcerelapse='/Users/salvador/Google Drive/OXFORD2021/TODOSJUNTOS0.001_RIPS/NonRelapseAnalysis'
sourcenonrelapse='/Users/salvador/Google Drive/Oxford2021/TODOSJUNTOS0.001_RIPS/RelapseAnalysis'

dfnotnormed=pd.read_csv('/'.join(sourcerelapse.split('/')[0:-1])+'/'+'Analysis_0001_SE_NJ.csv')

TODOSSEV=['CD10','CD123','CD13','CD19','CD20','CD24','CD33','CD34','CD38','CD45','CD66c','CD7','CD9','FSC.H','FSC.A','SSC.A','KAPPA','LAMBDA']
TODOSNJ=['CD10','CD13','CD15','CD19','CD20','CD22','CD24','CD33','CD34','CD38','CD45','CD58','CD66c','CD7','CD71','HLADR','IGM','cyCD3']
TODOSMU=['CD10','CD13','CD14+CD8','CD19','CD20','CD22','CD3','CD33','CD34','CD38','CD4','CD45','CD58','CD66','CD79a','CD79b','HLADR','IGM','TDTc','cMPO','cyCD3','cyIGM']
TODOSCONJ=list(set(TODOSSEV).intersection(TODOSNJ))
TODOSCONJ=list(set(TODOSCONJ).intersection(TODOSMU))
TODOSCONJ.sort()
TODOSJUNTOS=['CD34','CD20','CD10','CD19','CD45','CD13','CD33','cyCD3','cyMPO','CD22','IGM','CD38','cyTDT','CD3','CD66','CD58']
TODOSJUNTOS.sort()
#SELECCION=maxselected
TODOSEL=['CD20','CD34','CD38','CD58','cyMPO','cyTDT']
TODOS=TODOSJUNTOS
aucmean=[[]]*len(TODOS)
for i in range(0,len(TODOS)):
    aucaux=np.zeros(len(TODOS)-1)
    k=0
    for j in range(0,len(dfnotnormed)): 
        lista=dfnotnormed.iloc[j,0].split('-')
        if TODOS[i] in lista and lista[1-lista.index(TODOS[i])] in TODOS:
            aucaux[k]=dfnotnormed.iloc[j,1]
            k=k+1
    aucmean[i]=[aucaux]
accmean=[[]]*len(TODOS)
for i in range(0,len(TODOS)):
    accaux=np.zeros(len(TODOS)-1)
    k=0
    for j in range(0,len(dfnotnormed)): 
        lista=dfnotnormed.iloc[j,0].split('-')
        if TODOS[i] in lista and lista[1-lista.index(TODOS[i])] in TODOS:
            accaux[k]=dfnotnormed.iloc[j,-1]
            k=k+1
    accmean[i]=[accaux]
    
aucs=np.array(list(map(np.mean,aucmean)))
plt.figure(figsize=(15, 3),dpi=600)
plt.bar(TODOS,aucs)
plt.axhline(y=0.5, color='r', linestyle=':')
plt.ylim([0.5,1])
plt.xticks(rotation=45,size=20)
plt.yticks(size=20)





############### STEP 6: Vietoris-Rips of Biomarkers set CD10-20-38-45 ###############





# Repeat Steps 1 and 2 and obtain analysis. 
# Important: we accounted here for dimension 2 as well, reducing the number of landmarks to 1000.
# We ran this analysis for the pairwise combinations of files (see Previous Steps)





# Dimension 0 and 1: 





#
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from ripser import ripser
from persim import plot_diagrams
from numpy import loadtxt
import os
import errno
import multiprocessing
from joblib import Parallel, delayed

def ripsing(tube,pacient,direct,ripsdir):
    if tube[0]!='.' and tube.split('.')[-2][-6:]!='Others':
        print(tube)
        file_name=direct+'/'+pacient+'/'+tube
        file_rips=ripsdir+'/'+pacient+'/'+tube
        file_stats = os.stat(file_name)
        extension=file_name.split('.')[-1]
        MB_file=file_stats.st_size / (1000000)
        if not os.path.exists(file_rips+'_Rips0.txt'):
            if MB_file<=1.5:
                if extension=='txt':
                    loadfile=loadtxt(file_name, comments="#", delimiter=" ", unpack=False)
                elif extension=='csv':
                    loadfile=pd.read_csv(file_name).drop(columns='Unnamed: 0')
                else:
                    print('Problem reading.')
                print('Begin Rips')
                start_time = time.time()
                ripsfile=ripser(loadfile)['dgms']
                print("--- %s seconds ---" % (time.time() - start_time))
                #SAVED RIPS
                f = open(file_rips+'_Rips0.txt', "w")
                np.savetxt(f, ripsfile[0], fmt='%f')
                f.close()
                f = open(file_rips+'_Rips1.txt', "w")
                np.savetxt(f, ripsfile[1], fmt='%f')
                f.close()
                f = open('ListR.txt',"a")
                f.write(pacient+'/'+tube+' read. Size:'+str(MB_file)+' MBs.\n')
                f.close()
         
            else:
                print('Not in range(0,2): Bigger than '+str(MB_file)+' MBs')
                f = open('ListR.txt',"a")
                f.write(pacient+'/'+tube+' NOT read. Size>'+str(MB_file)+' MBs.\n')
                f.close()
        else:
            print('Tube '+pacient+'/'+tube+' already ripsed!')
            f = open('ListR.txt',"a")
            f.write(pacient+'/'+tube+' already ripsed. Size:'+str(MB_file)+' MBs.\n')
            f.close()

def general(direct,ripsdir,P):
    if P[0]!='.':
        print('----------')
        print(P)
        listpac=os.listdir(direct+'/'+P)
        listpac.sort()
        try:
            os.mkdir(ripsdir+'/'+P)
            for j in range(0,len(listpac)):
                ripsing(listpac[j],P,direct,ripsdir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Directory for Patient '+ ripsdir+'/'+P+' already exists!')
            else:
                raise

direct='/home/HospitalXLandmarks_CD10203845/NonRelapse'
ripsdir='/home/HospitalXLandmarks_CD10203845_RIPS/NonRelapse'

os.mkdir(ripsdir)
listdir=os.listdir(direct)
listdir.sort()
num_cores=multiprocessing.cpu_count()
Parallel(n_jobs=num_cores)(delayed(general)(direct,ripsdir,listdir[j]) for j in range(0,len(listdir)))





# Dimension 2






import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from ripser import ripser
from persim import plot_diagrams
from numpy import loadtxt
import os
import errno
import multiprocessing
from joblib import Parallel, delayed

def ripsing(tube,pacient,direct,ripsdir):
    if tube[0]!='.' and tube.split('.')[-2][-6:]!='Others':
        print(tube)
        file_name=direct+'/'+pacient+'/'+tube
        file_rips=ripsdir+'/'+pacient+'/'+tube
        file_stats = os.stat(file_name)
        extension=file_name.split('.')[-1]
        MB_file=file_stats.st_size / (1000000)
        if not os.path.exists(file_rips+'_Rips0.txt'):
            if MB_file<=1.5:
                if extension=='txt':
                    loadfile=loadtxt(file_name, comments="#", delimiter=" ", unpack=False)
                elif extension=='csv':
                    loadfile=pd.read_csv(file_name).drop(columns='Unnamed: 0')
                else:
                    print('Problem reading.')
                m=1000 #Number of selected rows
                loadfile=loadfile.iloc[0:m,:];
                print('Begin Rips')
                start_time = time.time()
                ripsfile=ripser(loadfile,maxdim=2)['dgms']
                print("--- %s seconds ---" % (time.time() - start_time))
                #SAVED RIPS
                f = open(file_rips+'_Rips0.txt', "w")
                np.savetxt(f, ripsfile[0], fmt='%f')
                f.close()
                f = open(file_rips+'_Rips1.txt', "w")
                np.savetxt(f, ripsfile[1], fmt='%f')
                f.close()
                f = open(file_rips+'_Rips2.txt', "w")
                np.savetxt(f, ripsfile[2], fmt='%f')
                f.close()
            else:
                print('Not in range(0,2): Bigger than '+str(MB_file)+' MBs')
                f = open('ListR.txt',"a")
                f.write(pacient+'/'+tube+' NOT read. Size>'+str(MB_file)+' MBs.\n')
                f.close()
        else:
            print('Tube '+pacient+'/'+tube+' already ripsed!')
            f = open('ListR.txt',"a")
            f.write(pacient+'/'+tube+' already ripsed. Size:'+str(MB_file)+' MBs.\n')
            f.close()

def general(direct,ripsdir,P):
    if P[0]!='.':
        print('----------')
        print(P)
        listpac=os.listdir(direct+'/'+P)
        listpac.sort()
        try:
            os.mkdir(ripsdir+'/'+P)
            for j in range(0,len(listpac)):
                ripsing(listpac[j],P,direct,ripsdir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Directory for Patient '+ ripsdir+'/'+P+' already exists!')
            else:
                raise

direct='/home/HospitalXLandmarks_CD10203845/NonRelapse'
ripsdir='/home/HospitalXLandmarks_CD10203845_RIPS2/NonRelapse'

listdir=os.listdir(direct)
listdir.sort()
num_cores=multiprocessing.cpu_count()
Parallel(n_jobs=num_cores)(delayed(general)(direct,ripsdir,listdir[j]) for j in range(0,len(listdir)))





############### STEP 7: Statistical Analysis of Biomarkers CD10-20-38-45 ###############




import pandas as pd
import os

def read_and_save(filedir,tube):
    if tube[0]!='.':
        file_name, file_extension = os.path.splitext(filedir+'/'+tube)
        tubename=tube.split('_')[0].split('.')[0] #TubeX "name"
        tubenamerips=tube.split('_')[-1].split('.')[0] #Feature from Tube X: Rips 0 or 1
        if file_extension != '.pdf' and tubenamerips=='Rips0':
            Rips0=np.array(pd.read_csv(filedir+'/'+'_'.join(tube.split('_')[:-1])+'_Rips0.txt', sep=" ", header=None))
            if np.isinf(Rips0[-1,1]):
                Rips0=Rips0[:-1]
            Rips1=np.array(pd.read_csv(filedir+'/'+'_'.join(tube.split('_')[:-1])+'_Rips1.txt', sep=" ", header=None))
            if np.isnan(Rips1[-1,1]):
                Rips1[-1,1]=0
            Rips=[Rips0,Rips1]
            Biglist=[None]*2
            isinf=[False]*2
            for j in range(0,len(Rips)):
                Persistlist=[None]*len(Rips[j])
                for i in range(0,len(Rips[j])):
                    if not np.isinf(Rips[j][i][1]) and not np.isinf(Rips[j][i][0]):
                        Persistlist[i]=(Rips[j][i][1]-Rips[j][i][0])
                    else:
                        isinf[j]=True
                        Persistlist.remove(None)
                Biglist[j]=Persistlist
            return [Rips,Biglist,isinf]
        else:
            return []

#Reading Dimension 2
        
def read_and_save2(filedir,tube):
    if tube[0]!='.':
        file_name, file_extension = os.path.splitext(filedir+'/'+tube)
        tubename=tube.split('_')[0].split('.')[0]
        tubenamerips=tube.split('_')[-1].split('.')[0] 
        if file_extension != '.pdf' and tubenamerips=='Rips2':
            Rips0=[]
            Rips1=[]
            Rips2=np.array(pd.read_csv(filedir+'/'+'_'.join(tube.split('_')[:-1])+'_Rips2.txt', sep=" ", header=None))
            if np.isnan(Rips2[-1,1]):
                Rips2[-1,1]=0
            Rips=Rips2
            Biglist=[None]*3
            isinf=[False]*3
            for j in range(0,len(Rips)):
                Persistlist=[None]*len(Rips[j])
                for i in range(0,len(Rips[j])):
                    if not np.isinf(Rips[j][i][1]) and not np.isinf(Rips[j][i][0]):
                        Persistlist[i]=(Rips[j][i][1]-Rips[j][i][0])
                    else:
                        isinf[j]=True
                        Persistlist.remove(None)
                Biglist[j]=Persistlist

            return Rips
        else:
            return []





import os
###### PERSISTENCE CURVES CONSTRUCTION for DIM 0
dpi=300
base='home/HospitalXLandmarks_CD10203845_RIPS'
markers='' #The option markers can be used to select specific pairwise combinations
direct=base+'/'+'Relapse'+markers
listdirR=os.listdir(direct)
listdirR=[x for x in listdirR if not x.startswith('.')]
RipsR=[[]]*len(listdirR)
listdirR.sort()
for j in range(0,len(listdirR)):
    if listdirR[j][0]!='.' :
        listpac=os.listdir(direct+'/'+listdirR[j])
        listpac.sort()
        Ripsaux=[[]]*len(listpac)
        for i in range(0,len(listpac)):
            Ripsaux[i]=read_and_save(direct+'/'+listdirR[j],listpac[i])
        RipsR[j]=[item for item in Ripsaux if len(item)!=0]

RipsR=[item for item in RipsR if len(item)!=0]
direct=base+'/'+'NonRelapse'+markers
listdirNR=os.listdir(direct)
listdirNR=[x for x in listdirNR if not x.startswith('.')]
listdirNR.sort()
RipsNR=[[]]*len(listdirNR)
for j in range(0,len(listdirNR)):
    if listdirNR[j][0]!='.':
        listpac=os.listdir(direct+'/'+listdirNR[j])
        listpac.sort()
        Ripsaux=[[]]*len(listpac)
        for i in range(0,len(listpac)):
            Ripsaux[i]=read_and_save(direct+'/'+listdirNR[j],listpac[i])
        RipsNR[j]=[item for item in Ripsaux if len(item)!=0]

        
RipsNR=[item for item in RipsNR if len(item)!=0]
auxR0=[np.array(RipsR[i][0][1][0]) for i in range(0,len(RipsR))]
auxNR0=[np.array(RipsNR[i][0][1][0]) for i in range(0,len(RipsNR))]
auxR1=[np.array(RipsR[i][0][1][1]) for i in range(0,len(RipsR))]
auxNR1=[np.array(RipsNR[i][0][1][1])for i in range(0,len(RipsNR))]
########
span_threshold=np.linspace(lim_down_0,lim_up_0,200)
beta0_NR=[[]]*len(auxNR0)
span0_NR=[[]]*len(span_threshold)
beta0_R=[[]]*len(auxR0)
span0_R=[[]]*len(span_threshold)
for j in range(0,len(span_threshold)):
    for i in range(0,len(auxNR0)):
        beta0_NR[i]=(len(auxNR0[i][auxNR0[i]>span_threshold[j]])+1)
    span0_NR[j]=np.mean(beta0_NR)
    for i in range(0,len(auxR0)):
        beta0_R[i]=(len(auxR0[i][auxR0[i]>span_threshold[j]])+1)
    span0_R[j]=np.mean(beta0_R)
fig1=plt.figure(figsize=(3,3),dpi=dpi)
plt.plot(span_threshold,span0_NR,span_threshold,span0_R)
plt.xticks(fontname=fontname, fontsize=fontsize)
plt.yticks(fontname=fontname, fontsize=fontsize)
plt.grid(False)
g1 = span0_NR
g2 = span0_R
t, p = stats.ttest_ind(g1, g2)
print(t,p)

plt.show()



import os
import os
markers=''
###### PERSISTENCE CURVES CONSTRUCTION for DIM 1
dpi=300
markers=''
######
direct=base+'/'+'Relapse'+markers
listdirR=os.listdir(direct)
listdirR=[x for x in listdirR if not x.startswith('.')]
RipsR=[[]]*len(listdirR)
listdirR.sort()
for j in range(0,len(listdirR)):
    if listdirR[j][0]!='.' and listdirR[j][0:3]=="HMU":
        listpac=os.listdir(direct+'/'+listdirR[j])
        listpac.sort()
        Ripsaux=[[]]*len(listpac)
        for i in range(0,len(listpac)):
            Ripsaux[i]=read_and_save(direct+'/'+listdirR[j],listpac[i])
        RipsR[j]=[item for item in Ripsaux if len(item)!=0]
        
RipsR=[item for item in RipsR if len(item)!=0]
direct=base+'/'+'NonRelapse'+markers
listdirNR=os.listdir(direct)
listdirNR=[x for x in listdirNR if not x.startswith('.')]
listdirNR.sort()
RipsNR=[[]]*len(listdirNR)
for j in range(0,len(listdirNR)):
    if listdirNR[j][0]!='.' and listdirNR[j][0:3]=="HMU":
        listpac=os.listdir(direct+'/'+listdirNR[j])
        listpac.sort()
        Ripsaux=[[]]*len(listpac)
        for i in range(0,len(listpac)):
            Ripsaux[i]=read_and_save(direct+'/'+listdirNR[j],listpac[i])
        RipsNR[j]=[item for item in Ripsaux if len(item)!=0]
        
        
RipsNR=[item for item in RipsNR if len(item)!=0]
auxR0=[np.array(RipsR[i][0][1][0]) for i in range(0,len(RipsR))]
auxNR0=[np.array(RipsNR[i][0][1][0]) for i in range(0,len(RipsNR))]
auxR1=[np.array(RipsR[i][0][1][1]) for i in range(0,len(RipsR))]
auxNR1=[np.array(RipsNR[i][0][1][1])for i in range(0,len(RipsNR))]
################
span_threshold=np.linspace(lim_down_1,lim_up_1,200)
beta1_NR=[[]]*len(auxNR1)
span1_NR=[[]]*len(span_threshold)
beta1_R=[[]]*len(auxR0)
span1_R=[[]]*len(span_threshold)
for j in range(0,len(span_threshold)):
    for i in range(0,len(auxNR1)):
        beta1_NR[i]=(len(auxNR1[i][auxNR1[i]>span_threshold[j]]))#/len(auxNR0[i])*100

    span1_NR[j]=np.mean(beta1_NR)
    for i in range(0,len(auxR1)):
        beta1_R[i]=(len(auxR1[i][auxR1[i]>span_threshold[j]]))#/len(auxR0[i])*100

    span1_R[j]=np.mean(beta1_R)
fig1=plt.figure(figsize=(3,3),dpi=dpi)
plt.plot(span_threshold,span1_NR,span_threshold,span1_R)
plt.xticks(fontname=fontname, fontsize=fontsize)
plt.yticks(fontname=fontname, fontsize=fontsize)
plt.grid(False)
g1 = span1_NR
g2 = span1_R
t, p = stats.ttest_ind(g1, g2)
plt.show()




#STATISTICAL DIFFERENCE IN MAX, MIN, MEAN, STD in each dimension
import seaborn as sns
df0=pd.DataFrame([
                [np.max(auxR0[i]) for i in range(len(auxR0))],
                [np.min(auxR0[i]) for i in range(len(auxR0))],
                [np.mean(auxR0[i]) for i in range(len(auxR0))],
                [np.std(auxR0[i]) for i in range(len(auxR0))],
                [np.max(auxNR0[i]) for i in range(len(auxNR0))], 
                [np.min(auxNR0[i]) for i in range(len(auxNR0))],
                [np.mean(auxNR0[i]) for i in range(len(auxNR0))],
                [np.std(auxNR0[i]) for i in range(len(auxNR0))]
                ])
df0=df0.transpose()
df0.columns=['maxR','minR','meanR','stdR','maxNR','minNR','meanNR','stdNR']
df1=pd.DataFrame([
                [np.max(auxR1[i]) for i in range(len(auxR1))],
                [np.min(auxR1[i]) for i in range(len(auxR1))],
                [np.mean(auxR1[i]) for i in range(len(auxR1))],
                [np.std(auxR1[i]) for i in range(len(auxR1))],
                [np.max(auxNR1[i]) for i in range(len(auxNR1))], 
                [np.min(auxNR1[i]) for i in range(len(auxNR1))],
                [np.mean(auxNR1[i]) for i in range(len(auxNR1))],
                [np.std(auxNR1[i]) for i in range(len(auxNR1))]
                ])
df1=df1.transpose()
df1.columns=['maxR','minR','meanR','stdR','maxNR','minNR','meanNR','stdNR']

fig3, axes = plt.subplots(2, 4, dpi=300,figsize=(11,6))
fig3.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5,hspace=0.3)

features=['maxNR','maxR','Max. Persistence']
ax=sns.boxplot(data=[df0[features[0]],df0[features[1]]], linewidth=2.5,ax=axes[0][0])
ax.set(ylabel=features[2],xticklabels=['NR','R'])
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)

features=['minNR','minR','Min. Persistence']
ax=sns.boxplot(data=[df0[features[0]],df0[features[1]]], linewidth=2.5,ax=axes[0][1])
ax.set(ylabel=features[2],xticklabels=['NR','R'])
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)

features=['meanNR','meanR','Mean Persistence']
ax=sns.boxplot(data=[df0[features[0]],df0[features[1]]], linewidth=2.5,ax=axes[0][2])
ax.set(ylabel=features[2],xticklabels=['NR','R'])
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)

features=['stdNR','stdR','Std. Persistence']
ax=sns.boxplot(data=[df0[features[0]],df0[features[1]]], linewidth=2.5,ax=axes[0][3])
ax.set(ylabel=features[2],xticklabels=['NR','R'])
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)


features=['maxNR','maxR','Max. Persistence']
ax=sns.boxplot(data=[df1[features[0]],df1[features[1]]], linewidth=2.5,ax=axes[1][0])
ax.set(ylabel=features[2],xticklabels=['NR','R'])
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)

features=['minNR','minR','Min. Persistence']
ax=sns.boxplot(data=[df1[features[0]],df1[features[1]]], linewidth=2.5,ax=axes[1][1])
ax.set(ylabel=features[2],xticklabels=['NR','R'])
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)

features=['meanNR','meanR','Mean Persistence']
ax=sns.boxplot(data=[df1[features[0]],df1[features[1]]], linewidth=2.5,ax=axes[1][2])
ax.set(ylabel=features[2],xticklabels=['NR','R'])
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)

features=['stdNR','stdR','Std. Persistence']
ax=sns.boxplot(data=[df1[features[0]],df1[features[1]]], linewidth=2.5,ax=axes[1][3])
ax.set(ylabel=features[2],xticklabels=['NR','R'])
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)




from scipy import stats


items =[
    ['maxNR','maxR','Max. Persistence'],
    ['minNR','minR','Min. Persistence'],
    ['meanNR','meanR','Mean Persistence'],
    ['stdNR','stdR','Std. Persistence']
]
for features in items:
    g1 = df0[features[0]].values
    g1 = g1[~np.isnan(g1)]
    g2 = df0[features[1]].values
    g2 = g2[~np.isnan(g2)]
    t, p = stats.ttest_ind(g1, g2)
    print(t,p)



from scipy import stats


items =[
    ['maxNR','maxR','Max. Persistence'],
    ['minNR','minR','Min. Persistence'],
    ['meanNR','meanR','Mean Persistence'],
    ['stdNR','stdR','Std. Persistence']
]
for features in items:
    g1 = df1[features[0]].values
    g1 = g1[~np.isnan(g1)]
    g2 = df1[features[1]].values
    g2 = g2[~np.isnan(g2)]
    t, p = stats.ttest_ind(g1, g2)
    print(t,p)




## Iteration for construction of all pairwise combinations for markers CD10-20-38-45


fontsize=5

import os
import numpy as np
import matplotlib.pyplot as plt

lim_down_0=0.03
lim_up_0=0.07

lim_down_1=0.007
lim_up_1=0.015




dpi=300

pFrame=[]
tFrame=[]

from itertools import combinations
lista=list(combinations(['CD10','CD20','CD38','CD45'],2))
lista=['-'.join(item) for item in lista]

for markers in lista:
    print(markers)
    fig1,axs=plt.subplots(2,3,figsize=(3*2,2*2),dpi=dpi)

    direct=base+'/'+'Relapse/'+markers
    listdirR=os.listdir(direct)
    listdirR=[x for x in listdirR if not x.startswith('.')]
    RipsR=[[]]*len(listdirR)
    listdirR.sort()
    for j in range(0,len(listdirR)):
        if listdirR[j][0]!='.' and listdirR[j][0:3]!="HMU":
            listpac=os.listdir(direct+'/'+listdirR[j])
            listpac.sort()
            Ripsaux=[[]]*len(listpac)
            for i in range(0,len(listpac)):
                Ripsaux[i]=read_and_save(direct+'/'+listdirR[j],listpac[i])
            RipsR[j]=[item for item in Ripsaux if len(item)!=0]

    RipsR=[item for item in RipsR if len(item)!=0]
    direct=base+'/'+'NonRelapse/'+markers
    listdirNR=os.listdir(direct)
    listdirNR=[x for x in listdirNR if not x.startswith('.')]
    listdirNR.sort()
    RipsNR=[[]]*len(listdirNR)
    for j in range(0,len(listdirNR)):
        if listdirNR[j][0]!='.' and listdirNR[j][0:3]!="HMU":
            listpac=os.listdir(direct+'/'+listdirNR[j])
            listpac.sort()
            Ripsaux=[[]]*len(listpac)
            for i in range(0,len(listpac)):
                Ripsaux[i]=read_and_save(direct+'/'+listdirNR[j],listpac[i])
            RipsNR[j]=[item for item in Ripsaux if len(item)!=0]


    RipsNR=[item for item in RipsNR if len(item)!=0]
    auxR0=[np.array(RipsR[i][0][1][0]) for i in range(0,len(RipsR))]
    auxNR0=[np.array(RipsNR[i][0][1][0]) for i in range(0,len(RipsNR))]
    auxR1=[np.array(RipsR[i][0][1][1]) for i in range(0,len(RipsR))]
    auxNR1=[np.array(RipsNR[i][0][1][1])for i in range(0,len(RipsNR))]
    ########
    span_threshold=np.linspace(lim_down_0,lim_up_0,200)
    beta0_NR=[[]]*len(auxNR0)
    span0_NR=[[]]*len(span_threshold)
    beta0_R=[[]]*len(auxR0)
    span0_R=[[]]*len(span_threshold)
    for j in range(0,len(span_threshold)):
        for i in range(0,len(auxNR0)):
            beta0_NR[i]=(len(auxNR0[i][auxNR0[i]>span_threshold[j]])+1) #Include "/len(auxNR1[i])*100" for normalisation
        span0_NR[j]=np.mean(beta0_NR)
        for i in range(0,len(auxR0)):
            beta0_R[i]=(len(auxR0[i][auxR0[i]>span_threshold[j]])+1) #Include "/len(auxR0[i])*100" for normalisation
        span0_R[j]=np.mean(beta0_R)

    axs[0][0].plot(span_threshold,span0_NR,span_threshold,span0_R)
    axs[0][0].set_ylabel(r'Number (#) of bars > threshold',fontname=fontname, fontsize=fontsize)
    axs[0][0].set_xlabel('Threshold range',fontname=fontname, fontsize=fontsize)
    g1 = span0_NR
    g2 = span0_R
    t00, p00 = stats.ttest_ind(g1, g2)
    print(t00,p00)
    print("Dimension 0 Discovery")

    ########
    span_threshold=np.linspace(lim_down_1,lim_up_1,200)
    beta1_NR=[[]]*len(auxNR1)
    span1_NR=[[]]*len(span_threshold)
    beta1_R=[[]]*len(auxR0)
    span1_R=[[]]*len(span_threshold)
    for j in range(0,len(span_threshold)):
        for i in range(0,len(auxNR1)):
            beta1_NR[i]=(len(auxNR1[i][auxNR1[i]>span_threshold[j]])) #Include "/len(auxNR1[i])*100" for normalisation
        span1_NR[j]=np.mean(beta1_NR)
        for i in range(0,len(auxR1)):
            beta1_R[i]=(len(auxR1[i][auxR1[i]>span_threshold[j]])) #Include "/len(auxR1[i])*100" for normalisation
        span1_R[j]=np.mean(beta1_R)
    axs[1][0].plot(span_threshold,span1_NR,span_threshold,span1_R)
    axs[1][0].set_ylabel(r'Number (#) of bars > threshold',fontname=fontname, fontsize=fontsize)
    axs[1][0].set_xlabel('Threshold range',fontname=fontname, fontsize=fontsize)
    g1 = span1_NR
    g2 = span1_R
    t10, p10 = stats.ttest_ind(g1, g2)
    print(t10,p10)
    print("Dimension 1 Discovery")


    #############################

    ######
    direct=base+'/'+'Relapse/'+markers
    listdirR=os.listdir(direct)
    listdirR=[x for x in listdirR if not x.startswith('.')]
    RipsR=[[]]*len(listdirR)
    listdirR.sort()
    for j in range(0,len(listdirR)):
        if listdirR[j][0]!='.' and listdirR[j][0:3]=="HMU":
            listpac=os.listdir(direct+'/'+listdirR[j])
            listpac.sort()
            Ripsaux=[[]]*len(listpac)
            for i in range(0,len(listpac)):
                Ripsaux[i]=read_and_save(direct+'/'+listdirR[j],listpac[i])
            RipsR[j]=[item for item in Ripsaux if len(item)!=0]

    RipsR=[item for item in RipsR if len(item)!=0]
    direct=base+'/'+'NonRelapse/'+markers
    listdirNR=os.listdir(direct)
    listdirNR=[x for x in listdirNR if not x.startswith('.')]
    listdirNR.sort()
    RipsNR=[[]]*len(listdirNR)
    for j in range(0,len(listdirNR)):
        if listdirNR[j][0]!='.' and listdirNR[j][0:3]=="HMU":
            listpac=os.listdir(direct+'/'+listdirNR[j])
            listpac.sort()
            Ripsaux=[[]]*len(listpac)
            for i in range(0,len(listpac)):
                Ripsaux[i]=read_and_save(direct+'/'+listdirNR[j],listpac[i])
            RipsNR[j]=[item for item in Ripsaux if len(item)!=0]


    RipsNR=[item for item in RipsNR if len(item)!=0]
    auxR0=[np.array(RipsR[i][0][1][0]) for i in range(0,len(RipsR))]
    auxNR0=[np.array(RipsNR[i][0][1][0]) for i in range(0,len(RipsNR))]
    auxR1=[np.array(RipsR[i][0][1][1]) for i in range(0,len(RipsR))]
    auxNR1=[np.array(RipsNR[i][0][1][1])for i in range(0,len(RipsNR))]
    ########
    span_threshold=np.linspace(lim_down_0,lim_up_0,200)
    beta0_NR=[[]]*len(auxNR0)
    span0_NR=[[]]*len(span_threshold)
    beta0_R=[[]]*len(auxR0)
    span0_R=[[]]*len(span_threshold)
    for j in range(0,len(span_threshold)):
        for i in range(0,len(auxNR0)):
            beta0_NR[i]=(len(auxNR0[i][auxNR0[i]>span_threshold[j]])+1)  #Include "/len(auxNR0[i])*100" for normalisation
        span0_NR[j]=np.mean(beta0_NR)
        for i in range(0,len(auxR0)):
            beta0_R[i]=(len(auxR0[i][auxR0[i]>span_threshold[j]])+1) #Include "/len(auxR0[i])*100" for normalisation
        span0_R[j]=np.mean(beta0_R)
    axs[0][1].plot(span_threshold,span0_NR,span_threshold,span0_R)
    axs[0][1].set_ylabel(r'Number (#) of bars > threshold',fontname=fontname, fontsize=fontsize)
    axs[0][1].set_xlabel('Threshold range',fontname=fontname, fontsize=fontsize)
    g1 = span0_NR
    g2 = span0_R
    t01, p01 = stats.ttest_ind(g1, g2)
    print(t01,p01)
    print("Dimension 0 Validation")

    ########
    span_threshold=np.linspace(lim_down_1,lim_up_1,200)
    beta1_NR=[[]]*len(auxNR1)
    span1_NR=[[]]*len(span_threshold)
    beta1_R=[[]]*len(auxR0)
    span1_R=[[]]*len(span_threshold)
    for j in range(0,len(span_threshold)):
        for i in range(0,len(auxNR1)):
            beta1_NR[i]=(len(auxNR1[i][auxNR1[i]>span_threshold[j]])) #Include "/len(auxNR1[i])*100" for normalisation
        span1_NR[j]=np.mean(beta1_NR)
        for i in range(0,len(auxR1)):
            beta1_R[i]=(len(auxR1[i][auxR1[i]>span_threshold[j]])) #Include "/len(auxNR1[i])*100" for normalisation
        span1_R[j]=np.mean(beta1_R)
    axs[1][1].plot(span_threshold,span1_NR,span_threshold,span1_R)

    axs[1][1].set_ylabel(r'Number (#) of bars > threshold',fontname=fontname, fontsize=fontsize)
    axs[1][1].set_xlabel('Threshold range',fontname=fontname, fontsize=fontsize)
    g1 = span1_NR
    g2 = span1_R
    t11, p11 = stats.ttest_ind(g1, g2)
    print(t11,p11)
    print("Dimension 1 Validation")


    ############################# Disc and validation

    ######
    direct=base+'/'+'Relapse/'+markers
    listdirR=os.listdir(direct)
    listdirR=[x for x in listdirR if not x.startswith('.')]
    RipsR=[[]]*len(listdirR)
    listdirR.sort()
    for j in range(0,len(listdirR)):
        if listdirR[j][0]!='.':# and listdirR[j][0:3]=="HMU":
            listpac=os.listdir(direct+'/'+listdirR[j])
            listpac.sort()
            Ripsaux=[[]]*len(listpac)
            for i in range(0,len(listpac)):
                Ripsaux[i]=read_and_save(direct+'/'+listdirR[j],listpac[i])
            RipsR[j]=[item for item in Ripsaux if len(item)!=0]

    RipsR=[item for item in RipsR if len(item)!=0]
    direct=base+'/'+'NonRelapse/'+markers
    listdirNR=os.listdir(direct)
    listdirNR=[x for x in listdirNR if not x.startswith('.')]
    listdirNR.sort()
    RipsNR=[[]]*len(listdirNR)
    for j in range(0,len(listdirNR)):
        if listdirNR[j][0]!='.':
            listpac=os.listdir(direct+'/'+listdirNR[j])
            listpac.sort()
            Ripsaux=[[]]*len(listpac)
            for i in range(0,len(listpac)):
                Ripsaux[i]=read_and_save(direct+'/'+listdirNR[j],listpac[i])
            RipsNR[j]=[item for item in Ripsaux if len(item)!=0]


    RipsNR=[item for item in RipsNR if len(item)!=0]
    auxR0=[np.array(RipsR[i][0][1][0]) for i in range(0,len(RipsR))]
    auxNR0=[np.array(RipsNR[i][0][1][0]) for i in range(0,len(RipsNR))]
    auxR1=[np.array(RipsR[i][0][1][1]) for i in range(0,len(RipsR))]
    auxNR1=[np.array(RipsNR[i][0][1][1])for i in range(0,len(RipsNR))]
    ########
    span_threshold=np.linspace(lim_down_0,lim_up_0,200)
    beta0_NR=[[]]*len(auxNR0)
    span0_NR=[[]]*len(span_threshold)
    beta0_R=[[]]*len(auxR0)
    span0_R=[[]]*len(span_threshold)
    for j in range(0,len(span_threshold)):
        for i in range(0,len(auxNR0)):
            beta0_NR[i]=(len(auxNR0[i][auxNR0[i]>span_threshold[j]])+1) #Include "/len(auxNR0[i])*100" for normalisation
        span0_NR[j]=np.mean(beta0_NR)
        for i in range(0,len(auxR0)):
            beta0_R[i]=(len(auxR0[i][auxR0[i]>span_threshold[j]])+1) #Include "/len(auxR0[i])*100" for normalisation
        span0_R[j]=np.mean(beta0_R)
    axs[0][2].plot(span_threshold,span0_NR,span_threshold,span0_R)
    axs[0][2].set_ylabel(r'Number (#) of bars > threshold',fontname=fontname, fontsize=fontsize)
    axs[0][2].set_xlabel('Threshold range',fontname=fontname, fontsize=fontsize)
    g1 = span0_NR
    g2 = span0_R
    t02, p02 = stats.ttest_ind(g1, g2)
    print(t02,p02)
    print("Dimension 0 Disc and Validation")

    ########
    span_threshold=np.linspace(lim_down_1,lim_up_1,200)
    beta1_NR=[[]]*len(auxNR1)
    span1_NR=[[]]*len(span_threshold)
    beta1_R=[[]]*len(auxR0)
    span1_R=[[]]*len(span_threshold)
    for j in range(0,len(span_threshold)):
        for i in range(0,len(auxNR1)):
            beta1_NR[i]=(len(auxNR1[i][auxNR1[i]>span_threshold[j]])) #Include "/len(auxNR1[i])*100" for normalisation
        span1_NR[j]=np.mean(beta1_NR)
        for i in range(0,len(auxR1)):
            beta1_R[i]=(len(auxR1[i][auxR1[i]>span_threshold[j]])) #Include "/len(auxR1[i])*100" for normalisation
        span1_R[j]=np.mean(beta1_R)
    axs[1][2].plot(span_threshold,span1_NR,span_threshold,span1_R)
    axs[1][2].set_ylabel(r'Number (#) of bars > threshold',fontname=fontname, fontsize=fontsize)
    axs[1][2].set_xlabel('Threshold range',fontname=fontname, fontsize=fontsize)

    g1 = span1_NR
    g2 = span1_R
    t12, p12 = stats.ttest_ind(g1, g2)
    print(t12,p12)
    print("Dimension 1 Disc and Validation")

    for i in range(0,2):
        for j in range(0,3):
            axs[i][j].tick_params(labelsize=fontsize)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig1.savefig('/Users/salvador/Desktop/'+markers+'.png')
    
    pFrameaux=pd.DataFrame([{'Discovery_0':p00,'Validation_0':p01,'Discovery_Validation_0':p02,'Discovery_1':p10,'Validation_1':p11,'Discovery_Validation_1':p12}])
    tFrameaux=pd.DataFrame([{'Discovery_0':t00,'Validation_0':t01,'Discovery_Validation_0':t02,'Discovery_1':t10,'Validation_1':t11,'Discovery_Validation_1':t12}])
    if len(pFrame)==0:
        pFrame=pFrameaux
        tFrame=tFrameaux
        
    else:
        pFrame=pd.concat([pFrame,pFrameaux])
        tFrame=pd.concat([tFrame,tFrameaux])



pFrame.index=lista
tFrame.index=lista



import seaborn as sns
fig=plt.subplots(dpi=300)
sns.heatmap(pFrame,
                # cosmetics
                annot=True, vmin=0, vmax=0.15,center=0.05,
                cmap=sns.light_palette("seagreen", as_cmap=True).reversed(), linewidths=1, linecolor='black' ,cbar_kws={'orientation': 'vertical',"shrink": .85,'label': 'p-value'})
plt.xticks(rotation=60)



fig=plt.subplots(dpi=300)
sns.heatmap(tFrame,mask=pFrame>0.05,
            annot_kws={"style": "italic", "weight": "bold"},
                # cosmetics
                annot=True, vmin=-3, vmax=3,center=0,
                cmap=sns.color_palette("icefire_r"), linewidths=1, linecolor='black' ,cbar_kws={'orientation': 'vertical',"shrink": .85,'label': 't-test'})
plt.xticks(rotation=60)





############### STEP 8: Creation of Persistence Images ############### 



def read_and_save(filedir,tube):
    if tube[0]!='.':
        file_name, file_extension = os.path.splitext(filedir+'/'+tube)
        tubename=tube.split('_')[0].split('.')[0] #TubeX "name"
        tubenamerips=tube.split('_')[-1].split('.')[0] #Feature from Tube X: Rips 0 or 1
        if file_extension != '.pdf' and tubenamerips=='Rips0':
            Rips0=np.array(pd.read_csv(filedir+'/'+'_'.join(tube.split('_')[:-1])+'_Rips0.txt', sep=" ", header=None))
            if np.isinf(Rips0[-1,1]):
                Rips0=Rips0[:-1]
            Rips1=np.array(pd.read_csv(filedir+'/'+'_'.join(tube.split('_')[:-1])+'_Rips1.txt', sep=" ", header=None))
            if np.isnan(Rips1[-1,1]):
                Rips1[-1,1]=0
            Rips=[Rips0,Rips1]
            Biglist=[None]*2
            isinf=[False]*2
            for j in range(0,len(Rips)):
                Persistlist=[None]*len(Rips[j])
                for i in range(0,len(Rips[j])):
                    if not np.isinf(Rips[j][i][1]) and not np.isinf(Rips[j][i][0]):
                        Persistlist[i]=(Rips[j][i][1]-Rips[j][i][0])
                    else:
                        isinf[j]=True
                        Persistlist.remove(None)
                Biglist[j]=Persistlist
            return [Rips,Biglist,isinf]
        else:
            return []
        
def read_and_save2(filedir,tube):
    if tube[0]!='.':
        file_name, file_extension = os.path.splitext(filedir+'/'+tube)
        tubename=tube.split('_')[0].split('.')[0] 
        tubenamerips=tube.split('_')[-1].split('.')[0] 
        if file_extension != '.pdf' and tubenamerips=='Rips2':
            Rips0=[]
            Rips1=[]
            Rips2=np.array(pd.read_csv(filedir+'/'+'_'.join(tube.split('_')[:-1])+'_Rips2.txt', sep=" ", header=None))
            if np.isnan(Rips2[-1,1]):
                Rips2[-1,1]=0
            Rips=[Rips0,Rips1,Rips2]
            Biglist=[None]*3
            isinf=[False]*3
            for j in range(0,len(Rips)):
                Persistlist=[None]*len(Rips[j])
                for i in range(0,len(Rips[j])):
                    if not np.isinf(Rips[j][i][1]) and not np.isinf(Rips[j][i][0]):
                        Persistlist[i]=(Rips[j][i][1]-Rips[j][i][0])
                    else:
                        isinf[j]=True
                        Persistlist.remove(None)
                Biglist[j]=Persistlist
            return [Rips,Biglist,isinf]
        else:
            return []




from persim import PersImage
import pandas as pd

base='/home/HospitalXLandmarks_RIPS'

pixels_intervals=[5,10,25,50,100]
spread_intervals=[0.05,0.01]
dim_intervals=[2] #One can select the dimension to obtain the Persistence Images

for DIMENSION in dim_intervals:
    for p in pixels_intervals:
        pixels=[p,p]
        for spread in spread_intervals:
            direct=base+'/'+'NonRelapse'
            listdirNR=os.listdir(direct)
            listdirNR.sort()
            listdirNR=[x for x in listdirNR if not x.startswith('.')]
            RipsNR=[[]]*len(listdirNR)
            ImgsNR=[[]]*len(listdirNR)
            PimNR = PersImage(pixels=pixels, spread=spread)
           
            for j in range(0,len(listdirNR)):
                if listdirNR[j][0]!='.':
                    listpac=os.listdir(direct+'/'+listdirNR[j])
                    listpac.sort()
                    Ripsaux=[[]]*len(listpac)
                    for i in range(0,len(listpac)):
                        Ripsaux[i]=read_and_save2(direct+'/'+listdirNR[j],listpac[i])
                    RipsNR[j]=[item for item in Ripsaux if len(item)!=0]
                    ImgsNR[j]=PimNR.transform(RipsNR[j][0][0][DIMENSION])




            direct=base+'/'+'Relapse'
            listdirR=os.listdir(direct)
            listdirR.sort()
            listdirR=[x for x in listdirR if not x.startswith('.')]
            RipsR=[[]]*len(listdirR)
            ImgsR=[[]]*len(listdirR)
            PimR = PersImage(pixels=pixels, spread=spread)
            for j in range(0,len(listdirR)):

                if listdirR[j][0]!='.':
                    listpac=os.listdir(direct+'/'+listdirR[j])
                    listpac.sort()
                    Ripsaux=[[]]*len(listpac)
                    for i in range(0,len(listpac)):
                        Ripsaux[i]=read_and_save2(direct+'/'+listdirR[j],listpac[i])
                    RipsR[j]=[item for item in Ripsaux if len(item)!=0]
                    ImgsR[j]=PimR.transform(RipsR[j][0][0][DIMENSION])

            print('Done spread '+str(spread)+', pixels '+str(p))

            folder='PersistenceImages'+str(DIMENSION)
            dirR=base+'/'+folder+'/'+str(pixels[0])+'_'+str(spread)+'/'+'Relapse'
            dirNR=base+'/'+folder+'/'+str(pixels[0])+'_'+str(spread)+'/'+'NonRelapse'
            subfolder=base+'/'+folder+'/'+str(pixels[0])+'_'+str(spread)
            os.mkdir(subfolder)
            os.mkdir(subfolder+'/'+'Relapse')
            os.mkdir(subfolder+'/'+'NonRelapse')

            for i in range(0,len(ImgsR)):
                np.savetxt(subfolder+'/Relapse/'+listdirR[i]+'.csv',ImgsR[i])
            for i in range(0,len(ImgsNR)):               np.savetxt(subfolder+'/NonRelapse/'+listdirNR[i]+'.csv',ImgsNR[i])



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

base='/Users/salvador/Desktop/TODOSJUNTOS0.001_10203845_RIPSdim2'



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

base='/Users/salvador/Google Drive/OXFORD2021/TODOSJUNTOS0.001_10203845_RIPS'


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


base='/Users/salvador/Google Drive/OXFORD2021/TODOSJUNTOS0.001_10203845_RIPS'



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

