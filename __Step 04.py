#!/usr/bin/env python
# coding: utf-8



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


