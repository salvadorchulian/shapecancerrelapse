#!/usr/bin/env python
# coding: utf-8


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




