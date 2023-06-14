#!/usr/bin/env python
# coding: utf-8


############### STEP 3: VIETORIS-RIPS in each patient for all pairwise combinations of markers ###############





import numpy as np
import pandas as pd
import time
# Install ripser and persim from https://ripser.scikit-tda.org/en/latest/
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
    print("Loading RIPS in this file. Please consider that this step takes a lot of computing resources from your machine.")
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
                    print("File is too big. Please select landmarks in your file")
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



