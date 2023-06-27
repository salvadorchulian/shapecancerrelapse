#!/usr/bin/env python
# coding: utf-8



############### STEP 6: Vietoris-Rips of Biomarkers set CD10-20-38-45 ###############




# Important: From the previous Steps, please select only the parameters CD10-20-38-45 to perform these analysis.
# Here we will repeat Step 3.
# Important: we accounted here for dimension 2 as well, reducing the number of landmarks to 1000.




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

# This function performs the Vietoris-rips analysis (ripser) in each of the patient's data, looking for dimension 0 and 1
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
#Loops in the patients subfolders
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
#Parallelisation of the code: this is computational costy
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
# This function performs the Vietoris-rips analysis (ripser) in each of the patient's data, looking for dimension 2
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
                #Selects 1000 points
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
            
#Loops in the patients subfolders
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
#Parallelisation of the code: this is computational costy
Parallel(n_jobs=num_cores)(delayed(general)(direct,ripsdir,listdir[j]) for j in range(0,len(listdir)))

