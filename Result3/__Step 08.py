#!/usr/bin/env python
# coding: utf-8


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
import numpy as np
base='/home/HospitalXLandmarks_RIPS'

pixels_intervals=[5,10,25,50,100]
spread_intervals=[0.05,0.01]
dim_intervals=[2] #One can select the dimension to obtain the Persistence Images [0,1] or [2]

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

                 print("NR patient counter: "+str(j))


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
                 print("R patient counter: "+str(j))
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


