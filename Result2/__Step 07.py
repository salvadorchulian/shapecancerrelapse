#!/usr/bin/env python
# coding: utf-8




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



