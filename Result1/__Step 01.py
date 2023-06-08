#!/usr/bin/env python
# coding: utf-8


############### STEP 1: Reading files ############### 



#Import basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import cytoflow as flow # Please install the following library https://cytoflow.readthedocs.io/en/stable/index.html

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




