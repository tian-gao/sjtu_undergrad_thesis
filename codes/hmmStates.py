# # # Graduation Thesis - Number of States 
#
# This program is very similar to the main one, except that is elminates 
# the part of prediction and focuses on evaluating the goodness of fit of 
# models with different number of hidden states. Notice that the analysis 
# is performed only for CSI300 daily data.
#
# Data files are import from .csv files and results will be stored in .csv
# format as well.
 
# # # # # Program Starts Here # # # # #

## Library Import
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from hmmClass import HMM

## Data Loading
dataPath = 'states'
dataPathMin = 240
dataRaw = pd.DataFrame.from_csv('daily/CSI300Data_daily.csv',
                                index_col = False)
dataRet = pd.DataFrame({'ret': np.diff(np.log(dataRaw['close'])),
                        'time': dataRaw['time'][1:]})
                        
## In-sample & Out-of-sample Definition
numMin = 240 / dataPathMin
numDay = len(dataRaw) / numMin
numStdDay = 5
lenInSample = int((numDay - numStdDay)*2/3)
lenOutSample = numDay - lenInSample - numStdDay
dataRetIn = dataRet.ix[0:(lenInSample + numStdDay)*numMin-1].copy()
dataRetIn.ix[:,'std'] = 0.0
for k in range(numStdDay*numMin,(lenInSample + numStdDay)*numMin):
    dataRetIn.ix[k,'std'] = np.std(dataRetIn.ix[(k-numStdDay*numMin):(k-1),
                                                'ret'])
dataRetIn = dataRetIn.tail(lenInSample*numMin)

## Loop for Different Number of States
seqAIC = pd.DataFrame({'number':range(2,7),'value':0},index = range(2,7))
seqBIC = pd.DataFrame({'number':range(2,7),'value':0},index = range(2,7))

for numState in range(2,7):
    print 'Number of States:', numState
    ## K-Means Clustering
    label = range(0,numState)
    modelKMeans = KMeans(n_clusters = numState).\
                         fit(dataRetIn[['ret','std']])
    order = np.argsort(modelKMeans.cluster_centers_.T[0,:])
    matParam = modelKMeans.cluster_centers_.T[:,order]
    dataRetIn['label'] = ''
    for k in range(0,len(dataRetIn)):
        dataRetIn.ix[dataRetIn.index[k],'label'] = label[np.where(order == modelKMeans.labels_[k])[0]]
    matInit = np.array((dataRetIn['label'].value_counts())[label]/\
            len(dataRetIn))
    matTrans = np.ones([numState,numState])/numState
    
    ## Entire Period Estimation
    dataRet = dataRet.ix[numStdDay*numMin:,]
    model = HMM(dataRet,matTrans,matParam,matInit)
    model.EM()
    seqAIC.ix[numState,'value'] = model.valAIC
    seqBIC.ix[numState,'value'] = model.valBIC
    
seqAIC.to_csv(dataPath + '/aic.csv',index = False,encoding = 'utf-8')
seqBIC.to_csv(dataPath + '/bic.csv',index = False,encoding = 'utf-8')