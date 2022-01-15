# # # Graduation Thesis - Main
#
# The project is intended to construct a complete system for stock returns
# series prediction based on Hidden Markov Model. This program is the main 
# part of the project, realizing all aims of the empirical analysis part 
# along with other source code files.
#
# Source file lists
#   - hmmMain.py  : main part of the program
#   - hmmClass.py : include construction of the model, in the form of class
#   - hmmStates.py: a separate program to find the optimal number of hidden
#                   states
#
# Data files are import from .csv files and results will be stored in .csv
# format as well. Visualization analysis and presentation of the data will
# be realized in other programs.
 
# # # # # Program Starts Here # # # # #

## Library Import
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.cluster import KMeans
from hmmClass import HMM

## Data Loading
dataYear = ''
dataFreq = 'daily'
dataFreqMin = 240
#dataFile = '/SP500Data_' + dataFreq + dataYear + '.csv'
dataFile = '/CSI300Data_' + dataFreq + dataYear + '.csv'
dataRaw = pd.DataFrame.from_csv(dataFreq + dataFile,index_col = False)
dataRet = pd.DataFrame({'ret': np.diff(np.log(dataRaw['close'])),
                        'time': dataRaw['time'][1:]})
                        
## In-sample & Out-of-sample Definition
numMin = 240 / dataFreqMin
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

## K-Means Clustering
# K-Means is used to find the initial probability distribution of the three
# states and corresponding conditional distribution parameters. 
# The parameters are used for HMM initialization.
numState = 3
label = ['bear','intermediate','bull']
modelKMeans = KMeans(n_clusters = numState).fit(dataRetIn[['ret','std']])
order = np.argsort(modelKMeans.cluster_centers_.T[0,:])
matParam = modelKMeans.cluster_centers_.T[:,order]
dataRetIn['label'] = ''
for k in range(0,len(dataRetIn)):
    dataRetIn.ix[dataRetIn.index[k],'label'] = label[np.where(order == modelKMeans.labels_[k])[0]]
matInit = np.array((dataRetIn['label'].value_counts())[label]/\
          len(dataRetIn))
matTrans = np.ones([numState,numState])/numState
dataRetIn.to_csv(dataFreq + '/dataReturnInSample' + dataYear + '.csv',
    index = False,encoding='utf-8',columns=['time','ret','std','label'])
dataRetIn.to_csv(dataFreq + '/dataPredict' + dataYear + '.csv',
    index = False,encoding='utf-8',columns=['time','ret','std'])

## Entire Period Estimation
dataRet = dataRet.ix[numStdDay*numMin:,]
model = HMM(dataRet,matTrans,matParam,matInit)
model.EM()
model.viterbi()
dataRet['label'] = ''
for k in range(0,len(dataRet)):
    dataRet.ix[dataRet.index[k],'label'] = label[int(model.matState[k])]
dataRet.to_csv(dataFreq + '/dataHistory' + dataYear + '.csv',
    index = False,encoding = 'utf-8',columns = ['time','ret','label'])

## Initialization
dataPredict = pd.DataFrame({'time':dataRet['time'],'ret':0.0,'std':0.0},
                            index = dataRet.index)
seqTrans = pd.DataFrame({'x11':0.0,'x12':0.0,'x13':0.0,
                         'x21':0.0,'x22':0.0,'x23':0.0,
                         'x31':0.0,'x32':0.0,'x33':0.0},
                         index = dataRet.index)
seqParam = pd.DataFrame({'mu1':0.0,'mu2':0.0,'mu3':0.0,
                         'sigma1':0.0,'sigma2':0.0,'sigma3':0.0},
                         index = dataRet.index)
seqParam.ix[(lenInSample+numStdDay)*numMin-1,] = \
    matParam.reshape(1,2*numState)
                         
## Estimation & Prediction
print 'HMM Loop starts now!'
for k in range(0,lenOutSample*numMin):
    print dataRet.ix[(lenInSample+numStdDay)*numMin+k,'time']
    model = HMM(dataRet.ix[:(lenInSample+numStdDay)*numMin+k-1],
                matTrans,matParam,matInit)
    model.EM()
    #matTrans = model.matTrans.copy()
    #matParam = model.matParam.copy()
    #matInit = model.matInit.copy()
    seqTrans.ix[(lenInSample+numStdDay)*numMin+k,] = model.matTrans.reshape(1,model.numState**2)
    seqParam.ix[(lenInSample+numStdDay)*numMin+k,] = model.matParam.reshape(1,2*model.numState)
    dataPredict.ix[(lenInSample+numStdDay)*numMin+k,'ret'] = model.matEnd.dot(model.matTrans).dot(model.matParam[0,])
    dataPredict.ix[(lenInSample+numStdDay)*numMin+k,'std'] = sqrt(((model.matEnd.dot(model.matTrans))**2).dot(model.matParam[1,]**2))
    pd.DataFrame(dataPredict.ix[(lenInSample+numStdDay)*numMin+k]).T.\
        to_csv(dataFreq + '/dataPredict' + dataYear + '.csv',mode = 'a',
        index = False,encoding = 'utf-8',
        header = False,columns = ['time','ret','std'])

seqTrans['time'] = dataRet['time']
seqParam['time'] = dataRet['time']
seqTrans.to_csv(dataFreq + '/matTrans' + dataYear + '.csv',
                index = False,encoding = 'utf-8',
                columns = ['time','x11','x12','x13',
                                  'x21','x22','x23',
                                  'x31','x32','x33'])
seqParam.to_csv(dataFreq + '/matParam' + dataYear + '.csv',
                index = False,encoding = 'utf-8',
                columns = list(['time']) + list(np.core.defchararray.add(list(np.repeat('mu',3)),map(str,range(1,4)))) \
                                         + list(np.core.defchararray.add(list(np.repeat('sigma',3)),map(str,range(1,4)))))