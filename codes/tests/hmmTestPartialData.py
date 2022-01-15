# HMM Test - Baum Welch
import numpy as np
import pandas as pd
from matplotlib.mlab import *

# Fake Data
o = pd.DataFrame.from_csv('test.csv',index_col = False)
o = pd.DataFrame({'time': o['time'][1:],'ret': np.diff(np.log(o['close']))})
    
# HMM Object
class HMM:
    def __init__(self,seqObserv,matTrans,matParam,matInit):
        self.obs = seqObserv['ret']
        self.matTrans = matTrans
        self.matParam = matParam
        self.matDist = np.zeros([self.matTrans.shape[0],len(o)])
        matDistTemp = pd.DataFrame(index = self.obs.index)
        for k in range(0,self.matTrans.shape[0]):
            matDistTemp['x' + str(k)] = normpdf(self.obs,self.matParam[0,k],self.matParam[1,k])
        self.matDist = np.array(matDistTemp.T)
        self.matInit = matInit
        self.numState = matTrans.shape[0]
        self.numObserv = 3
        #for t in range(0,len(seqObserv)):
        #    if self.obs.ix[t,0] == 0:
        #        self.matDist[:,t] = np.array([0.3,0.2])
        #    elif self.obs.ix[t,0] == 1:
        #        self.matDist[:,t] = np.array([0.3,0.5])
        #    else:
        #        self.matDist[:,t] = np.array([0.4,0.3])
    
    def forward(self):
        T = len(self.obs)
        alpha = np.zeros([self.numState,T])
        scale = np.zeros(T)
        alpha[:,0] = self.matInit[:] * self.matDist[:,0]
        scale[0] = np.sum(alpha[:,0])
        alpha[:,0] /= scale[0]
        for t in range(1,T):
            if np.sum(self.matDist[:,t]) != 0:
                alpha[:,t] = np.dot(alpha[:,t-1].T, self.matTrans).T * self.matDist[:,t]
                scale[t] = np.sum(alpha[:,t])
                alpha[:,t] /= scale[t]
            else:
                alpha[:,t] = alpha[:,t-1]
                scale[t] = scale[t-1] 
                #print 'wrong at', t
                #break
            #print(t)
            #print(alpha[:,t])
        logp = np.sum(np.log(scale[:]))
        return alpha, scale, logp
        
    def backward(self,scale):
        T = len(self.obs)
        beta = np.zeros([self.numState,T])
        beta[:,T-1] = 1/scale[T-1]
        for t in range(T-1,0,-1):
            beta[:,t-1] = np.dot(self.matTrans, (self.matDist[:,t] * beta[:,t]))
            beta[:,t-1] /= scale[t-1]
        return beta
        
    def expState(self,alpha,beta):
        gamma = np.zeros(alpha.shape)
        gamma = alpha[:,:] * beta[:,:]
        gamma = gamma / ((np.sum(gamma,0) == 0) + np.sum(gamma,0))
        return gamma
        
    def expStateTrans(self,alpha,beta):
        T = len(self.obs)
        xi = np.zeros((self.numState,self.numState,T-1))   
        for t in range(T-1):
            denom = np.dot(np.dot(alpha[:,t].T, self.matTrans) * self.matDist[:,t+1].T,beta[:,t+1])
            for i in range(self.numState):
                numer = alpha[i,t] * self.matTrans[i,:] * self.matDist[:,t+1].T * beta[:,t+1].T
                xi[i,:,t] = numer / (denom + (denom == 0))
        #for t in range(0,T-1):        
        #    for i in range(0,self.numState):
        #        for j in range(0,self.numState):
        #            xi[i,j,t] = alpha[i,t] * self.matTrans[i,j] * self.matDist[j,t+1] * beta[j,t+1]
        #    xi[:,:,t] /= np.sum(np.sum(xi[:,:,t],1),0)    
        return xi
        
    def EM(self):
        T = len(self.obs)
        criterion = 0.0001
        alpha, scale, logp = self.forward()
        beta = self.backward(scale)            
        gamma = self.expState(alpha,beta)    
        xi = self.expStateTrans(alpha,beta)
        matDistTemp = self.matDist
        matTransTemp = self.matTrans
        loop = 0
        while True:
            #matTransTemp = np.sum(xi,2)/np.sum(np.sum(xi,2),1)
            for i in range(0,self.numState):
                denominator = np.sum(gamma[i,0:T-1])
                for j in range(0,self.numState): 
                    numerator = np.sum(xi[i,j,0:T-1])
                    self.matTrans[i,j] = numerator / (denominator + (denominator == 0))
            
            #temp = np.zeros([self.numState,self.numObserv])
            #for j in range(0,self.numState):
            #    denominator = np.sum(gamma[j,:])
            #    for k in range(0,self.numObserv):
            #        numerator = 0.0
            #        for t in range(0,T):
            #            if self.obs.ix[t,0] == k:
            #                numerator += gamma[j,t]
            #        temp[j,k] = numerator / denominator
            #
            #for t in range(0,T):
            #    matDistTemp[:,t] = temp[:,self.obs.ix[t,0]]
            
            tempMu = np.array([np.sum(gamma*np.array(self.obs.T),1)/np.sum(gamma,1)])
            #tempSigma = np.sqrt(np.array([np.sum(gamma*(np.repeat(np.array(self.obs.T),self.numState,axis = 0) - tempMu.T)**2,1) / np.sum(gamma,1)]))
            tempSigma = np.sqrt(np.array([np.sum(gamma*(np.array(self.obs.T) - tempMu.T)**2,1) / np.sum(gamma,1)]))
            self.matParam[0,:] = tempMu
            self.matParam[1,:] = tempSigma
            matDistTemp = pd.DataFrame(index = self.obs.index)
            for k in range(0,self.matTrans.shape[0]):
                matDistTemp['x' + str(k)] = normpdf(self.obs,self.matParam[0,k],self.matParam[1,k])
            matDistTemp = np.array(matDistTemp.T)
            
            #self.matTrans = matTransTemp
            self.matDist = matDistTemp
            self.matInit = gamma[:,0]
            self.matEnd = gamma[:,-1]
            
            alpha, scale, logpNew = self.forward()
            beta = self.backward(scale)            
            gamma = self.expState(alpha,beta)    
            xi = self.expStateTrans(alpha,beta)
            
            delta = abs(logpNew - logp)
            #delta = max(np.max(abs(logpNew - logp)), \
            #            np.max(abs(gamma[:,0]-self.matInit)), \
            #            np.max(abs(matTransTemp-self.matTrans)), \
            #            np.max(abs(matDistTemp-self.matDist)))
            #print logp,logpNew
            #print delta
            #print alpha, scale, beta
            #print gamma
            #print xi
            #break
            #self.matTrans = matTransTemp
            #self.matDist = matDistTemp
            #self.matInit = gamma[:,0]
            #self.matEnd = gamma[:,-1]
            logp = logpNew
            loop += 1
            print loop,logp
            if delta <= criterion:
                break

delta = np.array([0.333, 0.333, 0.333])
A = np.array([[0.333, 0.333, 0.333],
              [0.333, 0.333, 0.333],
              [0.333, 0.333, 0.333]])
#B = np.array([[0.3, 0.3, 0.4],
#              [0.2, 0.5, 0.3]])
              
param = np.array([[-1e-4,0,1e-4],
                  [0.001,0.005,0.01]])
#tempB = pd.DataFrame(index = o.index)
#for k in range(0,A.shape[0]):
#    tempB['x' + str(k)] = normpdf(o,model.matParam[0,k],model.matParam[1,k])

#B = np.zeros([A.shape[0],len(o)])
#for k in range(0,len(o)):
#    B[0,k] = 0.2 + float(o.ix[k])/10
#    B[1,k] = 0.3 + float(o.ix[k])/10

model = HMM(o,A,param,delta)
model.EM()
print model.matTrans
print model.matParam