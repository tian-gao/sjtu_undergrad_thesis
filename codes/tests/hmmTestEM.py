# HMM Test - Baum Welch
import numpy as np
import pandas as pd

# Fake Data
o = np.array([ 0.,  2.,  0.,  0.,  0.,  1.,  2.,  2.,  2.,  0.,  0.,  0.,  2.,
    2.,  2.,  2.,  2.,  2.,  2.,  2.,  0.,  0.,  0.,  0.,  0.,  0.,
    2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  0.,  0.,  0.,  2.,  2.,
    2.,  2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  2.,  2.,  2.,  2.,
    2.,  1.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
    2.,  2.,  2.,  0.,  0.,  0.,  0.,  2.,  0.,  1.,  0.,  0.,  1.,
    2.,  2.,  2.,  2.,  2.,  0.,  0.,  0.,  2.,  0.,  0.,  2.,  2.,
    2.,  0.,  2.,  1.,  0.,  0.,  0.,  2.,  2.,  2.,  2.,  2.,  0.,
    0.,  0.,  0.,  0.,  0.,  0.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
    0.,  1.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  1.,  2.,  2.,
    2.,  0.,  0.,  1.,  0.,  0.,  0.,  2.,  2.,  2.,  2.,  2.,  2.,
    2.,  2.,  2.,  2.,  0.,  0.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
    2.,  0.,  0.,  2.,  0.,  1.,  2.,  2.,  2.,  2.,  2.,  0.,  0.,
    0.,  0.,  0.,  0.,  0.,  0.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
    2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  0.,  0.,
    1.,  0.,  1.,  2.,  2.,  2.,  2.,  0.,  0.,  0.,  2.,  0.,  0.,
    0.,  2.,  0.,  0.,  0.,  0.,  0.,  0.,  2.,  2.,  2.,  2.,  2.,
    2.,  2.,  2.,  2.,  0.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
    0.,  0.,  0.,  2.,  2.,  2.,  2.,  1.,  0.,  0.,  0.,  0.,  0.,
    0.,  1.,  2.,  2.,  2.,  2.,  0.,  2.,  2.,  2.,  2.,  0.,  2.,
    0.,  2.,  0.,  0.,  0.,  0.,  1.,  2.,  0.,  0.,  0.,  1.,  0.,
    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
    0.,  0.,  0.,  0.,  1.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
    2.,  0.,  0.,  0.,  0.,  2.,  2.,  0.,  1.,  0.,  2.,  2.,  2.,
    2.,  2.,  2.,  2.,  2.,  2.,  0.,  2.,  2.,  2.,  2.,  2.,  2.,
    2.,  2.,  2.,  2.,  2.,  2.,  2.,  0.,  2.,  0.,  0.,  2.,  2.,
    2.,  1.,  0.,  0.,  0.,  2.,  2.,  2.,  2.,  2.,  0.,  0.,  0.,
    0.,  0.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  0.,  2.,  0.,  1.,
    2.,  2.,  2.,  0.,  0.,  0.,  2.,  0.,  2.,  2.,  2.,  2.,  2.,
    0.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
    2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
    0.,  2.,  0.,  0.,  0.,  0.,  2.,  1.,  0.,  0.,  0.,  0.,  0.,
    2.,  2.,  2.,  2.,  2.,  2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
    0.,  2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  2.,  0.,
    0.,  2.,  0.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
    2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
    2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
    2.,  2.,  2.,  0.,  0.,  0.,  0.,  2.,  2.,  2.,  2.,  2.,  2.,
    2.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,
    0.,  0.,  2.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  0.,  0.,  1.,
    2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  2.,  2.,  2.,  2.,
    2.,  2.,  0.,  1.,  2.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  2.,
    2.,  2.,  2.,  2.,  2.,  0.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
    2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  0.,  0.,  0.,
    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  2.,  0.,  0.,  2.,  2.,
    2.,  2.,  2.,  2.,  2.,  0.,  0.,  1.,  2.,  1.,  0.,  2.,  2.,
    2.,  2.,  2.,  0.,  2.,  2.,  0.,  0.,  0.,  0.,  0.,  2.,  1.,
    2.,  0.,  2.,  2.,  2.,  2.,  2.,  2.,  0.,  2.,  2.,  2.,  2.,
    2.,  2.,  2.,  0.,  2.,  2.,  2.,  2.,  2.,  2.,  0.,  0.,  2.,
    2.,  2.,  1.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,
    0.,  1.,  0.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
    2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  0.,  0.,  2.,
    1.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  2.,  2.,  2.,  2.,  0.,
    0.,  0.,  0.,  2.,  0.,  0.,  1.,  0.,  0.,  2.,  0.,  2.,  2.,
    2.,  2.,  0.,  0.,  0.,  1.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
    2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
    2.,  2.,  1.,  0.,  0.,  2.,  0.,  0.,  2.,  2.,  2.,  2.,  0.,
    0.,  0.,  1.,  0.,  1.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
    2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  0.,
    0.,  2.,  0.,  2.,  2.,  2.,  2.,  2.,  2.,  0.,  0.,  2.,  2.,
    2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  0.,  0.,  2.,
    0.,  1.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  0.,
    0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  2.,  0.,  0.,  0.,
    0.,  2.,  2.,  0.,  0.,  0.,  2.,  2.,  2.,  0.,  0.,  0.,  0.,
    1.,  0.,  2.,  0.,  0.,  0.,  0.,  1.,  0.,  2.,  0.,  0.,  0.,
    2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  2.,  0.,  2.,  2.,
    2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
    2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
    2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  1.,  0.,  0.,
    2.,  2.,  2.,  2.,  2.,  0.,  0.,  1.,  2.,  0.,  2.,  0.,  2.,
    0.,  0.,  2.,  2.,  2.,  2.,  2.,  2.,  0.,  0.,  2.,  2.,  2.,
    2.,  0.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  1.,
    0.,  0.,  0.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  0.,  2.,
    2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  0.,
    0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  2.,  2.,  0.,  0.,
    0.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
    2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.])
    
# HMM Object
class HMM:
    def __init__(self,seqObserv,matTrans,matDist,matInit):
        self.obs = seqObserv
        self.matTrans = matTrans
        self.matDist = matDist
        self.matInit = matInit
        self.numState = matTrans.shape[0]
        self.numObserv = matDist.shape[1]
    
    def forward(self):
        T = len(self.obs)
        alpha = np.zeros([self.numState,T])
        scale = np.zeros(T)
        alpha[:,0] = self.matInit[:] * self.matDist[:,self.obs[0]]
        scale[0] = np.sum(alpha[:,0])
        for t in range(1,T):
            alpha[:,t] = np.dot(alpha[:,t-1].T, self.matTrans).T * self.matDist[:,self.obs[t]]
            scale[t] = np.sum(alpha[:,t])
            alpha[:,t] /= scale[t]
        logp = np.sum(np.log(scale[:]))
        return alpha, scale, logp
        
    def backward(self,scale):
        T = len(self.obs)
        beta = np.zeros([self.numState,T])
        beta[:,T-1] = 1/scale[T-1]
        for t in range(T-1,0,-1):
            beta[:,t-1] = np.dot(self.matTrans, (self.matDist[:,self.obs[t]] * beta[:,t]))
            beta[:,t-1] /= scale[t-1]
        return beta
        
    def expState(self,alpha,beta):
        gamma = np.zeros(alpha.shape)
        gamma = alpha[:,:] * beta[:,:]
        gamma = gamma / np.sum(gamma,0)
        return gamma
        
    def expStateTrans(self,alpha,beta):
        T = len(self.obs)
        xi = np.zeros((self.numState,self.numState,T-1))   
        #for t in range(T-1):
        #    denom = np.dot(np.dot(alpha[:,t].T, self.matTrans) * self.matDist[:,self.obs[t+1]].T,beta[:,t+1])
        #    for i in range(self.numState):
        #        numer = alpha[i,t] * self.matTrans[i,:] * self.matDist[:,self.obs[t+1]].T * beta[:,t+1].T
        #        xi[i,:,t] = numer / denom 
        for t in range(0,T-1):        
            for i in range(0,self.numState):
                for j in range(0,self.numState):
                    xi[i,j,t] = alpha[i,t] * self.matTrans[i,j] * self.matDist[j,self.obs[t+1]] * beta[j,t+1]
            xi[:,:,t] /= np.sum(np.sum(xi[:,:,t],1),0)    
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
            matTransTemp = np.sum(xi,2)/np.sum(np.sum(xi,2),1)
            #for i in range(0,self.numState):
            #    denominator = np.sum(gamma[i,0:T-1])
            #    for j in range(0,self.numState): 
            #        numerator = np.sum(xi[i,j,0:T-1])
            #        self.matTrans[i,j] = numerator / denominator
            
            for j in range(0,self.numState):
                denominator = np.sum(gamma[j,:])
                for k in range(0,self.numObserv):
                    numerator = 0.0
                    for t in range(0,T):
                        if self.obs[t] == k:
                            numerator += gamma[j,t]
                    matDistTemp[j,k] = numerator / denominator
            
            alpha, scale, logpNew = self.forward()
            beta = self.backward(scale)            
            gamma = self.expState(alpha,beta)    
            xi = self.expStateTrans(alpha,beta)
            delta = max(np.max(abs(logpNew - logp)), \
                        np.max(abs(gamma[:,0]-self.matInit)), \
                        np.max(abs(matTransTemp-self.matTrans)), \
                        np.max(abs(matDistTemp-self.matDist)))
            
            self.matTrans = matTransTemp
            self.matDist = matDistTemp
            self.matInit = gamma[:,0]
            logp = logpNew
            loop += 1
            print loop,logp
            if delta <= criterion:
                break

delta = np.array([0.5, 0.5])
A = np.array([[0.5, 0.5],
              [0.5, 0.5]])
B = np.array([[0.3, 0.3, 0.4],
              [0.2, 0.5, 0.3]])
model = HMM(o,A,B,delta)
model.EM()
print model.matTrans
print model.matDist