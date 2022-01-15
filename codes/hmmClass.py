# # # Graduation Thesis - HMM Class Definition
#
# This program include the construction of a HMM class. The members are 
# listed as follows:
#
# Member Data:
#   - obs: the observed sequence/time series
#   - numState: the number of hidden states
#   - matInit: the original probability distribution of hidden states
#   - matTrans: the state transition probability matrix
#   - matParam: the conditional distribution parameters of the observed 
#               sequence, corresponding to each hidden state
#   - matDist: the probability distribution matrix, corresponding to each 
#              observation at each state
#   - matEnd: the probability distribution of hidden states at the last 
#             observation
#
# Member Functions:
#   - __init__: built-in initialization function of the class
#   - forward: calculates the forward probability of observed sequence
#   - backward: calculates the backward probability of observed sequence
#   - expState: calculates the expectation of states
#   - expStateTrans: calculates the expected state transition probability
#   - EM: the major part of HMM parameter (including state transition 
#         probabilities, conditional distribution parameter, initial and 
#         final state distribution) estimation using Baum-Welch algorithm
#   - viterbi: global decoding to find the most probable sequence of 
#              hidden states using Viterbi algorithm

# # # # # Program Starts Here # # # # #
import numpy as np
import pandas as pd
from matplotlib.mlab import *

class HMM:
    def __init__(self,seqObserv,matTrans,matParam,matInit):
        '''HMM is the class defined to hold necessary data and functions 
        for a standard hidden Markov model. The initialization of a HMM 
        class requires 4 inputs given in the parameter list.
        
        Parameters
        ----------
        seqObserv : a data frame holding the stock return series
        matTrans : the initial state transition probability matrix
        matParam : the initial conditional distribution parameters 
                   corresponding to each hidden state
        matInit : the initial probability distribution of hidden states'''
        self.obs = seqObserv['ret'].copy()
        self.matTrans = matTrans.copy()
        self.matInit = matInit.copy()
        self.numState = matTrans.shape[0]
        self.matParam = matParam.copy()
        self.matDist = np.zeros([self.matTrans.shape[0],len(self.obs)])
        matDistTemp = pd.DataFrame(index = self.obs.index)
        for k in range(0,self.matTrans.shape[0]):
            matDistTemp['x' + str(k)] = normpdf(self.obs,self.matParam[0,k],self.matParam[1,k])
        self.matDist = np.array(matDistTemp.T)
    
    def forward(self):
        T = len(self.obs)
        alpha = np.zeros([self.numState,T])
        scale = np.zeros(T)
        alpha[:,0] = self.matInit[:] * self.matDist[:,0]
        scale[0] = np.sum(alpha[:,0])
        alpha[:,0] /= scale[0]
        for t in range(1,T):
            if np.sum(self.matDist[:,t]) != 0:
                alpha[:,t] = np.dot(alpha[:,t-1].T,self.matTrans).T *\
                             self.matDist[:,t]
                scale[t] = np.sum(alpha[:,t])
                alpha[:,t] /= scale[t]
            else:
                alpha[:,t] = alpha[:,t-1]
                scale[t] = scale[t-1] 
        logp = np.sum(np.log(scale[:]))
        return alpha, scale, logp
        
    def backward(self,scale):
        T = len(self.obs)
        beta = np.zeros([self.numState,T])
        beta[:,T-1] = 1/scale[T-1]
        for t in range(T-1,0,-1):
            beta[:,t-1] = np.dot(self.matTrans,
                                 (self.matDist[:,t]*beta[:,t]))
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
            denom = np.dot(np.dot(alpha[:,t].T, self.matTrans) * \
                    self.matDist[:,t+1].T,beta[:,t+1])
            for i in range(self.numState):
                numer = alpha[i,t] * self.matTrans[i,:] * \
                        self.matDist[:,t+1].T * beta[:,t+1].T
                xi[i,:,t] = numer / (denom + (denom == 0))
        return xi
        
    def EM(self):
        T = len(self.obs)
        criterion = 0.001
        alpha, scale, logp = self.forward()
        beta = self.backward(scale)            
        gamma = self.expState(alpha,beta)    
        xi = self.expStateTrans(alpha,beta)
        matDistTemp = self.matDist
        loop = 0
        while True:
            for i in range(0,self.numState):
                denominator = np.sum(gamma[i,0:T-1])
                for j in range(0,self.numState): 
                    numerator = np.sum(xi[i,j,0:T-1])
                    self.matTrans[i,j] = numerator / \
                                    (denominator + (denominator == 0))
            
            tempMu = np.array([np.sum(gamma*np.array(self.obs.T),1)/np.sum(gamma,1)])
            tempSigma = np.sqrt(np.array([np.sum(gamma*(np.array(self.obs.T) - tempMu.T)**2,1) / np.sum(gamma,1)]))
            self.matParam[0,:] = tempMu
            self.matParam[1,:] = tempSigma
            matDistTemp = pd.DataFrame(index = self.obs.index)
            for k in range(0,self.matTrans.shape[0]):
                matDistTemp['x' + str(k)] = normpdf(self.obs,self.matParam[0,k],self.matParam[1,k])
            matDistTemp = np.array(matDistTemp.T)
            
            self.matDist = matDistTemp
            self.matInit = gamma[:,0]
            self.matEnd = gamma[:,-1]
            
            alpha, scale, logpNew = self.forward()
            beta = self.backward(scale)            
            gamma = self.expState(alpha,beta)    
            xi = self.expStateTrans(alpha,beta)
            
            delta = abs(logpNew - logp)
            logp = logpNew
            loop += 1
            # print loop,logp
            if delta <= criterion:
                break
        
        self.valAIC = -2*logp + 2*(self.numState**2 + 2*self.numState - 1)
        self.valBIC = -2*logp + \
                      np.log(T)*(self.numState**2 + 2*self.numState - 1)
                
    def viterbi(self):
        T = len(self.obs)
        psi = np.zeros([self.numState,T])
        delta = np.zeros([self.numState,T])
        matState = np.zeros(T)
        temp = np.zeros([self.numState,self.numState])

        delta[:,0] = np.log(self.matInit) + np.log(self.matDist[:,0])
        for t in range(1,T):
            temp = (delta[:,t-1] + np.log(self.matTrans.T)).T
            ind = np.argmax(temp, axis = 0)
            psi[:,t] = ind
            delta[:,t] = np.log(self.matDist[:,t]) + \
                         temp[ind,range(self.numState)]
        
        max_ind = np.argmax(delta[:,T-1])
        matState[T-1] = max_ind
        for t in reversed(range(0,T-1)):
            matState[t] = psi[matState[t+1],t+1] 
        
        self.logProbState = delta[:,T-1][max_ind]
        self.matState = matState