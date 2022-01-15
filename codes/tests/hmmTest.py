# -*- coding: utf-8 -*-
import numpy as np

#######################

#-*-coding:utf-8-*-
def forward(N,M,A,B,P,observed):
    p = 0.0
    #观察到的状态数目
    LEN = len(observed)
    #中间概率LEN*M
    Q = [([0]*N) for i in range(LEN)]
    #第一个观察到的状态,状态的初始概率乘上隐藏状态到观察状态的条件概率。
    for j in range(N):
        Q[0][j] = P[j]*B[j][observation.index(observed[0])]
    #第一个之后的状态，首先从前一天的每个状态，转移到当前状态的概率求和，然后乘上隐藏状态到观察状态的条件概率。
    for i in range(1,LEN):
        for j in range(N):
            sum = 0.0
            for k in range(N):
                sum += Q[i-1][k]*A[k][j]
            Q[i][j] = sum * B[j][observation.index(observed[i])]
    for i in range(N):
        p += Q[LEN-1][i]
    return p
# 3 种隐藏层状态:sun cloud rain
hidden = []
hidden.append('sun')
hidden.append('cloud')
hidden.append('rain')
N = len(hidden)
# 4 种观察层状态:dry dryish damp soggy
observation = []
observation.append('dry')
observation.append('damp')
observation.append('soggy')
M = len(observation)
# 初始状态矩阵（1*N第一天是sun，cloud，rain的概率）
P = (0.3,0.3,0.4)
# 状态转移矩阵A（N*N 隐藏层状态之间互相转变的概率）
A=((0.2,0.3,0.5),(0.1,0.5,0.4),(0.6,0.1,0.3))
# 混淆矩阵B（N*M 隐藏层状态对应的观察层状态的概率）
B=((0.1,0.5,0.4),(0.2,0.4,0.4),(0.3,0.6,0.1))
#假设观察到一组序列为observed，输出HMM模型（N，M，A，B，P）产生观察序列observed的概率
observed = ['dry']
print forward(N,M,A,B,P,observed)
observed = ['damp']
print forward(N,M,A,B,P,observed)
observed = ['dry','damp']
print forward(N,M,A,B,P,observed)
observed = ['dry','damp','soggy','soggy','soggy','soggy','damp']
print forward(N,M,A,B,P,observed)

#######################


def viterbi(N,M,A,B,P,hidden,observed):
    sta = []
    LEN = len(observed)
    Q = [([0]*N) for i in range(LEN)]
    path = [([0]*N) for i in range(LEN)]
    #第一天计算，状态的初始概率*隐藏状态到观察状态的条件概率
    for j in range(N):
        Q[0][j]=P[j]*B[j][observation.index(observed[0])]
        path[0][j] = -1
    # 第一天以后的计算
    # 前一天的每个状态转移到当前状态的概率最大值
    # *
    # 隐藏状态到观察状态的条件概率
    for i in range(1,LEN):
        for j in range(N):
            max = 0.0
            index = 0
            for k in range(N):
                if(Q[i-1][k]*A[k][j] > max):
                    max = Q[i-1][k]*A[k][j]
                    index = k
            Q[i][j] = max * B[j][observation.index(observed[i])]
            path[i][j] = index
    #找到最后一天天气呈现哪种观察状态的概率最大
    max = 0.0
    idx = 0
    for i in range(N):
        if(Q[LEN-1][i]>max):
            max = Q[LEN-1][i]
            idx = i
    print "最可能隐藏序列的概率："+str(max)
    sta.append(hidden[idx])
    #逆推回去找到每天出现哪个隐藏状态的概率最大
    for i in range(LEN-1,0,-1):
        idx = path[i][idx]
        sta.append(hidden[idx])
    sta.reverse()
    return sta;
# 3 种隐藏层状态:sun cloud rain
hidden = []
hidden.append('sun')
hidden.append('cloud')
hidden.append('rain')
N = len(hidden)
# 4 种观察层状态:dry dryish damp soggy
observation = []
observation.append('dry')
observation.append('damp')
observation.append('soggy')
M = len(observation)
# 初始状态矩阵（1*N第一天是sun，cloud，rain的概率）
P = (0.3,0.3,0.4)
# 状态转移矩阵A（N*N 隐藏层状态之间互相转变的概率）
A=((0.2,0.3,0.5),(0.1,0.5,0.4),(0.6,0.1,0.3))
# 混淆矩阵B（N*M 隐藏层状态对应的观察层状态的概率）
B=((0.1,0.5,0.4),(0.2,0.4,0.4),(0.3,0.6,0.1))
#假设观察到一组序列为observed，输出HMM模型（N，M，A，B，P）产生观察序列observed的概率
observed = ['dry','damp','soggy']
print viterbi(N,M,A,B,P,hidden,observed)




#######################

DELTA = 0.001

class HMM:


    def __init__(self, pi, A, B):
        self.pi = pi
        self.A = A
        self.B = B
        self.M = B.shape[1]
        self.N = A.shape[0]
        
    def forward(self,obs):
        T = len(obs)
        N = self.N
        
        alpha = np.zeros([N,T])
        alpha[:,0] = self.pi[:] * self.B[:,obs[0]-1]                                                                                                      
    
        for t in xrange(1,T):
            for n in xrange(0,N):
                alpha[n,t] = np.sum(alpha[:,t-1] * self.A[:,n]) * self.B[n,obs[t]-1]
                     
        prob = np.sum(alpha[:,T-1])
        return prob, alpha
        
    def forward_with_scale(self, obs):
        """see scaling chapter in "A tutorial on hidden Markov models and 
        selected applications in speech recognition." 
        """
        T = len(obs)
        N = self.N
        alpha = np.zeros([N,T])
        scale = np.zeros(T)

        alpha[:,0] = self.pi[:] * self.B[:,obs[0]-1]
        scale[0] = np.sum(alpha[:,0])
        alpha[:,0] /= scale[0]

        for t in xrange(1,T):
            for n in xrange(0,N):
                alpha[n,t] = np.sum(alpha[:,t-1] * self.A[:,n]) * self.B[n,obs[t]-1]
            scale[t] = np.sum(alpha[:,t])
            alpha[:,t] /= scale[t]

        logprob = np.sum(np.log(scale[:]))
        return logprob, alpha, scale    
        
    def backward(self, obs):
        T = len(obs)
        N = self.N
        beta = np.zeros([N,T])
        
        beta[:,T-1] = 1
        for t in reversed(xrange(0,T-1)):
            for n in xrange(0,N):
                beta[n,t] = np.sum(self.B[:,obs[t+1]-1] * self.A[n,:] * beta[:,t+1])
                
        prob = np.sum(beta[:,0])
        return prob, beta

    def backward_with_scale(self, obs, scale):
        T = len(obs)
        N = self.N
        beta = np.zeros([N,T])

        beta[:,T-1] = 1 / scale[T-1]
        for t in reversed(xrange(0,T-1)):
            for n in xrange(0,N):
                beta[n,t] = np.sum(self.B[:,obs[t+1]-1] * self.A[n,:] * beta[:,t+1])
                beta[n,t] /= scale[t]
        
        return beta

    def viterbi(self, obs):
        T = len(obs)
        N = self.N
        psi = np.zeros([N,T]) # reverse pointer
        delta = np.zeros([N,T])
        q = np.zeros(T)
        temp = np.zeros(N)        
        
        delta[:,0] = self.pi[:] * self.B[:,obs[0]-1]    
        
        for t in xrange(1,T):
            for n in xrange(0,N):
                temp = delta[:,t-1] * self.A[:,n]    
                max_ind = argmax(temp)
                psi[n,t] = max_ind
                delta[n,t] = self.B[n,obs[t]-1] * temp[max_ind]

        max_ind = argmax(delta[:,T-1])
        q[T-1] = max_ind
        prob = delta[:,T-1][max_ind]

        for t in reversed(xrange(1,T-1)):
            q[t] = psi[q[t+1],t+1]    
            
        return prob, q, delta    
        
    def viterbi_log(self, obs):
        
        T = len(obs)
        N = self.N
        psi = np.zeros([N,T])
        delta = np.zeros([N,T])
        pi = np.zeros(self.pi.shape)
        A = np.zeros(self.A.shape)
        biot = np.zeros([N,T])

        pi = np.log(self.pi)        
        A = np.log(self.A)
        biot = np.log(self.B[:,obs[:]-1])

        delta[:,0] = pi[:] + biot[:,0]

        for t in xrange(1,T):
            for n in xrange(0,N):
                temp = delta[:,t-1] + self.A[:,n]    
                max_ind = argmax(temp)
                psi[n,t] = max_ind
                delta[n,t] = temp[max_ind] + biot[n,t]   

        max_ind = argmax(delta[:,T-1])
        q[T-1] = max_ind            
        logprob = delta[max_ind,T-1]      
                
        for t in reversed(xrange(1,T-1)):
            q[t] = psi[q[t+1],t+1]    

        return logprob, q, delta

    def baum_welch(self, obs):
        T = len(obs)
        M = self.M
        N = self.N        
        alpha = np.zeros([N,T])
        beta = np.zeros([N,T])
        scale = np.zeros(T)
        gamma = np.zeros([N,T])
        xi = np.zeros([N,N,T-1])
    
        # caculate initial parameters
        logprobprev, alpha, scale = self.forward_with_scale(obs)
        beta = self.backward_with_scale(obs, scale)            
        gamma = self.compute_gamma(alpha, beta)    
        xi = self.compute_xi(obs, alpha, beta)    
        logprobinit = logprobprev        
        
        # start interative 
        while True:
            # E-step
            self.pi = 0.001 + 0.999*gamma[:,0]
            for i in xrange(N):
                denominator = np.sum(gamma[i,0:T-1])
                for j in xrange(N): 
                    numerator = np.sum(xi[i,j,0:T-1])
                    self.A[i,j] = numerator / denominator
                                   
            self.A = 0.001 + 0.999*self.A
            for j in xrange(0,N):
                denominator = np.sum(gamma[j,:])
                for k in xrange(0,M):
                    numerator = 0.0
                    for t in xrange(0,T):
                        if obs[t]-1 == k:
                            numerator += gamma[j,t]
                    self.B[j,k] = numerator / denominator
            self.B = 0.001 + 0.999*self.B

            # M-step
            logprobcur, alpha, scale = self.forward_with_scale(obs)
            beta = self.backward_with_scale(obs, scale)            
            gamma = self.compute_gamma(alpha, beta)    
            xi = self.compute_xi(obs, alpha, beta)    

            delta = logprobcur - logprobprev
            logprobprev = logprobcur
            # print "delta is ", delta
            if delta <= DELTA:
                break     
                
        logprobfinal = logprobcur
        return logprobinit, logprobfinal                
            
    def compute_gamma(self, alpha, beta):
        gamma = np.zeros(alpha.shape)
        gamma = alpha[:,:] * beta[:,:]
        gamma = gamma / np.sum(gamma,0)
        return gamma
            
    def compute_xi(self, obs, alpha, beta):
        T = len(obs)
        N = self.N
        xi = np.zeros((N, N, T-1))
            
        for t in xrange(0,T-1):        
            for i in xrange(0,N):
                for j in xrange(0,N):
                    xi[i,j,t] = alpha[i,t] * self.A[i,j] * \
                                self.B[j,obs[t+1]-1] * beta[j,t+1]
            xi[:,:,t] /= np.sum(np.sum(xi[:,:,t],1),0)    
        return xi

def read_hmm(hmmfile):
    fhmm = open(hmmfile,'r') 

    M = int(fhmm.readline().split(' ')[1])
    N = int(fhmm.readline().split(' ')[1]) 
    
    A = np.array([])
    fhmm.readline()
    for i in xrange(N):
        line = fhmm.readline()
        if i == 0:
            A = np.array(map(float,line.split(',')))
        else:
            A = np.vstack((A,map(float,line.split(','))))
        
    B = np.array([])
    fhmm.readline()
    for i in xrange(N):
        line = fhmm.readline()
        if i == 0:
            B = np.array(map(float,line.split(',')))
        else:
            B = np.vstack((B,map(float,line.split(','))))
    
    fhmm.readline()
    line = fhmm.readline()
    pi = np.array(map(float,line.split(',')))
    
    fhmm.close()
    return M, N, pi, A, B 
    
def read_sequence(seqfile):
    fseq = open(seqfile,'r') 
    
    T = int(fseq.readline().split(' ')[1])
    line = fseq.readline()
    obs = np.array(map(int,line.split(',')))
    
    fseq.close()
    return T, obs