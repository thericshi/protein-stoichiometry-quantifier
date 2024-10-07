# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 08:31:35 2021

@author: Tiya
"""

import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import rv_discrete
from scipy.special import binom
import random
random.seed=1
'''Negative binomial distribution'''

def q(N, h, theta):
    '''
    Computes q(N|h), the pmf for labelling ineffciency
    '''
    return binom(h, N) * (theta**N) * (1-theta)**(h-N)

class Nbinorm_gen(rv_discrete):
    def _pmf(self,B, lamda,N):
       lamda = lamda + 0*np.random.randn()
       return (np.exp(1/lamda)-1)**N * np.exp(-B/lamda) *binom(B-1,N-1)
   
NB = Nbinorm_gen(name='NB',a=1)

'''EM-algorithm for 3 clusters'''

class EM3:

    def __init__(self, X, pi=None, lam=None, status_callback=None):
        if pi is None:
            self.pi = [0, 0, 0]
        else:
            self.pi = pi

        if lam is None:
            self.lam = 0
        else:
            self.lam = lam

        self.iterations = 0
        self.X = X
        self.N = [1, 2, 3]
        self.AIC = 0
        self.pre_lam = 0
        self.LogL = 0
        self.status_callback = status_callback  # Status callback

        self.theta = 1

    def EMstep(self):
        """
        E-Step: Calculate the expectation (responsibility)
        """
        # Update lambda and increment iterations
        self.pre_lam = self.lam
        self.iterations += 1

        # Create responsibility array r
        r = np.zeros((len(self.X), 3))

        # Probability calculation for each data point belonging to a cluster
        for co, p in zip(range(3), self.pi):
            r[:, co] = p * NB.pmf(self.X, lamda=self.lam, N=self.N[co])

        # Normalize the probabilities
        for i in range(len(r)):
            r[i] = r[i] / (np.sum(self.pi) * np.sum(r, axis=1)[i])

        """ M-Step: Recalculate parameters """
        # Calculate pi
        m_c = []
        for c in range(len(r[0])):
            m = np.sum(r[:, c])
            m_c.append(m)

        for k in range(len(m_c)):
            self.pi[k] = (m_c[k] / np.sum(m_c))

        # Calculate lambda
        X = self.X
        self.lam = 1 / np.log((np.dot(r[:, 0], X) + np.dot(r[:, 1], X) + np.dot(r[:, 2], X)) / (
                np.dot(r[:, 0], X - self.N[0]) + np.dot(r[:, 1], X - self.N[1]) + np.dot(r[:, 2], X - self.N[2])))

        """ Log-likelihood and AIC calculation """
        self.LogL = np.sum(
            np.log(NB.pmf(self.X, lamda=self.lam, N=self.N[0]) * self.pi[0] +
                   NB.pmf(self.X, lamda=self.lam, N=self.N[1]) * self.pi[1] +
                   NB.pmf(self.X, lamda=self.lam, N=self.N[2]) * self.pi[2]))
        k = 3
        self.AIC = 2 * k - self.LogL

        print(self.iterations, self.lam)

        # Use the status callback to update the status
        if self.status_callback:
            self.status_callback(f"Iteration {self.iterations}: Lambda = {self.lam:.4f}, AIC = {self.AIC:.4f}")
        
    def initialize(self, lam_min=1, lam_max=6):
        pi_start = [[0.33,0.33,0.33],[0.33,0.33,0.33],[0.2,0.6,0.2],[0.6,0.2,0.2],[0.05,0.85,0.10],[0.85,0.10,0.05]]
        lam_initial = np.zeros(len(pi_start))
        pi1_initial = np.zeros(len(pi_start))
        pi2_initial = np.zeros(len(pi_start))
        pi3_initial = np.zeros(len(pi_start))   
        LogL_initial = np.zeros(len(pi_start))   
        for j in range(len(pi_start)):
            self.lam = lam_min+(random.random() * (lam_max - lam_min))
            self.pi = pi_start[j]
            while self.iterations<30:
                self.EMstep() 
            self.iterations = 0
            lam_initial[j] = self.lam
            pi1_initial[j] = self.pi[0]
            pi2_initial[j] = self.pi[1]
            pi3_initial[j] = self.pi[2]
            LogL_initial[j] = self.LogL
        print(LogL_initial)
        print(lam_initial)
        print(pi1_initial)
        print(pi2_initial)
        print(pi3_initial)
        best = np.nanargmin(-LogL_initial)
        self.pi = [pi1_initial[best],pi2_initial[best],pi3_initial[best]]
        self.lam = lam_initial[best]  
        print('best lam = ',self.lam)
        print('best pi = ',self.pi)
    
    def run(self, conv_lv=10e-4):
        """
        Run the EM algorithm until convergence
        """
        while not abs(self.lam - self.pre_lam) < conv_lv:
            self.EMstep()
            if self.status_callback:
                self.status_callback(f"Lambda difference = {abs(self.lam - self.pre_lam):.6f}")

    def gamma(self):
        product = self.pi[2] * ((q(1, 1, self.theta) * (q(2, 2, self.theta) - q(2, 3, self.theta))) 
                                + q(1, 2, self.theta) * q(2, 3, self.theta) - q(1, 3, self.theta) * q(2, 2, self.theta))
        return product + q(3, 3, self.theta) * (self.pi[1] * (q(1, 1, self.theta) - q(1, 2, self.theta)) + self.pi[0] * q(2, 2, self.theta))

    def apply_lab_ineff(self):
        g = self.gamma()
        conv_pi = [0, 0, 0]
        conv_pi[2] = (self.pi[2] * q(1, 1, self.theta) * q(2, 2, self.theta)) / g
        conv_pi[1] = (q(1, 1, self.theta) * (self.pi[1] * q(3, 3, self.theta) - self.pi[2] * q(2, 3, self.theta))) / g
        conv_pi[0] = 1 - conv_pi[1] - conv_pi[2]

        self.pi = conv_pi

class EM2:

    def __init__(self,X, pi=None, lam=None):
        if pi ==None:
            self.pi = [0,0]
        else:
            self.pi = pi
        
        if lam ==None:
            self.lam = 0
        else:
            self.lam = lam
        self.iterations = 0
        self.X = X
        self.N = [1,2]
        self.AIC = 0
        self.pre_lam = 0
        self.LogL = 0

        self.theta = 1

         
    def EMstep(self):
        """
        E-Step Calculate the expectation (responsibility)
        """
        #Update parameter
        self.pre_lam = self.lam
        self.iterations +=1
            
        #Create an array for responsibility r
        r = np.zeros((len(self.X),2))
        
        #Calculate the probability for each datapoint x_i to belong to each cluster     
        for co,p in zip(range(2),self.pi):
            r[:,co] = p*NB.pmf(self.X,lamda=self.lam,N=self.N[co])  
           
        #Normalize the proability
        for i in range(len(r)):
            r[i] = r[i]/(np.sum(self.pi)*np.sum(r,axis=1)[i])
            
        """M-Step"""
        """calculate lamda_max """
        # Calculate pi
        m_c = []
        for c in range(len(r[0])):
            m = np.sum(r[:,c])
            m_c.append(m)  
            
        for k in range(len(m_c)):
            self.pi[k] = (m_c[k]/np.sum(m_c)) 
            
        #Calculate lambda
        X = self.X            
        self.lam = 1/np.log((np.dot(r[:,0],X)+np.dot(r[:,1],X))/(np.dot(r[:,0],X-self.N[0])+np.dot(r[:,1],X-self.N[1])))             
         
        """Model selection"""
        """Calculate AIC/Log-likelihood"""    
        self.LogL = np.sum(np.log(NB.pmf(self.X,lamda=self.lam,N=self.N[0])*self.pi[0]+ NB.pmf(self.X,lamda=self.lam,N=self.N[1])*self.pi[1]))    
        k = 2
        self.AIC = 2*k-self.LogL

        print(self.iterations)
        
    def initialize(self, lam_min=2, lam_max=6):
        pi_start = [[0.5,0.5],[0.5,0.5],[0.2,0.8],[0.8,0.2,0.2],[0.05,0.95],[0.95,0.05]] # [0.8,0.2,0.2]??
        lam_initial = np.zeros(len(pi_start))
        pi1_initial = np.zeros(len(pi_start))
        pi2_initial = np.zeros(len(pi_start)) 
        LogL_initial = np.zeros(len(pi_start))   
        for j in range(len(pi_start)):
            self.lam = lam_min+(random.random() * (lam_max - lam_min))
            self.pi = pi_start[j]
            for self.iterations in range(35):
                self.EMstep() 
            self.iterations = 0
            lam_initial[j] = self.lam
            pi1_initial[j] = self.pi[0]
            pi2_initial[j] = self.pi[1]
            LogL_initial[j] = self.LogL
        print("all", pi1_initial)
        best = np.nanargmin(-LogL_initial)
        self.pi = [pi1_initial[best],pi2_initial[best]]
        self.lam = lam_initial[best]       
    
    def run(self,conv_lv=10e-4):
        while not abs(self.lam - self.pre_lam) <conv_lv:
            self.EMstep() 
    
    def apply_lab_ineff(self):
        print(q(1, 1, self.theta), q(1, 2, self.theta), q(2, 2, self.theta))
        self.pi[1] = (q(1, 1, self.theta)) / (q(1, 1, self.theta) - q(1, 2, self.theta) + (self.pi[0]/self.pi[1]) * (q(2, 2, self.theta)))
        self.pi[0] = 1 - self.pi[1]


class EM1:

    def __init__(self, X, pi=None, lam=None, status_callback=None):
        if pi is None:
            self.pi = [0]
        else:
            self.pi = pi

        if lam is None:
            self.lam = 0
        else:
            self.lam = lam

        self.iterations = 0
        self.X = X
        self.N = [1]
        self.AIC = 0
        self.pre_lam = 0
        self.LogL = 0
        self.status_callback = status_callback  # Status callback

    def EMstep(self):
        """
        E-Step: Calculate the expectation (responsibility)
        """
        # Update lambda and increment iterations
        self.pre_lam = self.lam
        self.iterations += 1

        # Create responsibility array r
        r = np.zeros((len(self.X), 1))

        # Probability calculation for each data point belonging to a cluster
        for co, p in zip(range(1), self.pi):
            r[:, co] = p * NB.pmf(self.X, lamda=self.lam, N=self.N[co])

        # Normalize the probabilities
        for i in range(len(r)):
            r[i] = r[i] / (np.sum(self.pi) * np.sum(r, axis=1)[i])

        """ M-Step: Recalculate parameters """
        # Calculate pi
        m_c = []
        for c in range(len(r[0])):
            m = np.sum(r[:, c])
            m_c.append(m)

        for k in range(len(m_c)):
            self.pi[k] = (m_c[k] / np.sum(m_c))

        # Calculate lambda
        X = self.X
        self.lam = 1 / np.log((np.dot(r[:, 0], X))/(np.dot(r[:, 0], X-self.N[0])))             

        """ Log-likelihood and AIC calculation """
        self.LogL = np.sum(
            np.log(NB.pmf(self.X, lamda=self.lam, N=self.N[0]) * self.pi[0]))
        k = 1
        self.AIC = 2 * k - self.LogL

        print(self.iterations)

        # Use the status callback to update the status
        if self.status_callback:
            self.status_callback(f"Iteration {self.iterations}: Lambda = {self.lam:.4f}, AIC = {self.AIC:.4f}")
        
    def initialize(self, lam_min=1, lam_max=6):
        pi_start = [[1]]
        lam_initial = np.zeros(len(pi_start))
        pi1_initial = np.zeros(len(pi_start)) 
        LogL_initial = np.zeros(len(pi_start))   
        for j in range(len(pi_start)):
            self.lam = lam_min+(random.random() * (lam_max - lam_min))
            self.pi = pi_start[j]
            while self.iterations<30:
                self.EMstep() 
            self.iterations = 0
            lam_initial[j] = self.lam
            pi1_initial[j] = self.pi[0]
            LogL_initial[j] = self.LogL
        print(LogL_initial)
        print(lam_initial)
        print(pi1_initial)
        best = np.nanargmin(-LogL_initial)
        self.pi = [pi1_initial[best]]
        self.lam = lam_initial[best] 
    
    def run(self, conv_lv=10e-5):
        """
        Run the EM algorithm until convergence
        """
        while not abs(self.lam - self.pre_lam) < conv_lv:
            self.EMstep()
            if self.status_callback:
                self.status_callback(f"Lambda difference = {abs(self.lam - self.pre_lam):.6f}")
       
#%%
'''___________________________Sample code_______________________________________________'''        
def BunchDyeSimple(x,pi1,pi2,pi3,size):
    '''Function for creating the mixture from the experimental data'''
    x1 = np.random.choice(x,int(pi1*size),replace=True) # generate the # blinks of monomer population 
    x2 = np.random.choice(x,int(pi2*size),replace=True)+np.random.choice(x,int(pi2*size),replace=True) # generate the # blinks of dimer population 
    x3 = np.random.choice(x,int(pi3*size),replace=True)+np.random.choice(x,int(pi3*size),replace=True)+np.random.choice(x,int(pi3*size),replace=True) # generate the # blinks of trimer population 
    xT = np.concatenate((x1,x2,x3), axis=None)
    return xT     

'''_______________________________________________________________________________________'''

if __name__ == "__main__":
    # Load and generate the experimental data
    Data = np.genfromtxt("nbBlinks_per_Dye_all.csv", delimiter=",")
    size = 1000
    pi = [0.2,0.8,0]

    Blinks = BunchDyeSimple(Data, pi[0], pi[1], pi[2], size)

    print(Data.shape, Blinks.shape)

    #%%
    # Run EM-algorithm for 3 distribution 
    # Initialize the algorithm with lambda between 2 and 5
    EM3_Blinks= EM3(Data)
    EM3_Blinks.initialize(lam_min=2, lam_max=5)
    EM3_Blinks.run()

    print(r'Estimated lambda=',EM3_Blinks.lam)
    print('Estimated pi=',EM3_Blinks.pi)
    print('AIC =',EM3_Blinks.AIC)
    
    #%% 
    # Run EM-algorithm for 2 distributions
    # Set initial guess for lambda and pi to be 2, and [0.3,0.7]
    EM2_Blinks= EM2(Data, pi=[0.3,0.7], lam=5)
    EM2_Blinks.initialize()
    EM2_Blinks.run()

    print(r'Estimated lambda=',EM2_Blinks.lam)
    print('Estimated pi=',EM2_Blinks.pi)
    print('AIC =',EM2_Blinks.AIC)
