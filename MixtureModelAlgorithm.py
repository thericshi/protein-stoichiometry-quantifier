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

def fp(B, lamda, N):
    """
    Computes the probability mass function (PMF) of the negative binomial distribution for a vector of values.

    Args:
        B: A NumPy array of the number of failures.
        lamda: The mean number of successes.
        N: The number of successes.

    Returns:
        A NumPy array of the PMF values corresponding to each element in B.
    """

    lamda = lamda + 0 * np.random.randn()  # Ensure lamda is a scalar
    return (np.exp(1/lamda)-1)**N * np.exp(-B/lamda) * binom(B-1, N-1)

class Nbinorm_gen(rv_discrete):
    def _pmf(self,B, lamda,N):
       lamda = lamda + 0*np.random.randn()
       return (np.exp(1/lamda)-1)**N * np.exp(-B/lamda) *binom(B-1,N-1)
   
NB = Nbinorm_gen(name='NB',a=1)

'''EM-algorithm for 3 clusters'''

class EM3:
    def __init__(self, X, pi=None, lam=None):
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
        self.theta = 0.75

        # Caching precomputations
        self.X_minus_N = [X - self.N[c] for c in range(3)]  # X - N[c] values

    def EMstep(self):
        """
        E-Step: Calculate the expectation (responsibility) using vectorization.
        M-Step: Update parameters and perform model selection.
        """
        # Update lambda from previous iteration
        self.pre_lam = self.lam
        self.iterations += 1
        
        # Precompute fp values for each cluster and data point
        fp_vals = np.array([fp(self.X, lamda=self.lam, N=self.N[c]) for c in range(3)])
        
        # E-Step: Calculate responsibility (r) matrix using broadcasting
        r = np.array(self.pi)[:, np.newaxis] * fp_vals  # Convert self.pi to np.array and broadcast
        
        # Normalize responsibilities across clusters
        r /= np.sum(r, axis=0, keepdims=True)  # Broadcasting to divide by the sum for each sample
        
        # Transpose r to make it (n_samples, 3) for further calculations
        r = r.T
        
        # M-Step: Update pi using vectorized operations
        m_c = np.sum(r, axis=0)  # Shape: (3,)
        self.pi = m_c / np.sum(m_c)
        
        # Calculate lambda using optimized dot products

        self.lam = 1 / np.log((np.dot(r[:, 0], self.X) + np.dot(r[:, 1], self.X) + np.dot(r[:, 2], self.X)) / (
            np.dot(r[:, 0], self.X_minus_N[0]) +
            np.dot(r[:, 1], self.X_minus_N[1]) +
            np.dot(r[:, 2], self.X_minus_N[2])
        ))
        """
        # Accelerate convergence by halving the difference between new_lam and pre_la
        
        lam_diff = self.lam - self.pre_lam
        if lam_diff < 0.1 and lam_diff > 10e-7:
            self.lam = self.lam + 1 * lam_diff  # Apply damping factor of 0.5 to adjust the update
        """

        
        # Log-likelihood using vectorized operation
        self.LogL = np.sum(np.log((
            fp_vals[0] * self.pi[0] + 
            fp_vals[1] * self.pi[1] + 
            fp_vals[2] * self.pi[2]
        )))
        
        # Calculate AIC (k = 3, number of parameters)
        self.AIC = 2 * 3 - self.LogL

    def initialize(self, lam_min=1, lam_max=6):
        pi_start = [[0.75, 0.20, 0.05],[0.60, 0.25, 0.15],[0.50, 0.35, 0.15],[0.85,0.10,0.05]]
        lam_initial = np.zeros(len(pi_start))
        pi1_initial = np.zeros(len(pi_start))
        pi2_initial = np.zeros(len(pi_start))
        pi3_initial = np.zeros(len(pi_start))   
        LogL_initial = np.zeros(len(pi_start))  
        self.lam = lam_min+(random.random() * (lam_max - lam_min))
 
        for j in range(len(pi_start)):
            self.pi = pi_start[j]
            while self.iterations<30:
                self.EMstep() 
            self.iterations = 0
            lam_initial[j] = self.lam
            pi1_initial[j] = self.pi[0]
            pi2_initial[j] = self.pi[1]
            pi3_initial[j] = self.pi[2]
            LogL_initial[j] = self.LogL
        best = np.nanargmin(-LogL_initial)
        self.pi = [pi1_initial[best],pi2_initial[best],pi3_initial[best]]
        self.lam = lam_initial[best]  
        # print('best lam = ',self.lam)
        # print('best pi = ',self.pi)

    def initialize_old(self, lam_min=1, lam_max=6):
        pi_start = [[0.33,0.33,0.33],[0.2,0.6,0.2],[0.6,0.2,0.2],[0.05,0.85,0.10],[0.85,0.10,0.05]]
        lam_initial = np.zeros(len(pi_start))
        pi1_initial = np.zeros(len(pi_start))
        pi2_initial = np.zeros(len(pi_start))
        pi3_initial = np.zeros(len(pi_start))   
        LogL_initial = np.zeros(len(pi_start))  
        self.lam = lam_min+(random.random() * (lam_max - lam_min))
 
        for j in range(len(pi_start)):
            self.pi = pi_start[j]
            while self.iterations<30:
                self.EMstep() 
            self.iterations = 0
            lam_initial[j] = self.lam
            pi1_initial[j] = self.pi[0]
            pi2_initial[j] = self.pi[1]
            pi3_initial[j] = self.pi[2]
            LogL_initial[j] = self.LogL
        best = np.nanargmin(-LogL_initial)
        self.pi = [pi1_initial[best],pi2_initial[best],pi3_initial[best]]
        self.lam = lam_initial[best]  
        # print('best lam = ',self.lam)
        # print('best pi = ',self.pi)
    
    def run(self,conv_lv=10e-5):
        while not abs(self.lam - self.pre_lam) <conv_lv:
            self.EMstep() 

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
    def __init__(self, X, pi=None, lam=None):
        if pi is None:
            self.pi = [0, 0]
        else:
            self.pi = pi
        
        if lam is None:
            self.lam = 0
        else:
            self.lam = lam

        self.iterations = 0
        self.X = X
        self.N = [1, 2]
        self.AIC = 0
        self.pre_lam = 0
        self.LogL = 0
        self.theta = 0.75

        # Caching precomputations
        self.X_minus_N = [X - self.N[c] for c in range(2)]  # X - N[c] values

    def EMstep(self):
        """
        E-Step: Calculate the expectation (responsibility) using vectorization.
        M-Step: Update parameters and perform model selection.
        """
        # Update lambda from previous iteration
        self.pre_lam = self.lam
        self.iterations += 1

        # Precompute fp values for each cluster and data point
        fp_vals = np.array([fp(self.X, lamda=self.lam, N=self.N[c]) for c in range(2)])

        # E-Step: Calculate responsibility (r) matrix using broadcasting
        r = np.array(self.pi)[:, np.newaxis] * fp_vals  # Convert self.pi to np.array and broadcast
        
        # Normalize responsibilities across clusters
        r /= np.sum(r, axis=0, keepdims=True)  # Broadcasting to divide by the sum for each sample

        # Transpose r to make it (n_samples, 2) for further calculations
        r = r.T

        # M-Step: Update pi using vectorized operations
        m_c = np.sum(r, axis=0)  # Shape: (2,)
        self.pi = m_c / np.sum(m_c)

        # Calculate lambda using optimized dot products
        self.lam = 1 / np.log((np.dot(r[:, 0], self.X) + np.dot(r[:, 1], self.X)) / (
            np.dot(r[:, 0], self.X_minus_N[0]) +
            np.dot(r[:, 1], self.X_minus_N[1])
        ))

        # Log-likelihood using vectorized operation
        self.LogL = np.sum(np.log((
            fp_vals[0] * self.pi[0] + 
            fp_vals[1] * self.pi[1]
        )))

        # Calculate AIC (k = 2, number of parameters)
        self.AIC = 2 * 2 - self.LogL

    def initialize(self, lam_min=1, lam_max=6):
        pi_start = [[0.75, 0.25], [0.60, 0.40], [0.50, 0.50], [0.85, 0.15]]
        lam_initial = np.zeros(len(pi_start))
        pi1_initial = np.zeros(len(pi_start))
        pi2_initial = np.zeros(len(pi_start))
        LogL_initial = np.zeros(len(pi_start))
        self.lam = lam_min + (random.random() * (lam_max - lam_min))

        for j in range(len(pi_start)):
            self.pi = pi_start[j]
            while self.iterations < 30:
                self.EMstep()
            self.iterations = 0
            lam_initial[j] = self.lam
            pi1_initial[j] = self.pi[0]
            pi2_initial[j] = self.pi[1]
            LogL_initial[j] = self.LogL
        best = np.nanargmin(-LogL_initial)
        self.pi = [pi1_initial[best], pi2_initial[best]]
        self.lam = lam_initial[best]
        # print('best lam = ', self.lam)
        # print('best pi = ', self.pi)

    def run(self, conv_lv=10e-5):
        while not abs(self.lam - self.pre_lam) < conv_lv:
            self.EMstep()

    def apply_lab_ineff(self):
        self.pi[1] = (q(1, 1, self.theta)) / (q(1, 1, self.theta) - q(1, 2, self.theta) + (self.pi[0] / self.pi[1]) * (q(2, 2, self.theta)))
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
        self.N = [1]  # Only one cluster
        self.AIC = 0
        self.pre_lam = 0
        self.LogL = 0
        self.status_callback = status_callback  # Status callback
        self.X_minus_N = X - self.N[0]  # Cache precomputed X - N[0]

    def EMstep(self):
        """
        E-Step: Calculate the expectation (responsibility)
        """
        # Update lambda and increment iterations
        self.pre_lam = self.lam
        self.iterations += 1

        r = np.ones((len(self.X), 1))

        """ M-Step: Recalculate parameters """
        # Calculate pi
        m_c = np.sum(r, axis=0, keepdims=True)

        self.pi = np.ones((1, 1))

        # Calculate lambda
        X = self.X
        self.lam = 1 / np.log((np.dot(r[:, 0], X))/(np.dot(r[:, 0], X-self.N[0])))             

        """ Log-likelihood and AIC calculation """
        self.LogL = np.sum(
            np.log(fp(self.X, lamda=self.lam, N=self.N[0]) * self.pi[0]))
        k = 1
        self.AIC = 2 * k - self.LogL

        print(self.iterations)

        # Use the status callback to update the status
        if self.status_callback:
            self.status_callback(f"Iteration {self.iterations}: Lambda = {self.lam:.4f}, AIC = {self.AIC:.4f}")

    def initialize(self, lam_min=1, lam_max=6):
        """
        Initialization of EM parameters with multiple starting values for pi and lambda
        """
        pi_start = [[1]]  # Only one cluster, so pi is always 1
        lam_initial = np.zeros(len(pi_start))
        pi1_initial = np.zeros(len(pi_start))
        LogL_initial = np.zeros(len(pi_start))

        for j in range(len(pi_start)):
            self.lam = lam_min + (random.random() * (lam_max - lam_min))
            self.pi = pi_start[j]
            while self.iterations < 30:
                self.EMstep()
            self.iterations = 0
            lam_initial[j] = self.lam
            pi1_initial[j] = self.pi[0]
            LogL_initial[j] = self.LogL

        best = np.nanargmin(-LogL_initial)
        self.pi = [pi1_initial[best]]
        self.lam = lam_initial[best]
        print('best lam = ', self.lam)
        print('best pi = ', self.pi)

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
