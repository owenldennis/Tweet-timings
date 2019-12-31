# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:48:56 2019
Classes for calculation and analysis of correlation measures for pointwise comparison of lagging time series
Class pairwise_stats uses 2019 pairwise correlation measure (Messager et al) to determined z-scores for two Bernoulli time series
Class tweet_data creates a set of time series and analyses correlation across all pairs

@author: owen
"""

import matplotlib.pyplot as plt
import random
import numpy as np
from numpy import corrcoef
import collections
import pandas as pd
import time

TEST_MATRIX =  [[0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]               

TEST_MATRIX1 = [[1,0,0,1,1,0,0,0,0,0,0,0,0,1,1],[1,1,0,0,0,0,0,1,0,0,0,1,0,0,1]]

ROOT_DIR="C:/Users/owen/Machine learning projects/Luc_tweet project"
RESULTS_DIR="{0}/Results" .format(ROOT_DIR)
TEMP_DIR="{0}/Temp".format(ROOT_DIR)




         

class pairwise_stats():
    """ Calculate parameters and z-scores based on Messager et al (2019) for two time series of the same length
    *   method run_test checks calculation of total 'marks' within a given time delta (lag)
    *   method calculate z_series finds z-scores across values of delta up to sqrt T (length of both time series)
    *   
    """
    
    def __init__(self,ts1,ts2,population_ps=[None,None,{}],delta=1,marks_dict={},
                 test=False,random_test=False,verbose=False,progress={}):
        if progress.get('one_percent_step'):
            if not progress.get('step')%progress.get('one_percent_step'):
                print(progress.get('step')/progress.get('one_percent_step'))
            
        if test:
            self.verbose=True
            self.run_test(random_test=random_test)
        else:
            self.ts1 = np.array(ts1)
            self.ts2 = np.array(ts2)
            self.T = len(self.ts1)
            self.population_ps=population_ps
            # if population probabilities are given, store - otherwise estimate from time series 
            if population_ps[0] and population_ps[2].get('Use population means'):
                self.p1=population_ps[0]
                self.p2=population_ps[1]
            else:
                self.n1=sum(self.ts1)
                self.n2=sum(self.ts2)
                self.p1 = self.n1/self.T
                self.p2 = self.n2/self.T

            if not len(self.ts1)==len(self.ts2):
                print("Warning - lengths of time series do not match")
            self.delta = delta
            
            if marks_dict.get(delta):
                self.marks_dict=marks_dict 
            else:
                self.marks_dict = {0:np.dot(self.ts1,self.ts2)}
                
            self.calculate_params()
            self.verbose=verbose

    def run_test(self,ts1=[],ts2=[],random_test=True):
        # calculate total marks 
        if not len(ts1):
            if random_test:
                self.ts1 = [np.random.randint(2) for i in range(20)]
                self.ts2 = [np.random.randint(2) for i in range(20)]
            else:
                self.ts1 = np.array([1,0,0,1,1,0,0,0,0,0,0,0,0,1,1])
                self.ts2 = np.array([1,1,0,0,0,0,0,1,0,0,0,1,0,0,1])
                #self.ts1 = TEST_MATRIX[0]
                #self.ts2 = TEST_MATRIX[1]
        self.p1 = sum(self.ts1)/len(self.ts1)
        self.p2 = sum(self.ts2)/len(self.ts2)
        self.T = len(self.ts1)
        self.marks_dict = {0:np.dot(self.ts1,self.ts2)}
        print("First test time series is {0}".format(self.ts1))
        print("Other test time series is {0}".format(self.ts2))
        print("Counting marks...")
        self.delta = 0
        self.calculate_params()
        stats = self.stats
        for self.delta in range(1,len(self.ts1)):           
            self.calculate_params()
            stats = pd.concat([self.stats,stats]) 
        print(stats)
    
    def count_marks_obsolete_slow_method(self):
        self.marks = np.dot(self.ts1,self.ts2)
        for shift in range(1,self.delta+1):
            self.marks += np.dot(self.ts1[shift:],self.ts2[:-shift])
            self.marks += np.dot(self.ts1[:-shift],self.ts2[shift:])
        self.marks_dict[self.delta]=self.marks
    
    def count_marks(self):
        m1=self.marks_dict.get(self.delta)
        m=self.marks_dict.get(self.delta-1)
        if m1:
            self.marks=m1 
        elif m:
            self.marks = m+np.dot(self.ts1[self.delta:],self.ts2[:-self.delta]) + np.dot(self.ts1[:-self.delta],self.ts2[self.delta:])       
            self.marks_dict[self.delta] = self.marks
        else:
            self.marks = np.dot(self.ts1,self.ts2)
            for shift in range(1,self.delta+1):
                self.marks += np.dot(self.ts1[shift:],self.ts2[:-shift])
                self.marks += np.dot(self.ts1[:-shift],self.ts2[shift:])
    
    def calculate_params(self,verbose=False):
        self.count_marks()
        w=2*self.delta+1
        pq=self.p1*self.p2
        if self.population_ps[2].get('Use population means'):
            self.var = w*pq*(1-pq)+2*self.delta*w*pq*(self.p1+self.p2-2*pq)
            self.sigma = np.sqrt(self.var)*np.sqrt(self.T)
        else:
            d=self.delta
            t=self.T
            n1=self.n1
            n2=self.n2
            N1=t*w-d*(d+1)
            N3=(2/3)*(d+1)*(d+2)*(7*d-3)-4*(d**2-1)+(t-2*d-2)*w*2*d
            N2=N1**2-N1-2*N3
            self.var=pq*(N1+N2*((n1-1)*(n2-1)/(t-1)**2)+N3*(n1+n2-2)/(t-1)-pq*N1**2)
            self.sigma=np.sqrt(self.var)
            
        self.mu = self.p1*self.p2*(self.T*(2*self.delta+1)-self.delta*(self.delta+1))
        if self.sigma:
            self.Z_score = (self.marks-self.mu)/(self.sigma)#*np.sqrt(self.T))
        else:
            self.Z_score = np.inf
        self.stats = pd.DataFrame([[self.mu,self.sigma,self.sigma*np.sqrt(self.T),self.marks,self.Z_score]],index = [''],columns = ['Mean','Sigma','Sigma*sqrt(T)','Total','Z score'])          
        if verbose:
            self.display_stats()
            
    def display_stats(self):
        print(pd.DataFrame([[self.p1,len(self.ts1)],[self.p2,len(self.ts2)]],index = ['Time series 1','Time series 2'],
                             columns = ['Probability','Length']))
        print(self.stats)
        
    def calculate_z_series(self,deltas = []):
        if self.verbose:
            print("Deltas being tested are {0}".format(deltas))
        self.z_series = []
        d = self.delta
        if not len(deltas):
            deltas = range(int(np.sqrt(self.T)))
        for delta in deltas:
            self.delta = delta
            self.calculate_params()
            self.z_series.append(self.Z_score)
        self.delta = d
        self.calculate_params


class tweet_data():
    """
    * Objects of class tweet_data store multiple time series with default length 50 (must be constant)
    * Delta is the time lag window under which statistical comparison will be made
    * method display_Z_vals uses class pairwise_stats to calculate z-values for pairs of time series and plots results in histogram
    * if disjoint sets is True, rows (time series) are matched from one matrix (population) to the other.  
    * Otherwise, only one population is used and all possible pairs are tested within it.
    * method test_delta analyses the sequences of z-values created by allowing delta to vary up to sqrt T for each pair of time series
    * If no axes are passed they are generated but this may not create good results
    * Two axes should be passed in a list: the first will display a histogram of the proportions of 1s in all time series;
    * the second will display a histogram of the z-scores for all correlations measured
    """
    
    def __init__(self,tweet_matrices = [],number=12000,length = 50,population_ps = [None,None,{}],bernoulli=True,delta = 2,
                disjoint_sets = False,test_delta = False,verbose=False,axes=None):
        self.axes = axes
        self.population_ps = population_ps
        self.ps=[[None for i in range(number)] for j in range(2)]
        self.verbose=verbose
        self.disjoint_sets=disjoint_sets
        
        if len(tweet_matrices):
            self.tweet_matrices = tweet_matrices
            self.T = len(self.tweet_matrices[0][0])
            self.n = len(self.tweet_matrices[0])
            self.MARKS = [[None for i in range(self.n)] for j in range(self.n)]
        else:
            if verbose:
                print("About to initialise tweet matrix")
            self.T=length
            self.n=number
            self.MARKS = [[None for i in range(self.n)] for j in range(self.n)]
            #if population_ps[2].get("random seed"):
            #    seeds = np.random.choice(10*self.n,2*self.n)
            #else:
            #    seeds = [None]*2*self.n      
            self.initialise_arrays()
            
        if not self.axes:
            f,self.axes=plt.subplots(1,len(self.tweet_matrices))

        for i,m in enumerate(self.tweet_matrices):           
            print("Analysis of tweet matrix {2}: {0} time series length {1}".format(len(m),len(m[0]),i))
            probs = np.sum([ts for ts in m],axis=1)/len(m[0])
            self.axes[0].hist(probs,bins=100)
            self.axes[0].set_title("Proportion of 1s in each time series")

        if test_delta:
            self.test_delta()               
        self.delta = delta

    def initialise_arrays(self):
        T=self.T
        n=self.n
        if self.population_ps[2]['Use fixed means for setup']:
            ps=self.population_ps[:2]
            self.tweet_matrices = [np.random.choice([0,1],p=[1-p,p],size=[n,T]) for p in ps]
            self.ps=[[p for i in range(n)] for p in ps]
        else:
            self.ps=[[np.random.chisquare(6)/30 for i in range(n)] for j in range(2)]
            #if self.verbose:
            #    print("{0} larger than 1".format(len([p for p in ps for ps in self.ps if p>=1])))
            self.ps=[[p if p<1 else np.random.uniform(0.1,0.3) for p in ps] for ps in self.ps]
            self.tweet_matrices = [[np.random.choice([0,1],p=[1-p,p],size=T) for p in ps] for ps in self.ps]
        
    def calculate_marks(self,verbose=False):
        M=np.array(self.tweet_matrices[0])
        self.MARKS_DICT={}
        if self.disjoint_sets:
            N=np.transpose(self.tweet_matrices[1])
        else:
            N=np.transpose(self.tweet_matrices[0])
        marks=np.dot(M,N)
        self.MARKS_DICT[0]=np.copy(marks)
        for i in range(1,self.delta+1):
            print("Calculating marks for delta {0}".format(i))
            marks += np.dot(M[:,i:],N[:-i,:])
            marks += np.dot(M[:,:-i],N[i:,:])
            self.MARKS_DICT[i] = np.copy(marks)
        
        self.MARKS=np.copy(marks)
        if verbose:
            print(self.MARKS_DICT)  
    
#    def calculate_Z_vals(self,verbose=False):
#        Ns=np.sum(self.tweet_matrices[0],axis=1)
#        if self.disjoint_sets:
#            N1s=np.sum(self.tweet_matrices[1],axis=1)            
#        else:
#            N1s = np.sum(self.tweet_matrices[0],axis=1)
#        if verbose:
#            print(self.tweet_matrices)
#            print(Ns)
#            print(N1s)
                   
    def display_Z_vals(self,ax=None):
        self.tweet_matrix=self.tweet_matrices[0]
        if self.disjoint_sets:
            self.tweet_matrix1=self.tweet_matrices[1]
            self.results = np.array([pairwise_stats(ts1=self.tweet_matrix[i],ts2=self.tweet_matrix1[i],delta=self.delta,progress={'step':i,'one_percent_step':int(self.n/100)},
                                population_ps = [self.ps[0][i],self.ps[1][i],self.population_ps[2]], marks_dict={self.delta: self.MARKS[i][i]},verbose=True).Z_score
                                for i in range(int(len(self.tweet_matrix)))])
        else:
            self.tweet_matrix1=self.tweet_matrices[0]
            self.ps=[self.ps[0],self.ps[0]]
            self.results = np.array([pairwise_stats(ts1=self.tweet_matrix[i],ts2=self.tweet_matrix[j],delta=self.delta,progress={'step':self.n*i+j+1,'one_percent_step':self.n*int(self.n/100+1)},
                                marks_dict={self.delta: self.MARKS[i][j]},population_ps = [self.ps[0][i],self.ps[0][j],self.population_ps[2]]).Z_score
                                for i in range(self.n-1) for j in range(i+1,self.n)])
        self.results = [r for r in self.results if r<np.inf]
        if ax==None:
            ax=self.axes[1]
        ax.hist(self.results,bins = 200,label='{0}'.format(self.T))
        if self.population_ps[0] and self.population_ps[2].get("Use population means"):
            s = "Z-scores based on known population means."
        else:
            s="Z-scores based on individual estimates of means."
        ax.set_title(s +  " Sample stats: mu={0:.2f},sigma={1:.2f}".format(np.mean(self.results),np.std(self.results)))
 
        print("Sample mean {0}, sample sigma {1}".format(np.mean(self.results),np.std(self.results)))
        print("Delta (max time-lag tested) is {0}".format(self.delta))
        if self.verbose:
            print("Statistics for randomly selected pair of time series")
            rans = np.random.choice(len(self.tweet_matrix),2)
            print("Entries {0} and {1} selected from total of {2} time series".format(rans[0],rans[1],len(self.tweet_matrix)))
            ps = pairwise_stats(delta=self.delta,ts1=self.tweet_matrix[rans[0]],ts2=self.tweet_matrix1[rans[1]],
                                population_ps = [self.ps[0][rans[0]],self.ps[1][rans[1]],self.population_ps[2]])
            ps.T=self.T
            ps.calculate_params(verbose=True)
            if self.population_ps[2]['Use fixed means for setup']:
                print("Probabilities of a particular entry in each time series is {0}".format(self.population_ps[0:2]))
            else:
                print("Probabilities are taken from a chai-squared distribution with cut-off at 1")
        
    
    def test_delta(self,max_delta=None,delta_step=1,ax=None):
        # creates a series of z-values based on increasing lag windows for each pair of time series
        z_results = []
        if max_delta:
            deltas = range(0,max_delta,delta_step)
        else:
            deltas=range(0,int(np.sqrt(self.T)),delta_step)
        if self.disjoint_sets:
            M=self.tweet_matrices[0]
            N=self.tweet_matrices[1]            
            for i in range(len(M)):
                if not i%int(len(M)/100+1):
                    print("{0}% complete".format(i*100/len(M)))
                ps = pairwise_stats(ts1 = M[i],ts2 = N[i],population_ps=self.population_ps)
                ps.calculate_z_series(deltas=deltas)
                if not np.inf in ps.z_series:
                    z_results.append(ps.z_series)
        else:
            M=self.tweet_matrices[0]
            for i in range(len(M)-1):
                if not i%int(len(M)/100+1):
                    print("{0}% complete".format(i*100/len(M)))
                for j in range(i+1,len(M)):
                    ps = pairwise_stats(ts1 = M[i],ts2 = M[j],population_ps=self.population_ps)
                    ps.calculate_z_series(deltas=deltas)
                    if not np.inf in ps.z_series:
                        z_results.append(ps.z_series) 

        ys = np.mean(z_results,axis=0)
        errs = np.std(z_results,axis=0)
        if ax==None:
            ax=self.axes[1]
        ax.set_title("Error bars for z-scores as delta increases")
        ax.errorbar(deltas,ys,errs,alpha=0.5)
        if self.verbose:
            print(M)
            print(z_results)        
                               
    def display(self,head = True):
        if head:
            print("Partial tweet matrix (10 rows,10 cols out of {0} by {1}):".format(len(self.tweet_matrices[0]),len(self.tweet_matrices[0][0].array)))
            print([self.tweet_matrices[0][i].array[:10] for i in range(10)])
        else:
            print(self.tweet_matrices[0])
    
    def display_coeffs(self):
        print("Number of individuals: {0}".format(len(self.tweet_matrix)))
        print("Length of time series: {0}".format(len(self.tweet_matrix[0])))
        corrs = corrcoef(self.tweet_matrix).flatten()
        corrs = [c for c in corrs if c < 0.99]
        print(corrs[:10])
        plt.hist(corrs, bins = 200)            


      

if __name__ == '__main__':
    TEST=False
    NEW_METHOD_TEST=True
    number=200
    length=10000
    p1=0.2
    p2=0.05
    params_dict = {'Use population means' : False,
                   'Use fixed means for setup' : False,
                       'random seed' : None,
                   'Test_mode' : False}
    if TEST:
        ps = pairwise_stats(test=True,random_test=False,delta=5)

    elif NEW_METHOD_TEST:
        number=20
        length=100
        params_dict['Test_mode']=True
        start=time.time()
        #td = tweet_data(tweet_matrices = [TEST_MATRIX1,TEST_MATRIX1],number=2,length=50,delta=14,disjoint_sets=False,verbose=True)
        td = tweet_data(number=number,length=length,population_ps=[p1,p2,params_dict],delta=int(np.sqrt(length)),disjoint_sets=False,verbose=True)
        #td.calculate_marks(verbose=False)
        print(time.time()-start)
        td.test_delta()
    else:
        td = tweet_data(number=number,length=length,population_ps=[p1,p2,params_dict],delta=int(np.sqrt(length)),disjoint_sets=False,verbose=True)
    """ 
    #print(td.ps)
    #td.calculate_Z_vals(verbose=True)
    start=time.time()
    print("Calculating marks")
    td.calculate_marks(verbose=False)
    t1=time.time()-start
    print("Time check: {0}".format(t1))

    print("Calculating z-values")
    td.display_Z_vals()
    t2=time.time()-start
    print("Total time: {0}".format(t2))
    
    td.MARKS = [[None for i in range(number)] for j in range(number)]
    print("Now calculating from scratch")
    start = time.time()
    td.display_Z_vals()
    t3=time.time()-start
    print("Time: {0}".format(t3))
    print(pd.DataFrame([t1,t2,t3],index=['Matrix method for marks','Total time using matrix method','Time using standard method']))
    """
      









                
                
                
        
        