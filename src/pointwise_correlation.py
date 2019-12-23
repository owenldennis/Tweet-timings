# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:48:56 2019
Classes for calculation and analysis of correlation measures for pointwise comparison of lagging time series
Class tweet_sequence creates an individual time series in which sparsity is determined by a probability passed or assigned
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

ROOT_DIR="C:/Users/owen/Machine learning projects/Luc_tweet project"
RESULTS_DIR="{0}/Results" .format(ROOT_DIR)
TEMP_DIR="{0}/Temp".format(ROOT_DIR)

class tweet_sequence_obsolete():
    """
    * Creates a (sparse) time series which can either be categorical (Bernoulli) or multiple values
    * Sparsity determined by p_tweet, the probability of a particular time-step being non-zero
    * If p_tweet is not passed, the value is determined by an exponential distribution based on the length of the time series
    * If Bernoulli==True non-zero entries are 1  
    * If Bernoulli==False non-zero entries are assigned a random integer value up to a maximum value
    * which is determined by an exponential distribution with parameter 10
    """
    
    def __init__(self, length=65, p_tweet=0.25, random_seed = None,bernoulli=True):
        self.length = length
        self.bernoulli = bernoulli
        self.p_tweet = p_tweet
        self.array = []
        self.random_seed = random_seed
        if not self.p_tweet:
            self.p_tweet = 10
            while self.p_tweet>1:
                self.p_tweet = np.random.chisquare(6)/30
        self.max_tweets = int(np.random.exponential(10))
        self.set_up()
        
    def set_up(self):
        # create time series (either categorical or multi-valued)
        if self.bernoulli:
            np.random.seed(self.random_seed)
            seeds = np.random.choice(10*self.length,self.length)
            for seed in seeds:
                np.random.seed(seed=seed)
                if np.random.uniform()<self.p_tweet:
                    self.array.append(1)
                else:
                    self.array.append(0)
        #else:
        #    self.indices = np.random.choice(range(self.length),np.int(self.length*self.p_tweet),replace=False)
        #    self.array = [np.random.randint(1,self.max_tweets) if i in self.indices else 0 for i in range(self.length)]
    
    def display(self):
        print(self.array)
        pass

class tweet_sequence():

    def __init__(self):
        print("Warning...referring to class tweet_sequence")   
         

class pairwise_stats():
    """ Calculate parameters and z-scores based on Messager et al (2019) for two time series of the same length
    *   method run_test checks calculation of total 'marks' within a given time delta (lag)
    *   method calculate z_series finds z-scores across values of delta up to sqrt T (length of both time series)
    *   
    """
    
    def __init__(self,ts1,ts2,population_ps=[None,None,{}],delta=1,
                 test=False,random_test=False,verbose=False):
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
        m=self.marks_dict.get(self.delta-1)
        if m:
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
    * method display_Z_vals uses class pairwise_stats to calculate z-values for each pair of time series and plots results in histogram
    * method test_delta analyses the sequences of z-values created by allowing delta to vary up to sqrt T for each pair of time series
    * 
    """
    
    def __init__(self,tweet_matrix = [],number=12000,length = 50,population_ps = [None,None,{}],bernoulli=True,delta = 2,
                disjoint_sets = False,test_delta = False,verbose=False,axes=None):
        if not axes:
            self.f,self.axes = plt.subplots(2,1)
        else:
            self.axes = axes
        self.population_ps = population_ps
        if len(tweet_matrix):
            self.tweet_matrix = tweet_matrix
            self.T = len(self.tweet_matrix[0])
            self.n = len(self.tweet_matrix)
        else:
            if verbose:
                print("About to initialise tweet matrix")
            self.T=length
            self.n=number
            if population_ps[2].get("random seed"):
                seeds = np.random.choice(10*self.n,2*self.n)
            else:
                seeds = [None]*2*self.n
            #self.tweet_matrix = [tweet_sequence(length=self.T,p_tweet=population_ps[0],random_seed = seeds[i],bernoulli=bernoulli) for i in range(self.n)]
            #self.tweet_matrix1 = [tweet_sequence(length=self.T,p_tweet=population_ps[1],random_seed = seeds[i+self.n],bernoulli=bernoulli) for i in range(self.n)]
            
            self.initialise_arrays() 
            
            print("Analysis of tweet matrix 1 : {0} time series length {1}".format(len(self.tweet_matrix),len(self.tweet_matrix[0])))
            probs = np.sum([ts for ts in self.tweet_matrix],axis=1)/len(self.tweet_matrix[0])
            self.axes[0].hist(probs,bins=100)
            axes[0].set_title("Proportion of 1s in each time series")
            print("Analysis of tweet matrix 2 : {0} time series length {1}".
                      format(len(self.tweet_matrix1),len(self.tweet_matrix1[0])))
            probs = np.sum([ts for ts in self.tweet_matrix1],axis=1)/len(self.tweet_matrix1[0])
            self.axes[0].hist(probs,bins=100)                

        self.verbose=verbose
        self.disjoint_sets=disjoint_sets
        if test_delta:
            self.test_delta()               
        self.delta = delta

    def initialise_arrays(self):
        ps=self.population_ps[:2]
        T=self.T
        n=self.n
        if self.population_ps[2]['Use population means']:
            p_arrays = [np.random.choice([0,1],p=[1-ps[i],ps[i]],size=[n,T]) for i in range(len(ps))]
        else:
            ps=[np.random.chisquare(6)/30 for i in range(n)]
            print("{0} larger than 1".format(len([p for p in ps if p>=1])))
            ps=[p if p<1 else np.random.uniform(0.1,0.3) for p in ps]
            p_arrays = [[np.random.choice([0,1],p=[1-ps[i],ps[i]],size=T) for i in range(len(ps))]for j in range(2)]
        self.tweet_matrix=p_arrays[0]
        self.tweet_matrix1=p_arrays[1]
        
    def display_Z_vals(self):
        if self.disjoint_sets:
            self.results = np.array([pairwise_stats(ts1=self.tweet_matrix[i],ts2=self.tweet_matrix1[i],
                                population_ps = self.population_ps, delta=self.delta,verbose=True).Z_score
                                for i in range(int(len(self.tweet_matrix)))])
        else:
            self.results = np.array([pairwise_stats(ts1=self.tweet_matrix[i],ts2=ts,delta=self.delta,
                                population_ps = self.population_ps).Z_score
                                for i in range(len(self.tweet_matrix)-1) for ts in self.tweet_matrix[i+1:] ])
        self.results = [r for r in self.results if r<np.inf]
        self.axes[1].hist(self.results,bins = 200,label='{0}'.format(self.T))
        if self.population_ps[0] and self.population_ps[2].get("Use population means"):
            s = "Z-scores based on known population means."
        else:
            s="Z-scores based on individual estimates of means."
        self.axes[1].set_title(s +  " Sample stats: mu={0:.2f},sigma={1:.2f}".format(np.mean(self.results),np.std(self.results)))
 
        print("Sample mean {0}, sample sigma {1}".format(np.mean(self.results),np.std(self.results)))
        print("Delta (max time-lag tested) is {0}".format(self.delta))
        if self.verbose:
            print("Statistics for randomly selected pair of time series")
            rans = np.random.choice(len(self.tweet_matrix),2)
            print("Entries {0} and {1} selected from total of {2} time series".format(rans[0],rans[1],len(self.tweet_matrix)))
            ps = pairwise_stats(delta=self.delta,ts1=self.tweet_matrix[rans[0]],ts2=self.tweet_matrix1[rans[1]],
                                population_ps = self.population_ps)
            ps.T=self.T
            ps.calculate_params(verbose=True)
            if self.population_ps[2]:
                print("Probabilities of a particular entry in each time series is {0}".format(self.population_ps[0:2]))
            else:
                print("Probabilities are taken from a chai-squared distribution with cut-off at 1")
        
    
    def test_delta(self):
        # creates a series of z-values based on increasing lag windows for each pair of time series
        z_results = []
        deltas=range(int(np.sqrt(self.T)))
        for i in range(len(self.tweet_matrix)-1):
            for j in range(i+1,len(self.tweet_matrix)):
                ps = pairwise_stats(ts1 = self.tweet_matrix[i],ts2 = self.tweet_matrix[j],population_ps=self.population_ps)
                ps.calculate_z_series(deltas=deltas)
                if not np.inf in ps.z_series:
                    z_results.append(ps.z_series) 

        ys = np.mean(z_results,axis=0)
        errs = np.std(z_results,axis=0)
        self.axes[1].errorbar(deltas,ys,errs,alpha=0.5)
        if self.verbose:
            print(self.tweet_matrix)
            print(z_results)        
                               
    def display(self,head = True):
        if head:
            print("Partial tweet matrix (10 rows,10 cols out of {0} by {1}):".format(len(self.tweet_matrix),len(self.tweet_matrix[0].array)))
            print([self.tweet_matrix[i].array[:10] for i in range(10)])
        else:
            print(self.tweet_matrix)
    
    def display_coeffs(self):
        print("Number of individuals: {0}".format(len(self.tweet_matrix)))
        print("Length of time series: {0}".format(len(self.tweet_matrix[0])))
        corrs = corrcoef(self.tweet_matrix).flatten()
        corrs = [c for c in corrs if c < 0.99]
        print(corrs[:10])
        plt.hist(corrs, bins = 200)            


def test_sigma_with_inferred_means(number=100,xs=[100,200,300,400,500],params=[0.1,0.5,{}],disjoint=False,verbose=False,axes=[]):
    """
    *Compares z_scores when means are inferred/known
    *Plot of sigma values for each against length of time series (given by parameter xs) is also shown
    *Number of time series given by parameter number (either in one or both populations)
    *If parameter disjoint is True, two separate populations are tested against each oterh pairwise
    *If disjoint is False, only one population is tested but all possible pairings are formed
    *If probs are None, individual probabilities are taken from a chai squared distribution
    """
    ts=[]
    ys = [] # sigma values for z scores based on inferred means
    zs =[] # mean values for z scores based on inferred means
    y1s=[] # sigma values for z scores based on known population means
    z1s=[] #sigma values for z scores based on known population means
    if len(params[2].keys()):
        params_dict = params[2]
    else:
        params_dict = {'Use population means' : False,
                       'random seed' : None}
    if params_dict['Use population means']:
        pop1_prob = probs[0] # probability of a 1 at each time step for first population
        pop2_prob = probs[1] # probability of a 1 at each time step for second population
    else:
        pop1_prob=None
        pop2_prob=None

    start_time = time.time()
    for T in xs:
        ts.append(T)        
        print("T is {0}".format(T))
        delta = int(np.sqrt(T))
        if not len(axes):
            f,axes = plt.subplots(2,2)
            f.suptitle("Comparison of inferred v known population means for T = {0},delta = {1}".format(T,delta),fontsize=16)
        for ax in axes[0]:
            ax.plot([0,1],[0,0],'.',alpha=0.2)
        for ax in axes[1]:
            ax.plot([-2,2],[0,0],'.',alpha=0.2)
            
        params_dict['Use population means']=False
        print("Running with inferred means.  Time elapsed: {0}".format(time.time()-start_time))
        td = tweet_data(number = number,length = T,population_ps = [pop1_prob,pop2_prob,params_dict],delta = delta,
                        disjoint_sets=disjoint,verbose=verbose,axes = [axes[0][0],axes[1][0]])
        td.display_Z_vals()
        ys.append(np.std(td.results))
        zs.append(np.mean(td.results))
        
        params_dict['Use population means']=True
        td.population_ps=[pop1_prob,pop2_prob,params_dict]
        td.axes=[axes[0][1],axes[1][1]]
        print("Running with population means. Time elapsed: {0}".format(time.time()-start_time))
        #td = tweet_data(number = number,length = T,population_ps = [pop1_prob,pop2_prob,params_dict],delta = delta,disjoint_sets=disjoint,verbose=False,axes = [axes[0][1],axes[1][1]])
        td.display_Z_vals()
        y1s.append(np.std(td.results))
        z1s.append(np.mean(td.results))
        pd.DataFrame(np.transpose([ts,zs,ys,z1s,y1s])).to_csv("{0}/Accumulating results.csv".format(TEMP_DIR))
    
    axes[0][1].errorbar(xs,zs,ys,color='r',label='inferred')
    axes[0][1].errorbar(xs,z1s,y1s,color='b',label='known')
    axes[0][1].legend()
    axes[1][0].legend()
    axes[1][1].legend()
    df=pd.concat([pd.DataFrame(xs,columns=['T']),
                   pd.DataFrame(ys,columns=['Z score std dev (inferred means)']),
                   pd.DataFrame(zs,columns=['Z score mean (inferred means)']),
                   pd.DataFrame(y1s,columns=['Z score std dev (known means)']),
                   pd.DataFrame(z1s,columns=['Z score mean (known means)'])],axis=1)
    df.to_csv("{0}/Sigma_comparison{1}.csv".format(TEMP_DIR,str(xs)[:10]),mode='a')  
    return df

def test_delta():
    params_dict = {'Use population means' : True,
                       'random seed' : None}
    pop1_prob = None
    pop2_prob = None
    disjoint=False
    start_time = time.time()
    number=10
    T=100
    delta=int(np.sqrt(T))
    f,axes=plt.subplots(2,1)
    f.suptitle("Comparison of inferred v known population means for T = {0},delta = {1}".format(T,delta),fontsize=16)
        
    td = tweet_data(number = number,length = T,population_ps = [pop1_prob,pop2_prob,params_dict],delta = delta,
                        disjoint_sets=disjoint,verbose=False,axes=[axes[0],axes[1]])
    td.test_delta()
      

if __name__ == '__main__':
    TEST=False
    number=10000
    repeats = 0
    xs=[10000,20000]
    p1=0.2
    p2=0.05
    params_dict = {'Use population means' : False,
                       'random seed' : None}
    if TEST:
        ps = pairwise_stats(test=True,random_test=False,delta=5)
        #td = tweet_data(tweet_matrix = TEST_MATRIX,number=2,length=50,delta=5,verbose=True)
    else:
        df1=pd.DataFrame()
        df1.to_csv("{0}/Sigma_comparison{1}.csv".format(TEMP_DIR,str(xs)[:10]),mode='w')

        if repeats:
            f,axes=plt.subplots(2,2)
        else:
            axes=[]
        
        for i in range(repeats+1):
            df=test_sigma_with_inferred_means(number=number,xs=xs,params=[p1,p2,params_dict],disjoint=True,axes=axes,verbose=False)
            df1=pd.concat([df1,df],axis=0)

        df1.to_csv("{0}/Sigma_comparison{1}_repeated_runs_{2}_time series.csv".format(TEMP_DIR,i+1,number))
        print("Means of each column of dataframe")
        print(np.mean(df1))
        print("Stdev of each column of dataframe")
        print(np.std(df1))








                
                
                
        
        