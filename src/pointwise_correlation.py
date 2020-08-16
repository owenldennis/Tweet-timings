# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:48:56 2019
Classes for calculation and analysis of correlation measures for pointwise comparison of lagging time series
Class pairwise_stats uses 2019 pairwise correlation measure (Messager et al) to determined z-scores for two Bernoulli time series
Class tweet_data creates a set of time series and analyses correlation across all pairs

******
TESTING BRANCHING - edited in new branch!
******

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
    
    def __init__(self,ts1,ts2,mean1=None,mean2=None,params={},delta=1,marks_dict={},
                 test=False,random_test=False,verbose=False,progress={}):
        if progress.get('one_percent_step'):
            if not progress.get('step')%progress.get('one_percent_step'):
                print(progress.get('step')/progress.get('one_percent_step'),end = ',')
        
        if True:
            self.ts1 = np.array(ts1)
            self.ts2 = np.array(ts2)
            self.params=params
            self.sparse=params.get('sparse')
            self.T=self.params.get('T')
            
            # if population probabilities are given and are to be used, store - otherwise estimate from time series 
            if self.params.get('Use population means'):
                self.n1=mean1*self.T
                self.n2=mean2*self.T
            else:
                self.n1=len(self.ts1)
                self.n2=len(self.ts2)                                                   
            self.p1 = self.n1/self.T
            self.p2 = self.n2/self.T
                
            self.delta = delta
            
            # if marks dictionary already passed, use this
            if marks_dict.get(delta):
                self.marks_dict=marks_dict 
            else:
                self.marks_dict={}  
            self.verbose=verbose
            self.calculate_params()


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
        
    def count_marks(self,max_delta=None,verbose=False):
        if not max_delta or max_delta<self.delta:
            max_delta=self.delta
        marks=0
        for t in self.ts1:
            marks+=len(np.where(np.abs(self.ts2-t)<=max_delta)[0])
        self.marks=marks

    def calculate_params(self,verbose=False):
        self.count_marks()
        w=2*self.delta+1
        pq=self.p1*self.p2
        # calcluation for sigma if using known probabilities to give means
        if self.params.get('Use population means'):
            self.var = w*pq*(1-pq)+2*self.delta*w*pq*(self.p1+self.p2-2*pq)
            self.sigma = np.sqrt(self.var)*np.sqrt(self.T)
        #calculation for sigma if using actual incidence of tweets to estimate means
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
            self.Z_score = (self.marks-self.mu)/(self.sigma)
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
    
    def __init__(self,tweet_matrices = [],params = {},bernoulli=True,delta = 2,
                disjoint_sets = False,test_delta = False,verbose=False,axes=[]):
        print("HI - POINTWISE CORRELATION VERSION BEING USED...HOW EXCITING!")
        self.axes = axes
        self.params = params
        self.verbose=verbose
        self.disjoint_sets=disjoint_sets
        self.T=self.params.get('T')
        self.n=self.params.get('n')
        self.ps=[[None for i in range(self.n)] for j in range(2)]
        self.tweet_matrices = np.array(tweet_matrices)
        self.sparse=params['sparse']
        # if matrices are passed and are dense, length of each time series should have been passed in params
        # if not, it is inferred as the largest final entry across all time series
        # if matrices passed are sparse, they are densified
        if len(self.tweet_matrices):
            if not self.T or not self.n:
                self.T=max([ts[-1] for i in range(len(tweet_matrices)) for ts in tweet_matrices[i]])
                self.n = len(self.tweet_matrices[0])
            self.ps = self.params.get('Known probabilities array')
            
            if self.sparse:
                self.densify()
        else:
            if verbose:
                print("About to initialise tweet matrix")    
            self.initialise_arrays()
            
        if not len(self.axes):
            f,self.axes=plt.subplots(1,len(self.tweet_matrices))

        if True:
            for i,m in enumerate(self.tweet_matrices):           
                print("Analysis of tweet matrix {2}: {0} time series length {1}".format(len(m),self.T,i))
                probs=[len(ts)/self.T for ts in m]
                self.axes[0].hist(probs,bins=100)
                
            self.axes[0].set_title("Proportion of 1s across all time series analysed")

        if test_delta:
            self.test_delta()               
        self.delta = delta

    def initialise_arrays(self):
        T=self.T
        n=self.n
        if self.params['Use fixed means for setup']:
            ps=[self.params['p1'],self.params['p2']]
            self.tweet_matrices = [np.random.choice([0,1],p=[1-p,p],size=[n,T]) for p in ps]
            self.ps=[[p for i in range(n)] for p in ps]
        else:
            self.ps=[[np.random.chisquare(6)/30 for i in range(n)] for j in range(2)]
            self.ps=[[p if p<1 else np.random.uniform(0.1,0.3) for p in ps] for ps in self.ps]
            self.tweet_matrices = [[np.random.choice([0,1],p=[1-p,p],size=T) for p in ps] for ps in self.ps]

    def densify(self,verbose=False):
        self.tweet_matrices=[[np.where(ts==1)[0] for ts in ts_matrix] for ts_matrix in self.tweet_matrices]
        if verbose:
            print(self.tweet_matrices)
        
                   
    def display_Z_vals(self,ax=None):
        self.tweet_matrix=self.tweet_matrices[0]
        if self.disjoint_sets:
            self.tweet_matrix1=self.tweet_matrices[1]
            self.results = np.array([pairwise_stats(ts1=self.tweet_matrix[i],ts2=self.tweet_matrix1[i],mean1=self.ps[0][i],mean2=self.ps[1][i],
                                                    delta=self.delta,progress={'step':i,'one_percent_step':int(self.n/100)},
                                                    params = self.params,verbose=True).Z_score
                                for i in range(int(len(self.tweet_matrix)))])
        else:
            self.tweet_matrix1=self.tweet_matrices[0]
            self.ps=[self.ps[0],self.ps[0]]
            self.results = np.array([pairwise_stats(ts1=self.tweet_matrix[i],ts2=self.tweet_matrix[j],mean1=self.ps[0][i],mean2=self.ps[0][j],
                                                    delta=self.delta,progress={'step':self.n*i+j+1,'one_percent_step':self.n*int(self.n/100+1)},
                                                    params =self.params).Z_score
                                for i in range(self.n-1) for j in range(i+1,self.n)])
        self.results = [r for r in self.results if r<np.inf]
        if ax==None:
            ax=self.axes[1]
        ax.hist(self.results,bins = 200,label='{0}'.format(self.T))
        if self.params['p1'] and self.params.get("Use population means"):
            s = "Z-scores based on known population means."
        else:
            s="Z-scores based on individual estimates of means."
        ax.set_title(s +  " Sample stats: mu={0:.2f},sigma={1:.2f}".format(np.mean(self.results),np.std(self.results)))
 
        print("Sample mean {0}, sample sigma {1}".format(np.mean(self.results),np.std(self.results)))
        print("Delta (max time-lag tested) is {0}".format(self.delta))
#        if self.verbose:
#            print("Statistics for randomly selected pair of time series")
#            rans = np.random.choice(len(self.tweet_matrix),2)
#            print("Entries {0} and {1} selected from total of {2} time series".format(rans[0],rans[1],len(self.tweet_matrix)))
#            ps = pairwise_stats(delta=self.delta,ts1=self.tweet_matrix[rans[0]],ts2=self.tweet_matrix1[rans[1]],
#                                params['p2'] = [self.ps[0][rans[0]],self.ps[1][rans[1]],self.params])
#            ps.T=self.T
#            ps.calculate_params(verbose=True)
#            if self.params['Use fixed means for setup']:
#                print("Probabilities of a particular entry in each time series is {0}".format([self.params['p1'],self.params['p2']]))
#            else:
#                print("Probabilities are taken from a chai-squared distribution with cut-off at 1")
        
    
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
                ps = pairwise_stats(ts1 = M[i],ts2 = N[i],params=self.params)
                ps.calculate_z_series(deltas=deltas)
                if not np.inf in ps.z_series:
                    z_results.append(ps.z_series)
        else:
            M=self.tweet_matrices[0]
            for i in range(len(M)-1):
                if not i%int(len(M)/100+1):
                    print("{0}% complete".format(i*100/len(M)))
                for j in range(i+1,len(M)):
                    ps = pairwise_stats(ts1 = M[i],ts2 = M[j],params=self.params)
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


def test_sigma_with_inferred_means(xs=[100,200,300,400,500],params={},disjoint=False,verbose=False,axes=[]):
    """
    *Compares z_scores when means are inferred/known
    *Plot of sigma values for each against length of time series (given by parameter xs) is also shown
    *Number of time series given by parameter number (either in one or both populations)
    *If parameter disjoint is True, two separate populations are tested against each other pairwise
    *If disjoint is False, only one population is tested but all possible pairings are formed
    *If probs are None, individual probabilities are taken from a chai squared distribution
    """
    ts=[]
    ys = [] # sigma values for z scores based on inferred means
    zs =[] # mean values for z scores based on inferred means
    y1s=[] # sigma values for z scores based on known population means
    z1s=[] #sigma values for z scores based on known population means
    if len(params.keys()):
        params_dict = params
    else:
        params_dict = {'T': 20,
                   'n': 5,
                   'p1': 0.2,
                   'p2': 0.05,
                   'Use population means' : False,
                       'random seed' : None}
    if params_dict['Use population means']:
        pop1_prob = params_dict['p1'] # probability of a 1 at each time step for first population
        pop2_prob = params_dict['p2'] # probability of a 1 at each time step for second population
    else:
        pop1_prob=None
        pop2_prob=None

    start_time = time.time()
    for T in xs:
        ts.append(T)
        params_dict['T']=T        
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
        td = tweet_data(params = params_dict,delta = delta,
                        disjoint_sets=disjoint,verbose=verbose,axes = [axes[0][0],axes[1][0]])
        td.display_Z_vals()
        ys.append(np.std(td.results))
        zs.append(np.mean(td.results))
        
        params_dict['Use population means']=True
        td.params=params_dict
        td.axes=[axes[0][1],axes[1][1]]
        print("Running with population means. Time elapsed: {0}".format(time.time()-start_time))
        #td = tweet_data(number = number,length = T,params = [pop1_prob,pop2_prob,params_dict],delta = delta,disjoint_sets=disjoint,verbose=False,axes = [axes[0][1],axes[1][1]])
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

      

if __name__ == '__main__':
    TEST=False
    TEST1=True
    length = 50
    number=200
    repeats = 10
    xs=[20,50]
    p1=0.2
    p2=0.05
    params_dict = {'T': length,
                   'n': number,
                   'p1': 0.2,
                   'p2': 0.05,
                   'Use population means' : False,
                   'Use fixed means for setup' : False,
                       'random seed' : None,
                'sparse': True}
    if TEST:
        if TEST1:
            params_dict['T']=len(TEST_MATRIX1[0])
            params_dict['n']=len(TEST_MATRIX1)
            TM=TEST_MATRIX1
        else:
            params_dict['T']=len(TEST_MATRIX[0])
            params_dict['n']=len(TEST_MATRIX)
            TM=TEST_MATRIX
        #ps = pairwise_stats(test=True,random_test=False,delta=5)
        td = tweet_data(tweet_matrices = [TM,TM],params=params_dict,delta=5,verbose=True)
    else:
        df1=pd.DataFrame()
        df1.to_csv("{0}/Sigma_comparison{1}.csv".format(TEMP_DIR,str(xs)[:10]),mode='w')

        if repeats:
            f,axes=plt.subplots(2,2)
        else:
            axes=[]
        
        for i in range(repeats+1):
            df=test_sigma_with_inferred_means(xs=xs,params=params_dict,disjoint=False,axes=axes,verbose=False)
            df1=pd.concat([df1,df],axis=0)

        df1.to_csv("{0}/Sigma_comparison{1}_repeated_runs_{2}_time series.csv".format(TEMP_DIR,i+1,number))
        print("Means of each column of dataframe")
        print(np.mean(df1))
        print("Stdev of each column of dataframe")
        print(np.std(df1))








                
                
                
        
        