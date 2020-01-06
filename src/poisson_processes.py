# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 16:44:59 2019

@author: owen
"""

import numpy as np
import time
import time_series_correlation as tsc
import matplotlib.pyplot as plt

class poisson_process():
    
    def __init__(self,number,length,lambdas,params={},verbose=False):
        self.start_time=time.time()
        self.T=length
        self.n=number
        self.lambdas=lambdas
        self.params=params
        self.ts_dict={}
        self.verbose=verbose
        
        self.initialise()
    
    def initialise(self):
        for key in self.lambdas.keys():
            print("Initialising for key {0}".format(key))
            self.create_poisson_processes(key,self.lambdas[key]['baseline'])
            print("Elapsed time: {0}".format(time.time()-self.start_time))
        if self.ts_dict.get('Z'):
            self.ts_dict['X1']=self.combine(self.ts_dict['Y1'],self.ts_dict['Z'])
        if self.ts_dict.get('Z_star'):
            self.ts_dict['X2']=self.combine(self.ts_dict['Y2'],self.ts_dict['Z_star'])
    
    def combine(self,ts1,ts2):
        return [np.sort(list(set(np.append(t,ts2[i])))) for i,t in enumerate(ts1)]
        
    def create_poisson_processes(self,key,create_delayed_also=False):
        l=self.lambdas[key]['lambda']
        # create sequence of time intervals that are expected to sum to n*T
        ts=np.random.exponential(l,size=int(1.1*self.T*self.n/l))
        if self.verbose:
            print("Random exponential set of time intervals totalling {0} initialised".format(self.T))
            print("Elapsed time {0}".format(time.time()-self.start_time))
        cts=np.cumsum(ts)
        if self.verbose:
            print("Cumulative sum initialised to time {0}".format(cts[-1]))
            print("Elapsed time {0}".format(time.time()-self.start_time))           
        # add extra time intervals if random process not sufficient
        while cts[-1]<self.T*self.n:
            cts=np.append(cts,cts[-1]+np.random.exponential(l))
        if self.verbose:
            print("Cumulative sum to time {0} initialised".format(self.T*self.n))
            print("Elapsed time {0}".format(time.time()-self.start_time))  
        # split into multiple time series
        self.ts_dict[key] = self.split_cumulative_time_intervals_into_time_series(cts)
        # create delayed time series if required  
        if create_delayed_also:
            mu=self.params.get('mu')
            if mu==None:
                mu=0
                sigma=1
            else:
                sigma=self.params.get('sigma')
            cts_star=[c+np.random.normal(mu,sigma) for c in cts]
            self.ts_dict[key+"_star"]=self.split_cumulative_time_intervals_into_time_series(cts_star)
    
    def split_cumulative_time_intervals_into_time_series(self,cumulative_time_intervals):
        if self.verbose:
            print("Splitting into {0} distinct series".format(self.n))
            print("Time elapsed {0}".format(time.time()-self.start_time))
        cts=np.sort(list(set([int(t) for t in cumulative_time_intervals])))
        pps=[]
        start_index=0
        index=0
        # separate time intervals into n arrays each with final entry less than self.T
        for i in range(self.n):
            while cts[index]<self.T*(i+1):
                index+=1
            pps.append(cts[start_index:index]%self.T)
            # deal with empty arrays, select one time step at random
            if not len(pps[-1]):
                pps[-1]=[np.random.randint(self.T)]
            start_index=index
        return np.array(pps)
        
    def convert_to_binary_time_series(self,poisson_process_interval_arrays):
        pps=poisson_process_interval_arrays
        ts_matrix=[[1 if i in pp else 0 for i in range(self.T)] for pp in pps]
        return np.array(ts_matrix)
    
    def convert_to_sparse_time_series(self,binary_time_series):
        bts=binary_time_series
        sts=[[i for i in range(len(b)) if b[i]] for b in bts]
        return sts
        
def compare_sparse_to_dense():
    test_sparse_conversion=False
    
    if test_sparse_conversion:
        p=0.5   
        l=int(1/p)    
        lambdas = {'Y1': {'lambda':l,'baseline':False},
                'Y2': {'lambda':l,'baseline':False},
                'Z': {'lambda':int(l/2), 'baseline' : True}
                }
        params={'mu':20,'sigma':3}
        pp=poisson_process(2,10,lambdas,params)
        bts=pp.convert_to_binary_time_series(pp.ts_dict['X1'])
        print(bts)
        sts=pp.convert_to_sparse_time_series(bts)
        print(sts)
    else:
        number=500
        length=1000
        delta=int(np.sqrt(length))
        p=0.05
        l=int(1/p)
        Y1=l
        Y2=l
        Z=5000
        f,axes=plt.subplots(2,2)
        
            
        lambdas = {'Y1': {'lambda':Y1,'baseline':False},
                'Y2': {'lambda':Y2,'baseline':False},
                'Z': {'lambda':Z, 'baseline' : True}
                }
        params={'mu':5,'sigma':3}
        pp=poisson_process(number,length,lambdas=lambdas,params=params)
        ran=np.random.randint(pp.n)
        for key in pp.ts_dict.keys():
            print("Key {2}. First 10 entries in row {0} are {1}".format(ran,pp.ts_dict[key][ran][:10],key))
        
        
        X1=pp.convert_to_binary_time_series(pp.ts_dict['X1'])
        X2=pp.convert_to_binary_time_series(pp.ts_dict['X2'])

        params_dict = {'T' : length,
                   'n' : number,
                   'Use population means' : False,
                   'Use fixed means for setup' : False,
                   'random seed' : None,
                   'Test_mode' : False,
                   'sparse' : False}
        print("Starting first test...")
        td=tsc.tweet_data([X1,X2],population_ps=[None,None,params_dict],disjoint_sets=True,delta=delta,axes=axes[0,:])
        start=time.time()
        #td.test_delta(max_delta=200,delta_step=5)
        td.display_Z_vals()
        #print(td.results)
        t2=time.time()-start
        print("Completed in {0}".format(time.time()-start))
        print("Second test")
        params_dict['sparse']=True
        X1=pp.ts_dict['Y1']
        X2=pp.ts_dict['Y2']
        td=tsc.tweet_data([X1,X2],population_ps=[None,None,params_dict],disjoint_sets=True,delta=delta,axes=axes[1,:])
        t1=time.time()
        #td.test_delta(max_delta=200,delta_step=5)
        td.display_Z_vals()
        #print(td.results)
        print("New method completed in {0}".format(time.time()-t1))
        print("Old method completed in {0}".format(t2))
            
if __name__=='__main__':
    test=False
    if test:
        compare_sparse_to_dense()
        
    else:
        number=5000
        length=100000
        delta=int(np.sqrt(length))
        p=0.1
        l=int(1/p)
        Y1=l
        Y2=l
        Z=l
        sparse=True
        verbose=True
        
            
        lambdas = {'Y1': {'lambda':Y1,'baseline':False},
                'Y2': {'lambda':Y2,'baseline':False},
                #'Z': {'lambda':Z, 'baseline' : True}
                }
        params={'mu':5,'sigma':3}
        pp=poisson_process(number,length,lambdas=lambdas,params=params,verbose=verbose)
        params_dict = {'T' : length,
                   'n' : number,
                   'Use population means' : False,
                   'Use fixed means for setup' : False,
                   'random seed' : None,
                   'Test_mode' : False,
                   'sparse' :  True}
        
        f,axes=plt.subplots(2,2)
        
        print("Starting poisson test...")
        X1=pp.ts_dict['Y1']
        X2=pp.ts_dict['Y2']
        p1=sum([len(x) for x in X1])/(number*length)
        p2=sum([len(x) for x in X2])/(number*length)
        print(p2)
        print(p1)
        td=tsc.tweet_data([X1,X2],population_ps=[p1,p2,params_dict],disjoint_sets=True,delta=delta,axes=axes[0,:])
        start=time.time()
        #td.test_delta(max_delta=200,delta_step=5)
        td.display_Z_vals(ax=axes[0][1])
        
        print("Starting reinitialised array test")
        params_dict = {'T' : length,
                   'n' : number,
                   'Use population means' : True,
                   'Use fixed means for setup' : False,
                   'random seed' : None,
                   'Test_mode' : False,
                   'sparse' :  False}  
        
        #X1=np.array([np.random.choice([0,1],p=[1-p1,p1],size=[number,length])])
        #X2=np.array([np.random.choice([0,1],p=[1-p2,p2],size=[number,length])])
        #X1=pp.convert_to_binary_time_series(X1)
        #X2=pp.convert_to_binary_time_series(X2)
        #td=tsc.tweet_data([X1,X2],population_ps=[p1,p2,params_dict],disjoint_sets=True,delta=25,axes=axes[1,:])
        #td.display_Z_vals(ax=axes[1][1])
        #t2=time.time()-start
        print("Completed in {0}".format(time.time()-start))


        
        
        
        