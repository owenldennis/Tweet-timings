# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 16:44:59 2019

@author: owen
"""

import numpy as np
import time
import time_series_correlation as tsc

class poisson_process():
    
    def __init__(self,number,length,lambdas,params={}):
        self.T=length
        self.n=number
        self.lambdas=lambdas
        self.params=params
        self.ts_dict={}
        self.initialise()
    
    def initialise(self):
        for key in self.lambdas.keys():
            self.create_poisson_processes(key,self.lambdas[key]['baseline'])
        self.ts_dict['X1']=self.combine(self.ts_dict['Y1'],self.ts_dict['Z'])
        self.ts_dict['X2']=self.combine(self.ts_dict['Y2'],self.ts_dict['Z_star'])
    
    def combine(self,ts1,ts2):
        return [np.sort(list(set(np.append(t,ts2[i])))) for i,t in enumerate(ts1)]
        
    def create_poisson_processes(self,key,create_delayed_also=False):
        l=self.lambdas[key]['lambda']
        # create sequence of time intervals that are expected to sum to n*T
        ts=np.random.exponential(l,size=int(self.T*self.n/l))
        cts=np.cumsum(ts)
        # add extra time intervals if random process not sufficient
        while cts[-1]<self.T*self.n:
            cts=np.append(cts,cts[-1]+np.random.exponential(l))
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
        cts=np.sort(list(set([int(t) for t in cumulative_time_intervals])))
        pps=[]
        start_index=0
        index=0
        # separate time intervals into n arrays each with final entry less than self.T
        for i in range(self.n):
            while cts[index]<self.T*(i+1):
                index+=1
            pps.append(cts[start_index:index]%self.T)
            start_index=index
        return np.array(pps)
        
    def convert_to_binary_time_series(self,poisson_process_interval_arrays):
        pps=poisson_process_interval_arrays
        ts_matrix=[[1 if i in pp else 0 for i in range(self.T)] for pp in pps]
        return np.array(ts_matrix)
            
if __name__=='__main__':
    p=0.01   
    l=int(1/p)    
    lambdas = {'Y1': {'lambda':l,'baseline':False},
            'Y2': {'lambda':l,'baseline':False},
            'Z': {'lambda':int(l/2), 'baseline' : True}
            }
    params={'mu':20,'sigma':3}
    pp=poisson_process(500,20000,lambdas=lambdas,params=params)
    ran=np.random.randint(pp.n)
    for key in pp.ts_dict.keys():
        print("Key {2}. First 10 entries in row {0} are {1}".format(ran,pp.ts_dict[key][ran][:10],key))
    
    X1=pp.convert_to_binary_time_series(pp.ts_dict['X1'])
    X2=pp.convert_to_binary_time_series(pp.ts_dict['X2'])
    td=tsc.tweet_data([X1,X2],disjoint_sets=True,delta=25)
    td.test_delta(max_delta=200,delta_step=5)
    
        
        
        