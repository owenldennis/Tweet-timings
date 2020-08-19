# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 16:44:59 2019

@author: owen
"""

import numpy as np
import time
import time_series_correlation as tsc
import pointwise_correlation as pc
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
        if 'Z' in self.ts_dict.keys():
            self.ts_dict['X1']=self.combine(self.ts_dict['Y1'],self.ts_dict['Z'])
        if 'Z_star' in self.ts_dict.keys():
            self.ts_dict['X2']=self.combine(self.ts_dict['Y2'],self.ts_dict['Z_star'])
    
    def combine(self,ts1,ts2):
        return [np.sort(list(set(np.append(t,ts2[i])))) for i,t in enumerate(ts1)]
        
    def create_poisson_processes(self,key,create_delayed_also=False):
        lambdas=self.lambdas[key]['lambda']
        # create sequence of time intervals that are expected to sum to n*T
        ts=[np.random.exponential(l,size=(int(2*self.T/l))) for l in lambdas]
        #ts=np.random.exponential(l,size=(self.n,int(2*self.T/l)))
        
        if self.verbose:
            print("Random exponential set of time intervals totalling {0} initialised".format(self.T))
            print("Elapsed time {0}.  Now cumulating...".format(time.time()-self.start_time))
        cts=np.array([np.cumsum(t) for t in ts])

        if self.verbose:
            print("Cumulative sum initialised to time {0}".format(self.T))
            print("Elapsed time {0}.  Now truncating....".format(time.time()-self.start_time)) 
        self.ts_dict[key]=self.truncate_to_T(cts)

        if self.verbose:
            print("Cumulative sums rounded, ordered and truncated")
        # create delayed time series if required  
        if create_delayed_also:
            mu=self.params.get('mu')
            if mu==None:
                mu=0
                sigma=1
            else:
                sigma=self.params.get('sigma')
                if self.verbose:
                    print("Now created lagging time series...")
            cts_star=[[c+np.random.normal(mu,sigma) for c in ct] for ct in cts]
            if self.verbose:
                print("Lag added.  Elapsed time {0}.  Now truncating...".format(time.time()-self.start_time))
            self.ts_dict[key+'_star']=self.truncate_to_T(cts_star)
            
    def truncate_to_T(self,cts):
        return np.array([np.sort(list(set([int(round(c)) for c in ct if c<self.T]))) for ct in cts])
        
    def convert_to_binary_time_series(self,dense_time_series_array):
        dts=dense_time_series_array
        ts_matrix=[[1 if i in ts else 0 for i in range(self.T)] for ts in dts]
        return np.array(ts_matrix)
    
    def convert_to_dense_time_series(self,binary_time_series):
        bts=binary_time_series
        dts=[[i for i in range(len(b)) if b[i]] for b in bts]
        return np.array(dts)
        
            
if __name__=='__main__':
    
    if True:
        number=50
        length=1000
        delta=int(np.sqrt(length))
        #p=0.01
        #l=int(1/p)

        Y1=[20]*number
        print(Y1)
        Y2=[20]*number
        Z=[1000]*number


        sparse=False
        dense=not sparse
        
            
        lambdas = {'Y1': {'lambda':Y1,'baseline':False},
                'Y2': {'lambda':Y2,'baseline':False},
                'Z': {'lambda':Z, 'baseline' : True}
                }
        params={'mu':5,'sigma':0.1}

        pp=poisson_process(number,length,lambdas=lambdas,params=params)
        
        #print(pp.ts_dict)
        
        p1=np.sum([len(x) for x in pp.ts_dict['X1']])/(number*length)
        print(p1)
        p2=np.sum([len(x) for x in pp.ts_dict['X2']])/(number*length)
        print(p2)
        
        
        params_dict = {'T' : length,
                   'n' : number,
                   'p1' : p1,
                   'p2' : p2, 
                   'Use population means' : False,
                   'Use fixed means for setup' : True,
                   'random seed' : None,
                   'Test_mode' : False,
                   'sparse' :  sparse,
                   'dense' : dense}
        
        f,axes=plt.subplots(2,2)
        
        print("Starting poisson test...")
        X1=pp.ts_dict['X1']
        X2=pp.ts_dict['X2']

        td=pc.tweet_data([X1,X2],params=params_dict,disjoint_sets=False,delta=delta,axes=axes[0,:])

        start=time.time()
        #td.test_delta(max_delta=200,delta_step=5)
        td.display_Z_vals(ax=axes[0][1])
        



        
        
        
        