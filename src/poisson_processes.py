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
        if self.ts_dict.get('Z'):
            self.ts_dict['X1']=self.combine(self.ts_dict['Y1'],self.ts_dict['Z'])
        if self.ts_dict.get('Z_star'):
            self.ts_dict['X2']=self.combine(self.ts_dict['Y2'],self.ts_dict['Z_star'])
    
    def combine(self,ts1,ts2):
        return [np.sort(list(set(np.append(t,ts2[i])))) for i,t in enumerate(ts1)]
        
    def create_poisson_processes(self,key,create_delayed_also=False):
        l=self.lambdas[key]['lambda']
        # create sequence of time intervals that are expected to sum to n*T
        ts=np.random.exponential(l,size=(self.n,int(2*self.T/l)))
        if self.verbose:
            print("Random exponential set of time intervals totalling {0} initialised".format(self.T))
            print("Elapsed time {0}.  Now cumulating...".format(time.time()-self.start_time))
        cts=np.array([np.cumsum(t) for t in ts])
        if self.verbose:
            print("Cumulative sum initialised to time {0}".format(self.T))
            print("Elapsed time {0}.  Now truncating....".format(time.time()-self.start_time)) 
        self.ts_dict[key]=self.truncate_to_T(cts)
        if self.verbose:
            print("Cumulative sums truncated")
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
        return np.array([[c for c in ct if c<self.T] for ct in cts])
        
    def convert_to_binary_time_series(self,poisson_process_interval_arrays):
        pps=poisson_process_interval_arrays
        ts_matrix=[[1 if i in pp else 0 for i in range(self.T)] for pp in pps]
        return np.array(ts_matrix)
    
    def convert_to_dense_time_series(self,binary_time_series):
        bts=binary_time_series
        dts=[[i for i in range(len(b)) if b[i]] for b in bts]
        return dts
        
def compare_sparse_to_dense():
    test_sparse_conversion=True
    
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
        dts=pp.convert_to_dense_time_series(bts)
        print(dts)
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
                   'dense' : False}
        print("Starting first test...")
        td=tsc.tweet_data([X1,X2],population_ps=[None,None,params_dict],disjoint_sets=True,delta=delta,axes=axes[0,:])
        start=time.time()
        #td.test_delta(max_delta=200,delta_step=5)
        td.display_Z_vals()
        #print(td.results)
        t2=time.time()-start
        print("Completed in {0}".format(time.time()-start))
        print("Second test")
        params_dict['dense']=True
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
    simplified=True
    if test:
        compare_sparse_to_dense()
        
    else:

        number=5000
        length=15000
        delta=int(np.sqrt(length))
        p=0.01
        l=int(1/p)
        Y1=l
        Y2=3
        Z=l


        sparse=False
        dense=not sparse
        
            
        lambdas = {'Y1': {'lambda':Y1,'baseline':False},
                'Y2': {'lambda':Y2,'baseline':False},
                #'Z': {'lambda':Z, 'baseline' : True}
                }
        params={'mu':5,'sigma':3}

        pp=poisson_process(number,length,lambdas=lambdas,params=params)
        
        
        
        p1=np.sum([len(x) for x in pp.ts_dict['Y1']])/(number*length)
        print(p1)
        p2=np.sum([len(x) for x in pp.ts_dict['Y2']])/(number*length)
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
        X1=pp.ts_dict['Y1']
        X2=pp.ts_dict['Y2']
#<<<<<<< HEAD
#        p1=sum([len(x) for x in X1])/(number*length)
#        p2=sum([len(x) for x in X2])/(number*length)
#        print(p2)
#        print(p1)
#        td=tsc.tweet_data([X1,X2],population_ps=[p1,p2,params_dict],disjoint_sets=True,delta=delta,axes=axes[0,:])
#=======

        #print(X1)
        #print(X2)
        if simplified:
            td=pc.tweet_data([X1,X2],params=params_dict,disjoint_sets=True,delta=delta,axes=axes[0,:])
        else:
            td=tsc.tweet_data([X1,X2],population_ps=[p1,p2,params_dict],disjoint_sets=True,delta=delta,axes=axes[0,:])
#>>>>>>> simplifying_code
        start=time.time()
        #td.test_delta(max_delta=200,delta_step=5)
        td.display_Z_vals(ax=axes[0][1])
        
#<<<<<<< HEAD
#        print("Starting reinitialised array test")
#        params_dict = {'T' : length,
#                   'n' : number,
#                   'Use population means' : True,
#                   'Use fixed means for setup' : False,
#                   'random seed' : None,
#                   'Test_mode' : False,
#                   'sparse' :  True}  
        
        #X1=np.array([np.random.choice([0,1],p=[1-p1,p1],size=[number,length])])
        #X2=np.array([np.random.choice([0,1],p=[1-p2,p2],size=[number,length])])
        #X1=pp.convert_to_binary_time_series(X1)
        #X2=pp.convert_to_binary_time_series(X2)
#        td=tsc.tweet_data([X1,X2],population_ps=[p1,p2,params_dict],disjoint_sets=True,delta=25,axes=axes[1,:])
#=======
#        print("Starting correlated test...")
#        X1=pp.ts_dict['X1']
#        X2=pp.ts_dict['X2']
#        #print(X1)
#        #print(X2)
#        if simplified:
#            td=pc.tweet_data([X1,X2],params=params_dict,disjoint_sets=True,delta=delta,axes=axes[0,:])
#        else:
#            td=tsc.tweet_data([X1,X2],population_ps=[p1,p2,params_dict],disjoint_sets=True,delta=delta,axes=axes[0,:])
#    
#        #td=tsc.tweet_data([X1,X2],population_ps=[None,None,params_dict],disjoint_sets=True,delta=25,axes=axes[1,:])
#>>>>>>> simplifying_code
#        td.display_Z_vals(ax=axes[1][1])
#        t2=time.time()-start
#        print("Completed in {0}".format(time.time()-start))


        
        
        
        