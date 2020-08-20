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
    
    def __init__(self,number,length,betas,prior_process_object = None,lag_params={},verbose=False):
        self.start_time=time.time()
        self.T=length
        self.n=number
        self.betas=betas
        if len(self.betas) and not self.n == len(self.betas):
            print("Expected {0} beta values, but only {1} values passed.  Initialising all time series with beta parameter {2}".format(self.n,len(betas),betas[0]))
            self.betas = [betas[0]]*self.n
        self.lag_params=lag_params
        self.t_series_array = []
        self.population_probabilities = [1/b for b in self.betas]
        self.verbose=verbose
        self.event_time_series = []
        self.prior_process_object = prior_process_object
        if self.prior_process_object:
            self.create_lagging_poisson_process()
        elif len(betas):
            self.create_random_poisson_process()
        else:
            print("Unable to initialise - no beta paramters passed and no prior object passed")

    def create_random_poisson_process(self):
        # create sequence of time intervals that are expected to sum to n*T
        ts=[np.random.exponential(b,size=(int(2*self.T/b))) for b in self.betas]
        #ts=np.random.exponential(l,size=(self.n,int(2*self.T/l)))        
        if self.verbose:
            print("Random exponential set of time intervals totalling {0} initialised".format(self.T))
            print("Elapsed time {0}.  Now cumulating...".format(time.time()-self.start_time))
        cts=np.array([np.cumsum(t) for t in ts])

        if self.verbose:
            print("Cumulative sum initialised to time {0}".format(self.T))
            print("Elapsed time {0}.  Now truncating....".format(time.time()-self.start_time)) 
        self.t_series_array=self.truncate_to_T(cts)


        if self.verbose:
            print("Cumulative sums rounded, ordered and truncated")
    
    def create_lagging_poisson_process(self):
        mus = self.lag_params['mus']
        sigmas = self.lag_params['sigmas']
        prior_array=self.prior_process_object.t_series_array
        self.population_probabilities=self.prior_process_object.population_probabilities
        
        if not (self.n == len(mus) and self.n == len(sigmas)):
            print("Expected {0} mu/sigma values but {1}/{2} passed.  All time series allocated mean/std lag {3}/{4}".format(self.n,len(mus),len(sigmas),mus[0],sigmas[0]))
            mus = [mus[0]]*self.n
            sigmas = [sigmas[0]]*self.n
            
        if not len(prior_array) == self.n:
            print("Creating lagging time series population based on single event time series")
            self.event_time_series = prior_array[0]
            prior_array = [self.event_time_series]*self.n
            self.population_probabilities=[self.prior_process_object.population_probabilties[0]]*self.n
                               
        #noise_time_series = self.poisson_process_dict['noise_time_series_key']
        #print(prior_array)
        #print(mus)
        #print(sigmas)
        lagging_population = [[event+np.random.normal(mus[i],sigmas[i]) for event in prior_array[i]] for i in range(self.n)]
        self.t_series_array = self.truncate_to_T(lagging_population)
#        self.poisson_process_dict['Lagging population'] = self.combine(lagging_population,noise_time_series)
#        self.poisson_process_dict['Lagging population mean'] = np.mean([len(ts) for ts in self.poisson_process_dict['Lagging population']])/self.T

    def display(self):
        print(self.t_series_array)
        print(self.population_probabilities)
    
    def pop_stats(self):
        p = np.mean([len(t_series) for t_series in self.t_series_array])/self.T
        return "From {0} time series length {1}, measured population event probability = {2}"\
                .format(len(self.t_series_array),max([t[-1] for t in self.t_series_array]),p)
    
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
    
    
class mixed_poisson_populations():
    
    def __init__(self,length,population_params,verbose=False):
        self.start_time=time.time()
        self.T=length
        self.population_params=population_params
        self.verbose=verbose
        
        self.poisson_process_dict={}
        
        self.initialise_distinct_populations()
        
    
    def initialise_distinct_populations(self):
        for key in self.population_params.keys():
            self.create_poisson_processes(key)
            print("Elapsed time: {0}".format(time.time()-self.start_time))
            if verbose:
                print("Key {0} links to poisson process array:".format(key))
                self.poisson_process_dict[key].display()
#        if 'Z' in self.poisson_process_dict.keys():
#            self.poisson_process_dict['X1']=self.combine(self.poisson_process_dict['Y1'],self.poisson_process_dict['Z'])
#        if 'Z_star' in self.poisson_process_dict.keys():
#            self.poisson_process_dict['X2']=self.combine(self.poisson_process_dict['Y2'],self.poisson_process_dict['Z_star'])

      
#    def create_lagging_population(self,prior_process_object,noise_time_series_key):##
#
#        mus = self.params['mus']
#        sigmas = self.params['sigmas']
#        noise_time_series = self.poisson_process_dict['noise_time_series_key']
#        
#        lagging_population = [[event+np.random.normal(mus[i],sigmas[i]) for event in event_time_series] for i in range(self.n)]
#        lagging_population = self.truncate_to_T(lagging_population)
#        self.poisson_process_dict['Lagging population'] = self.combine(lagging_population,noise_time_series)
#        self.poisson_process_dict['Lagging population mean'] = np.mean([len(ts) for ts in self.poisson_process_dict['Lagging population']])/self.T
        
        
    def create_poisson_processes(self,key):
        print("Initialising for key {0}".format(key))
        betas=self.population_params[key]['betas']
        prior_process_object = self.population_params[key].get('prior poisson process')
        lag_params = self.population_params[key].get('lag parameters')
        
        self.poisson_process_dict[key] = poisson_process(self.population_params[key]['n'],self.T,betas,
                                 prior_process_object = prior_process_object, lag_params = lag_params,verbose=self.verbose)
        previous_key = self.population_params[key].get('combine with')
        if previous_key:
            print("Now combining poisson processes {0} with {1}".format(key,previous_key))
            if verbose:
                print("About to overwrite processes for key {0} - currently:".format((key)))
                self.poisson_process_dict[key].display()
            self.poisson_process_dict[key]=self.combine(self.poisson_process_dict[key],self.poisson_process_dict[previous_key])
 #        
# create delayed time series if required  
#        if self.population_params[key]['Create lagging time series']:
#            mus=self.population_params[key].get('mus')
#            sigmas=self.population_params[key].get('sigmas')#
#
#            cts_star=[[c+np.random.normal(mus[i],sigmas[i]) for c in cts[i]] for i in range(len(cts))]
#            if self.verbose:
#                print("Lag added.  Elapsed time {0}.  Now truncating...".format(time.time()-self.start_time))
#            self.poisson_process_dict[key+'_star']=self.truncate_to_T(cts_star)
            
            
    def combine(self,pp_to_update,pp_to_copy):
        ts1 = pp_to_update.t_series_array
        p1 = pp_to_update.population_probabilities
        ts2 = pp_to_copy.t_series_array
        p2 = pp_to_copy.population_probabilities
        
        assert len(p1) == len(p2)
        assert pp_to_update.T == pp_to_copy.T
        assert len(ts1) == len(ts2)
        assert pp_to_update.T == self.T
        
        combined_ts = [np.sort(list(set(np.append(t,ts2[i])))) for i,t in enumerate(ts1)]
        pp_to_update.t_series_array = combined_ts
        combined_ps = [p1[i] + p2[i] - p1[i]*p2[i] for i in range(len(p1))]
        pp_to_update.population_probabilities = combined_ps
        return pp_to_update
    
    def display(self,stats=True,full=False):
        print("\n")
        if full:
            for key in self.poisson_process_dict.keys():
                self.poisson_process_dict[key].display()
        elif stats:
            for key in self.poisson_process_dict.keys():
                pps = self.poisson_process_dict[key]
                probs = pps.population_probabilities
                print("For key {0}: {1}".format(key,pps.pop_stats()))
                print("Theoretical population mean is {0} with std {1}\n".format(np.mean(probs),np.std(probs)))
            
    def randomly_mix_populations(self):
        pass
        
            
if __name__=='__main__':
    
    length = 500
    population_params = {'Random A' : {'n': 10,
                                       'betas': [length/5],
                                       'prior poisson process': [],
                                       'lag parameters' : {'mus': None,'sigmas': None},
                                       'combine with' : None,
                                       },
                         'Random B' :  {'n': 10,
                                       'betas': [length/10],
                                       'prior poisson process': [],
                                       'lag parameters' : {'mus': None,'sigmas': None},
                                       'combine with' : None,
                                       },
    
                                       
                         }
    
    verbose = False
    mpp = mixed_poisson_populations(length,population_params,verbose)
    mpp.display(stats=True,full=False)
    mpp.population_params['Random Z'] = {'n': 10,
                                       'betas': [],
                                       'prior poisson process': mpp.poisson_process_dict['Random A'],
                                       'lag parameters' : {'mus': [5],'sigmas': [3]},
                                       'combine with' : None,}
    mpp.create_poisson_processes('Random Z')
    mpp.display()
    print(mpp.poisson_process_dict['Random Z'].t_series_array)
    print("\n")
    print(mpp.poisson_process_dict['Random A'].t_series_array)
    
    if False:
        number=50
        length=1000
        delta=int(np.sqrt(length))
        #p=0.01
        #l=int(1/p)

        Y1=[50]*number
        #print(Y1)
        Y2=[20]*number
        Z=[1000]*number


        sparse=False
        dense=not sparse
        
        betas = Y1    
        #betas = {'Y1': {'betas':Y1,'baseline':False},
                #'Y2': {'beta':Y2,'baseline':False},
                #'Z': {'beta':Z, 'baseline' : False}
                #}
        params={'mus':[5]*number,'sigmas':[0.1]*number}

        pp=poisson_process(number,length,betas=betas,params=params)
        print(pp.poisson_process_dict.keys())
        
        
        p1=np.sum([len(x) for x in pp.poisson_process_dict['X1']])/(number*length)
        print(p1)
        p2=np.sum([len(x) for x in pp.poisson_process_dict['X2']])/(number*length)
        print(p2)
        known_probs_array = [[p1]*number,[p2]*number]
        #print(known_probs_array)
        tweet_matrices = [pp.poisson_process_dict['X1'], pp.poisson_process_dict['X2']]
        
        ts_matrices=[[pc.time_series(tweet_matrices[i][j],known_probs_array[i][j],length) for j in range(number)] 
                             for i in range(len(tweet_matrices))]
        
        
        
        params_dict = {'T' : length,
                   'n' : number,
                   'p1' : p1,
                   'p2' : p2, 
                   'Use population means' : False,
                   'Use fixed means for setup' : False, # must be true if using population means for z-score calculations
                   'random seed' : None,
                   'Test_mode' : False,
                   'sparse' :  sparse,
                   'dense' : dense}
        
        f,axes=plt.subplots(2,2)
        
        print("Starting poisson test...")
        #X1=pp.poisson_process_dict['X1']
        #X2=pp.poisson_process_dict['X2']

        td=pc.tweet_data(ts_matrices,params=params_dict,disjoint_sets=False,delta=delta,axes=axes[0,:])

        start=time.time()
        #td.test_delta(max_delta=200,delta_step=5)
        td.display_Z_vals(ax=axes[0][1])
        



        
        
        
        