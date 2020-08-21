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
        #elif len(self.betas):
        self.population_probabilities = [1/b for b in self.betas]
        self.lag_params=lag_params
        self.t_series_array = []

        self.verbose=verbose
        self.event_time_series = None
        self.prior_process_object = prior_process_object
        if self.prior_process_object:
            self.create_lagging_poisson_process()
        elif len(betas):
            self.create_random_poisson_process()
        else:
            print("Unable to initialise - no beta paramters passed and no prior object passed")
        #if not self.n == len(self.betas):
        self.population_probabilities=[np.mean([len(t_series) for t_series in self.t_series_array])/self.T]*self.n
        #self.estimated_population_probabilities = [np.mean([len(t_series) for t_series in self.t_series_array])/self.T]*self.n
        self.update()        
    
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
            self.lag_params['mus']=mus
            sigmas = [sigmas[0]]*self.n
            self.lag_params['sigmas']=sigmas
            
        if not (len(prior_array) == self.n):
            print("Creating lagging time series population based on single event time series")
            self.event_time_series = prior_array[0]
            prior_array = [self.event_time_series]*self.n
            self.population_probabilities=[self.prior_process_object.population_probabilities[0]]*self.n
        else:
            print("Creating lagging time series population based on prior population the same size")                      

        lagging_population = [[event+np.random.normal(mus[i],sigmas[i]) for event in prior_array[i]] for i in range(self.n)]
        self.t_series_array = self.truncate_to_T(lagging_population)

    def update(self,pp_to_copy=None):
        if pp_to_copy:
            ts1 = self.t_series_array
            p1 = self.population_probabilities
            ts2 = pp_to_copy.t_series_array
            p2 = pp_to_copy.population_probabilities
        
            assert len(p1) == len(p2)
            assert self.T == pp_to_copy.T
            assert len(ts1) == len(ts2)
        
            combined_ts = [np.sort(list(set(np.append(t,ts2[i])))) for i,t in enumerate(ts1)]
            self.t_series_array = combined_ts
            combined_ps = [p1[i] + p2[i] - p1[i]*p2[i] for i in range(len(p1))]
            self.population_probabilities = combined_ps
        
        #if self.prior_process_object:
        #    poisson_lag_params = list(zip(self.lag_params['mus'],self.lag_params['sigmas']))
        #else:
        #    poisson_lag_params=[None]*self.n
        self.time_series_objects = [pc.time_series(self.t_series_array[i],self.population_probabilities[i],self.T,
                                    self.event_time_series) for i in range(self.n)]
        
    
    def display(self):
        print(self.t_series_array)
        print(self.population_probabilities)
    
    def pop_stats(self):
        p = np.mean([len(t_series) for t_series in self.t_series_array])/self.T
        return "From {0} time series length {1}, measured population event probability = {2}"\
                .format(len(self.t_series_array),self.T,p)
    
    def truncate_to_T(self,cts):
        return np.array([np.sort(list(set([int(round(c)) for c in ct if (c<self.T+0.5 and c>0)]))) for ct in cts])
        
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
            if self.verbose:
                print("Key {0} links to poisson process array:".format(key))
                self.poisson_process_dict[key].display()
        
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
            self.poisson_process_dict[key]=self.combine(key,previous_key)

            
    def combine(self,key_to_update,key_to_copy):
        pp_to_update=self.poisson_process_dict[key_to_update]
        pp_to_copy=self.poisson_process_dict[key_to_copy]
        pp_to_update.update(pp_to_copy)
        #ts1 = pp_to_update.t_series_array
        #p1 = pp_to_update.population_probabilities
        #ts2 = pp_to_copy.t_series_array
        #p2 = pp_to_copy.population_probabilities
        
        #assert len(p1) == len(p2)
        #assert pp_to_update.T == pp_to_copy.T
        #assert len(ts1) == len(ts2)
        #assert pp_to_update.T == self.T
        
        #combined_ts = [np.sort(list(set(np.append(t,ts2[i])))) for i,t in enumerate(ts1)]
        #pp_to_update.t_series_array = combined_ts
        #combined_ps = [p1[i] + p2[i] - p1[i]*p2[i] for i in range(len(p1))]
        #pp_to_update.population_probabilities = combined_ps
        #return pp_to_update
    
    def display(self,stats=True,full=False):
        print("\n")
        if full:
            for key in self.poisson_process_dict.keys():
                print("Array for key {0} is :".format(key))
                self.poisson_process_dict[key].display()
                print("\n")
        elif stats:
            for key in self.poisson_process_dict.keys():
                pps = self.poisson_process_dict[key]
                probs = pps.population_probabilities
                print("For key {0}: {1}".format(key,pps.pop_stats()))
                print("Theoretical population mean is {0} with std {1}\n".format(np.mean(probs),np.std(probs)))
            
    def randomly_mix_populations(self,keys=[]):
        jumbled_time_series_obj_array = []
        if not len(keys):
            for key in self.poisson_process_dict.keys():
                
                pass


def create_paired_noisy_lagging_time_series(length,number,verbose=False):
    population_params = {'Random A' : {'n': number,
                                       'betas': [length/100],
                                       'prior poisson process': [],
                                       'lag parameters' : {'mus': None,'sigmas': None},
                                       'combine with' : None,
                                       },
                         'Random B' :  {'n': number,
                                       'betas': [length/100],
                                       'prior poisson process': [],
                                       'lag parameters' : {'mus': None,'sigmas': None},
                                       'combine with' : None,
                                       },
                         'Random Z' :  {'n': number,
                                       'betas': [length/2],
                                       'prior poisson process': None,
                                       'lag parameters' : {'mus': None,'sigmas': None},
                                       'combine with' : None
                                       },
                         }
    mpp = mixed_poisson_populations(length,population_params,verbose)
    
    if verbose:
        mpp.display(stats=True)
    
    mpp.population_params['Random Z*'] =  {'n': number,
                                       'betas': [],
                                       'prior poisson process': mpp.poisson_process_dict['Random Z'],
                                       'lag parameters' : {'mus': [5],'sigmas': [0]},
                                       'combine with' : None,}
    mpp.create_poisson_processes('Random Z*')
    if verbose:
        mpp.display(full=True)
        print("Now combining...:")
    mpp.combine(key_to_update='Random Z',key_to_copy='Random A')
    mpp.combine(key_to_update='Random Z*',key_to_copy='Random B')
    
    if verbose:
        mpp.display(full=True)
    return mpp
             
def initialise_population_params(number,betas=[],prior_poisson_process=None,lag_parameters={},combine_with = None):
    return {'n': number,'betas': betas,'prior poisson process': prior_poisson_process,
            'lag parameters' : lag_parameters,'combine with' : combine_with,}  

def initialise_lag_params(number,means_dist='Poisson',means_lambda = 100,sigmas_scaling=1000):
    if means_dist=='Poisson':
        means = np.random.poisson(lam=means_lambda,size=number)
        sigmas = [np.random.poisson(lam=m*sigmas_scaling) for m in means]
        return {'mus' : means, 'sigmas' : sigmas}
    
def create_single_array_of_mixed_populations(length=10000,number=50):
    pp_event = poisson_process(1,length,[20])
    pp_event.display()
    lag_params=initialise_lag_params(number)
    #lag_params={'mus' : np.random.choice([2,20],size=number), 'sigmas' : np.random.choice([1],size=number)}
    population_params = {'Pop 1' : initialise_population_params(number,prior_poisson_process=pp_event,lag_parameters=lag_params)}
    mpp = mixed_poisson_populations(length,population_params)
    mpp.display(stats=True)
    return mpp
    


        
        
#print(initialise_lag_params(20))
    
    
    
if __name__=='__main__':
    length=10000
    pair=False
    if pair:

        number=5000
        mpp_pair_of_populations = create_paired_noisy_lagging_time_series(length,number,verbose=False)
        mpp_pair_of_populations.display()
        ts_obj1 = mpp_pair_of_populations.poisson_process_dict['Random A'].time_series_objects
        ts_obj2 = mpp_pair_of_populations.poisson_process_dict['Random B'].time_series_objects
    else:
        number=50
        mpp=create_single_array_of_mixed_populations(length=length,number=number)
        ts_obj1=mpp.poisson_process_dict['Pop 1'].time_series_objects
        ts_obj2=[]
    #ts_array1 = mpp_pair_of_populations.poisson_process_dict['Random A'].t_series_array
    #ts_array2 = mpp_pair_of_populations.poisson_process_dict['Random B'].t_series_array
    #assert(len(ts_array1)==len(probs1))
    #assert(len(ts_array2)==len(probs2))
    
    #ts_obj1 = [pc.time_series(ts_array1[i],probs1[i],length) for i in range(len(probs1))]
    #ts_obj2 = [pc.time_series(ts_array2[i],probs2[i],length) for i in range(len(probs2))]
    #print(ts_obj1[0].t_series)
    
    ts_matrices=[ts_obj1,ts_obj2]
    
    
    
    if True:
        #p1=np.sum([len(x) for x in pp.poisson_process_dict['X1']])/(number*length)
        #print(p1)
        #p2=np.sum([len(x) for x in pp.poisson_process_dict['X2']])/(number*length)
        #print(p2)
        #known_probs_array = [[p1]*number,[p2]*number]
        #print(known_probs_array)
        #tweet_matrices = [pp.poisson_process_dict['X1'], pp.poisson_process_dict['X2']]
        
        #ts_matrices=[[pc.time_series(tweet_matrices[i][j],known_probs_array[i][j],length) for j in range(number)] 
        #                     for i in range(len(tweet_matrices))]
        
        
        sparse=False
        dense=True
        delta=np.sqrt(length)
        params_dict = {'T' : length,
                   'n' : number,
                   'p1' : 0.1,
                   'p2' : 0.3, 
                   'Use population means' : True,
                   'Use fixed means for setup' : False, # must be true if using population means for z-score calculations
                   'random seed' : None,
                   'Test_mode' : False,
                   'sparse' :  sparse,
                   'dense' : dense}
        
        f,axes=plt.subplots(2,2)
        
        print("Starting poisson test...")
        #X1=pp.poisson_process_dict['X1']
        #X2=pp.poisson_process_dict['X2']

        td=pc.tweet_data(ts_matrices,params=params_dict,disjoint_sets=pair,delta=delta,axes=axes[0,:])

        start=time.time()
        #td.test_delta(max_delta=200,delta_step=5)
        td.display_Z_vals(ax=axes[0][1])
        



        
        
        
        