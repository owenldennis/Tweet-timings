0# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 16:44:59 2019

@author: owen
"""

import numpy as np
import time
#import time_series_correlation as tsc
import pointwise_correlation as pc
import matplotlib.pyplot as plt
import pandas as pd



class poisson_process():
    """ poisson process objects contain *number* time series objects with T=*length*
        If a poisson_process_object is passed (as prior_process_object) then:
            the current instance of poisson_process will contain an array of lagging time series objects
            these are based either on n unique time series (if prior_process_object has exactly n time series objects)
            or on a single event times series (the first time series in prior_process_object)
        If no poisson_pocess_object is passed then at least one beta value must be passed in parameter betas:
            the betas array provides the parameter(s) for the exponential distribution on which each poisson process is based.
            if n beta values are passed, one time series is created for each.
            if not, the first value in the betas array is used to parametrise all time series.
        Once all poisson processes are created and turned into time series:
            update method is called:
                each time series is used to initialise a time_series object with a theoretical expected incidence calculated if possible
    
    """
    
    def __init__(self,number,length,betas,name=None,cutoff=1.,prior_process_object = None,lag_params={},verbose=False):
        self.start_time=time.time()
        self.name=name
        self.T=length
        self.n=number
        self.betas=betas
        self.beta_fixed=False
        if len(self.betas):
            for i,beta in enumerate(self.betas):
                if beta<cutoff: 
                    if verbose:
                        print("Found beta={0} in passed parameters - resetting to {1}".format(beta,cutoff))
                    self.betas[i]=cutoff
        
        # ensure that, if any beta values are passed, the size of the array matches the size of the population
        if len(self.betas) and not self.n == len(self.betas):
            print("Expected {0} beta values, but only {1} values passed.  Initialising all time series with beta parameter {2}".format(self.n,len(betas),betas[0]))
            self.betas = [betas[0]]*self.n
            self.beta_fixed=True
        # initialise population probabilities - will be changed after processes have been created
        # to reflect rounding to integers and consequent reduction in event probability
        self.population_probabilities = [1/b for b in self.betas]
        
        self.lag_params=lag_params
        self.t_series_array = []
        self.verbose=verbose
        self.event_time_series = None
        self.prior_process_object = prior_process_object
        
        # either initialise based on the poisson process population passed, or create a new random poisson process population
        if self.prior_process_object:
            self.create_lagging_poisson_process()
            ##################################################################
            print("Initialising population {1} based on a prior object with event beta {0}".format(self.prior_process_object.betas,self.name))
            print("Mean lags for population {0} are {1}".format(self.name,self.lag_params['mus']))
            
        elif len(betas):
            ##################################################################
            print("Initialising a random population {0}, size {1}, with betas {2}".format(self.name,self.n,self.betas))
            self.create_random_poisson_process()
        else:
            print("Unable to initialise - no beta paramters passed and no prior object passed")
                
        # call update to initialise time series objects
        self.update()        
    
    def create_random_poisson_process(self):
        """
        called if no prior_process_object is passed to _init_
        creates n random time series using n poisson processes parametised by n entries in self.betas
        stores theoretical mean event probability 

        Returns
        -------
        None.

        """
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
        
        #theoretical population probabilities have changed due to rounding
        self.population_probabilities = [1-np.exp(-1/b) for b in self.betas]


        if self.verbose:
            print("Cumulative sums rounded, ordered and truncated")
        
        
        ############################################
        print("Betas for population {0} are {1}".format(self.name,self.betas))
        

        ##########################################
    
    def create_lagging_poisson_process(self):
        """
        called if a prior_process_object is passed to __init

        either
            creates a set of lagging time series based on a single event time series, 
        or
            adds a lag to each of the n time series passed inside prior_process_object
            
        the lag parameters are stored in the self.lag_params dictionary
        lags are then taken from poisson distributions with means passed in self.lag_params dictionary

        Returns
        -------
        None.

        """
        mus = self.lag_params['mus']
        #sigmas = self.lag_params['sigmas']
        prior_array=self.prior_process_object.t_series_array
        self.population_probabilities=self.prior_process_object.population_probabilities
        
        if not (self.n == len(mus)):# and self.n == len(sigmas)):
            # if there are the wrong number of mus (mean lag times for each time series) then the first mu value is used for all lags
            print("Expected {0} mu values but {1}/{2} passed.  All time series allocated mean/std lag {3}/{4}".format(self.n,len(mus),len([]),mus[0],0))
            mus = [mus[0]]*self.n
            self.lag_params['mus']=mus
            #sigmas = [sigmas[0]]*self.n
            #self.lag_params['sigmas']=sigmas
            
        if not (len(prior_array) == self.n):
            # if the prior_process_object does not contain n time series objects:
            # - the first entry is turned into an event_time_series object and its time series is replicated n times
            # - population_probabilities updated accordingly
            print("Creating lagging time series population based on single event time series")
            self.event_time_series = pc.event_time_series(prior_array[0])
            prior_array = [self.event_time_series.t_series]*self.n
            self.population_probabilities=[self.prior_process_object.population_probabilities[0]]*self.n
        else:
            # if there are n time series objects, each is used in turn to create a lagging time series population
            print("Creating lagging time series population based on prior population the same size")
        #self.estimated_population_probabilities = [np.mean([len(t_series) for t_series in self.t_series_array])/self.T]*self.n
            if len(self.prior_process_object.population_probabilities)==self.n:
                self.population_probabilities=self.prior_process_object.population_probabilities
            else:
                print("Warning - unable to calculate theoretical probabilities for each individual time series. Estimating from total events in the population.")  
                print("This will give unreliable results if calculating z-scores using population means unless all time series come from the same population")
                self.population_probabilities=[np.mean([len(t_series) for t_series in self.t_series_array])/self.T]*self.n
                                  
        # create self.n lagging time series by adding a lag drawn from a poisson distribution to the event time series
        # there are self.n values held in the mus array - these are the rates for each poisson distribution
        # this ensures the lag is not constant for a specific time series but will have an expected value of the rate
        lagging_population = [[event+np.random.poisson(mus[i]) for event in prior_array[i]] for i in range(self.n)]
        self.t_series_array = self.truncate_to_T(lagging_population)
        
        ###############################################
        
        print("The poisson rates for lagging population {0} are {1}".format(self.name,mus))
        
        
        ###############################################
        
    def update(self,pp_to_copy=None):
        """
        
        update is called every time a poisson process object is initialised
        initalises the time series objects that contain each time series
        attempts to caclulate the theoretical population probabilites from which each time series is drawn
        
        Parameters
        ----------
        pp_to_copy : poisson_process_object, optional
            DESCRIPTION. If a poisson_process object is passed:
                -the union of each time series with the current object's time series in turn replaces the current object's time series
                -this is used to add noise to a time series
                -population probabilities are updated accordingly
        
                The default is None.

        Returns
        -------
        None.

        """
        
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
        
        # if no poisson_process object is passed, population probabilities are adjusted 
        if True:#self.beta_fixed:
        #    # better estimate is averaging over the population if a single beta value is used - theroetical measure has issues!
            self.population_probabilities=[np.mean([len(t_series) for t_series in self.t_series_array])/self.T]*self.n
           
        # time_series objects are (re)-initialised
        self.time_series_objects = [pc.time_series(self.t_series_array[i],self.population_probabilities[i],self.T,self.name,
                                    self.event_time_series) for i in range(self.n)]
        
    
    def display(self):
        print(self.t_series_array)
        print(self.population_probabilities)
    
    def pop_stats(self):
        p = np.mean([len(t_series) for t_series in self.t_series_array])/self.T
        return "From {0} time series length {1}, measured population event probability = {2}"\
                .format(len(self.t_series_array),self.T,p)
    
    def truncate_to_T(self,cts):
        return [np.sort(list(set([int(round(c)) for c in ct if (c<self.T+0.5 and c>0)]))) for ct in cts]
        
    def convert_to_binary_time_series(self,dense_time_series_array):
        dts=dense_time_series_array
        ts_matrix=[[1 if i in ts else 0 for i in range(self.T)] for ts in dts]
        return np.array(ts_matrix)
    
    def convert_to_dense_time_series(self,binary_time_series):
        bts=binary_time_series
        dts=[[i for i in range(len(b)) if b[i]] for b in bts]
        return np.array(dts)

    
    
class mixed_poisson_populations():
    """
    mixed_poisson_population objects initialise multiple populations of time_series objects
    The population_params dictionary parameter contains all the parameters required.  The first level keys are the population names.
    The'randomly_mix_populations' method extracts all time_series objects for each population and randomly shuffles them
    
    
    """
    
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
        """
        For each population, create a poisson_processes object.
        If this population is then to be replaced by the union of this and a previous object,
        the combine method is called.
        
        Parameters
        ----------
        key : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        print("Initialising for key {0}".format(key))
        betas=self.population_params[key]['betas']
        
        # if the whole sample is to be drawn from a single population then only use one beta parameter
        if self.population_params[key].get('use fixed means for setup'):
            betas=[betas[0]]
        
        # initialise the population for this key
        prior_process_object = self.population_params[key].get('prior poisson process')
        lag_params = self.population_params[key].get('lag parameters')
        
        self.poisson_process_dict[key] = poisson_process(self.population_params[key]['n'],self.T,betas,name=key,
                                 prior_process_object = prior_process_object, lag_params = lag_params,verbose=self.verbose)
        previous_key = self.population_params[key].get('combine with')
        
        # if this population is to be replaced by its union with a previous one then call 'combine' method
        if previous_key:
            print("Now combining poisson processes {0} with {1}".format(key,previous_key))
            if self.verbose:
                print("About to overwrite processes for key {0} - currently:".format((key)))
                self.poisson_process_dict[key].display()
            self.combine(self.poisson_process_dict[key],self.poisson_process_dict[previous_key])

            
    def combine(self,pp_to_update,pp_to_copy):
        """
        Uses the update method in poisson_process class to create the pointwise union of two arrays of time_series objects
        These will be stored as the array of time_series objects within pp_to_update

        Parameters
        ----------
        pp_to_update : TYPE poisson_processes object
            DESCRIPTION. the poisson_processes object that will have its time_series objects updated 
        pp_to_copy : TYPE poisson_process object
            DESCRIPTION.  The poisson process object that will be added into pp_to_update

        Returns
        -------
        None.

        """
        pp_to_update.update(pp_to_copy)

    
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
        """

        Parameters
        ----------
        keys : TYPE, optional : if keys are passed then only these populations are mixed
            DESCRIPTION. The default is [].

        Returns
        -------
        jumbled_ts_obj_array : TYPE array 
            DESCRIPTION. shuffled array of all time_series objects from each population specified (or all if none specified)

        """
        jumbled_ts_obj_array = []
        if not len(keys):
            # select all keys if none specified
            keys=self.poisson_process_dict.keys()        
        for key in keys:
            jumbled_ts_obj_array=np.append(jumbled_ts_obj_array,
                                           self.poisson_process_dict[key].time_series_objects,
                                           axis=0)
        
        np.random.shuffle(jumbled_ts_obj_array)
        return jumbled_ts_obj_array


def create_paired_noisy_lagging_time_series(length,number,verbose=False):
    """
    Initialises a mixed_poisson_population object consisting of four populations:
        'Random A' and 'Random B' are random poisson_processes objects with beta parameters assigned below
        'Random Z' is initialised as a random poisson_process object which is then updated with 'Random A'
        'Random Z*' is a lagging poisson_process object based on the initialised 'Random Z' and then updated with 'Random B'
        
    A and B are effectively the noise of the lagging time series Z and Z* which can then be analysed. 
        

    Parameters
    ----------
    length : TYPE integer, length of time series
        DESCRIPTION.
    number : TYPE integer, number of time series
        DESCRIPTION.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    mpp : TYPE
        DESCRIPTION.

    """
    population_params = {'Random A' : {'n': number,
                                       'betas': np.random.poisson(10,size=number),
                                       'prior poisson process': None,
                                       'lag parameters' : {'mus': None},
                                       'combine with' : None,
                                       },
                         'Random B' :  {'n': number,
                                       'betas': np.random.poisson(5,size=number),
                                       'prior poisson process': None,
                                       'lag parameters' : {'mus': None},
                                       'combine with' : None,
                                       },
                         'Random Z' :  {'n': number,
                                       'betas': np.random.poisson(1000,size=number),
                                       'prior poisson process': None,
                                       'lag parameters' : {'mus': None},
                                       'combine with' : None
                                       },
                         }
    mpp = mixed_poisson_populations(length,population_params,verbose)
    
    if verbose:
        mpp.display(stats=True)
    
    mpp.population_params['Random Z*'] =  {'n': number,
                                       'betas': [],
                                       'prior poisson process': mpp.poisson_process_dict['Random Z'],
                                       'lag parameters' : {'mus': [5]},
                                       'combine with' : None,}
    mpp.create_poisson_processes('Random Z*')
    if verbose:
        mpp.display(full=True)
        print("Now combining...:")
    pp_to_update1=mpp.poisson_process_dict['Random Z']
    pp_to_copy1=mpp.poisson_process_dict['Random A']
    pp_to_update2=mpp.poisson_process_dict['Random Z*']
    pp_to_copy2=mpp.poisson_process_dict['Random B']
    #mpp.combine(key_to_update='Random Z',key_to_copy='Random A')
    #mpp.combine(key_to_update='Random Z*',key_to_copy='Random B')
    mpp.combine(pp_to_update1,pp_to_copy1)
    mpp.combine(pp_to_update2,pp_to_copy2)
    
    if verbose:
        mpp.display(full=True)
    return mpp
             
#def initialise_population_params(number,betas=[],prior_poisson_process=None,lag_parameters={},combine_with = None):
#    return {'n': number,'betas': betas,'prior poisson process': prior_poisson_process,
#            'lag parameters' : lag_parameters,'combine with' : combine_with,}  

#def initialise_lag_params(number,means_dist='Poisson',mean_lag = 10):
#    if means_dist=='Poisson':
#        means = np.random.poisson(lam=mean_lag,size=number)
#        #sigmas = [np.random.poisson(lam=m*sigmas_scaling) for m in means]
#        return {'mus' : means}

def turn_lists_into_dicts(sizes,keys=[],event_probs=[],mean_lags=[],noise_probs=[]):
    """
    Given lists of parameters, they are formatted into a dictionary for multiple_poisson_processes to use
    If any optional lists are passed their length should match that of sizes parameter

    Parameters
    ----------
    sizes : TYPE list of integers
        DESCRIPTION. size of each population
    keys : TYPE, optional list of strings
        DESCRIPTION. The default is [].names of populations
    event_probs : TYPE, optional list of floats
        DESCRIPTION. The default is []. event incidence for each population
    mean_lags : TYPE, optional list of floats
        DESCRIPTION. The default is [].mean lags for each population
    noise_probs : TYPE, optional list of floats
        DESCRIPTION. The default is []. mean noise incidence for each population

    Returns
    -------
    dict
        DESCRIPTION.

    """
    if not len(keys):
        keys = np.random.choice(range(len(sizes)),size=len(sizes),replace=False)
        #print(len(keys))
        #print(len(sizes))
    if not len(event_probs):
        event_probs=np.random.uniform(0.1,0.05,size=len(sizes))
    if not len(mean_lags):
        mean_lags=np.random.uniform(2,20,size=len(sizes))
    if not len(noise_probs):
        noise_probs=np.random.uniform(0.001,0.005,size=len(sizes))
    assert len(sizes)==len(mean_lags)
    assert len(event_probs)==len(noise_probs)
    return {str(key):{'n' : sizes[i],'event probability' : event_probs[i], 'mean lag' : mean_lags[i],'noise probability' : noise_probs[i]} for i,key in enumerate(keys)}
        
def initialise_multiple_populations(length=10000,sizes=[50],keys=[],event_probs=[],mean_lags=[],noise_probs=[],use_fixed_means=False,verbose=False):
    params_for_populations=turn_lists_into_dicts(sizes,keys,event_probs,mean_lags,noise_probs)
    if verbose:
        print(params_for_populations)
    population_params={}
    for key in params_for_populations:
        number = params_for_populations[key]['n']
        p_event=params_for_populations[key]['event probability']
        event_beta=1/p_event
        mean_lag=params_for_populations[key]['mean lag']
        p_noise=params_for_populations[key]['noise probability']
        noise_beta=1/p_noise
        if use_fixed_means:
            betas=[noise_beta]
        else:
            betas=np.random.uniform(2,noise_beta*2,size=number)
        population_params[key+'_noise']={'n' : number,
                                    'betas' : betas
                                }
        population_params[key]={'n' : number,
                                'betas' : [],
                                'prior poisson process' : poisson_process(1,length,[event_beta]),
                                'lag parameters' : {'mus' : np.random.randint(1,mean_lag*2,size=number)},
                                                    #np.random.poisson(lam=mean_lag,size=number)},
                                'combine with' : key+'_noise'
                                }
        #print(population_params)
        #print("\n")
    
    
    #Copy parameters to csv file in TEMP directory
    df=pd.DataFrame.from_dict(population_params,orient='index')
    df['prior process beta'] = df['prior poisson process'].apply(lambda x: None if type(x)==float else x.betas)
    df['mean lag list']=df['lag parameters'].apply(lambda x: None if type(x)==float else x['mus'])
    df=df.drop(['prior poisson process','lag parameters'],axis=1)
    df.to_csv("{0}\population_parameters.csv".format(pc.TEMP_DIR))
    
    
    
    mpp = mixed_poisson_populations(length,population_params,verbose=verbose)
    mpp.display(stats=True)
    return mpp


if __name__=='__main__':
    length=10000
    pair=False
    if pair:
        # pair of lagging populations created with noise added
        number=5000
        mpp_pair_of_populations = create_paired_noisy_lagging_time_series(length,number,verbose=False)
        mpp_pair_of_populations.display()
        ts_obj1 = mpp_pair_of_populations.poisson_process_dict['Random Z'].time_series_objects
        ts_obj2 = mpp_pair_of_populations.poisson_process_dict['Random Z*'].time_series_objects
    else:
        clusters=7
        event_probability_range=[0.1,0.2]
        noise_scaling=0.001
        max_noise_prob=0.5
        pop_sizes=np.random.choice([14,15],size=clusters)  
        pop_names=np.random.choice(['Jill','Rosa','Flo','Luke','Owen','Tom','Elena'],size=clusters,replace=False)
        
        # set up allowable ranges for events,noise and lag
        ep_lower=event_probability_range[0]
        ep_higher=event_probability_range[1]
        noise_lower=min(ep_lower*noise_scaling,max_noise_prob)
        noise_higher=min(ep_higher*noise_scaling,max_noise_prob)
        
        # initialise event/noise probability and lag values
        event_probs=np.random.uniform(ep_lower,ep_higher,size=clusters)
        mean_lags=np.random.uniform(2,20,size=clusters)
        noise_probs=np.random.uniform(noise_lower,noise_higher,size=clusters)
        mpp=initialise_multiple_populations(length=length,sizes=pop_sizes,keys=pop_names,
                                            event_probs=event_probs,mean_lags=mean_lags,noise_probs=noise_probs,verbose=False)
        
        ts_obj1=mpp.randomly_mix_populations(pop_names)
        number=len(ts_obj1)        
        ts_obj2=[]
    
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

        td=pc.tweet_data(ts_matrices,params=params_dict,disjoint_sets=pair,delta=delta,axes=axes[0,:])

        start=time.time()
        #td.test_delta(max_delta=200,delta_step=5)
        td.display_Z_vals(ax=axes[0][1])
        #print(td.raw_results)
        
        if not pair:
            matching_scores=[]
            non_matching_scores=[]
            for result in td.raw_results:
                if result[0]>10000000:
                    continue
                #print(result)
                #print(ts_obj1)
                event1=result[1].event_t_series
                event2=result[2].event_t_series
                if event1==event2:   
                    
                    #print("Correlation from {0} to {1} is {2}".format(result[1].name,result[2].name,result[0]))
                    matching_scores.append(result[0])
                else:
                    non_matching_scores.append(result[0])
            print("Mean and std for {0} matching scores is {1}/{2}".format(len(matching_scores),np.mean(matching_scores),np.std(matching_scores)))
            print("Mean and std for {0} non matching scores is {1}/{2}".format(len(non_matching_scores),np.mean(non_matching_scores),np.std(non_matching_scores)))
            



        

        
        
        