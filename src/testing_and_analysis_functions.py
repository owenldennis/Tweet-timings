# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 12:30:22 2021
Functions for initialising multiple correlated populations (create_fixed_metaparams,create_random_metaparams,
directly_initialise_multiple_populations)
Methods will be added for testing and analysing pointwise correlations using lagging poisson processes. 
Also contains method for writing results from temp directory to main results directory (copy_from_temp)

@author: owen
"""
import poisson_processes as pp
import pointwise_correlation as pc
import numpy as np
import pandas as pd
import os
import shutil
import pointwise_correlation as pc
import time
import matplotlib.pyplot as plt
import Louvain

def create_population_names(number_of_populations):
    #Allocate a name for each population (cluster)
    clusters=number_of_populations
    if clusters < 7:
        pop_names=np.random.choice(['Jill','Rosa','Flo','Luke','Owen','Tom','Elena'],size=clusters,replace=False)
    else:
        pop_names=np.random.choice(list(range(clusters)),size=clusters,replace=False)  
        pop_names=[str(p) for p in pop_names]
    return pop_names
    
def create_fixed_metaparams(length,number_of_populations,size,lag,event_prob,noise_prob):
    pop_names=create_population_names(number_of_populations)
    metaparams={name: {'T': length,
                       'n':size,
                    'event probability':event_prob,
                    'max lag':lag,
                    'mean noise probability':noise_prob
                   }
             for name in pop_names}
    return metaparams
    
def create_random_metaparams(length,number_of_populations,max_size,max_lag,max_event_prob,max_noise_prob):
    if max_size < 2:
        max_size=2
    if max_event_prob<10/length:
        max_event_prob=10/length
    if max_noise_prob<10/length:
        max_noise_prob=10/length
       
    pop_names=create_population_names(number_of_populations)
        
    metadata={name:{'T': length,
                    'n':np.random.choice(range(2,max_size)),
                    'event probability':np.random.random()*max_event_prob,
                    'max lag':max_lag,
                    'mean noise probability':np.random.random()*max_noise_prob
                   }
             for name in pop_names}
    return metadata

def directly_initialise_multiple_populations(length,metaparams,use_fixed_means=False,verbose=False):
    """
    format for meta_params:
        name_key: {'n':size of population,
                   'event probability': probability of a 1 in event time series,
                   'max lag': max lag for this population,
                   'mean noise probability' : median probability of a 1 in noise time series across population}
                    
    """
    population_params={}
    for key in metaparams:
        number = metaparams[key]['n'] # size of population with this key
        p_event=metaparams[key]['event probability'] # underlying event frequency for correlated events within members of population
        event_beta=1/p_event 
        max_lag=metaparams[key]['max lag']    # max lag from event process for members of population
        p_noise=metaparams[key]['mean noise probability']   # mean proportion of noise for members of population 
        
       
        noise_beta=1/p_noise 
        if use_fixed_means:                
            betas=[noise_beta]    # if use_fixed_means is true all members of the population have the same beta for noise
        else:
            betas=np.random.uniform(2,noise_beta*2,size=number)         # else each member of the population has a different noise beta
        
        # set up parameters for random noise poisson processes for this population
        population_params[key+'_noise']={'n' : number,
                                    'betas' : betas
                                }
        
        # set up parameters for event-based lagging poisson processes for this population
        population_params[key]={'n' : number,
                                'betas' : [],
                                'prior poisson process' : pp.poisson_process(1,length,[event_beta]),   # single poisson process (event process) 
                                'lag parameters' : {'mus' : np.random.randint(1,max_lag,size=number)},    # list of mean lags for members 
                                                    #np.random.poisson(lam=mean_lag,size=number)},
                                'combine with' : key+'_noise'          # previously initialised noise to add in
                                }

    return population_params



def compare_inferred_and_known_means(xs,number,ts_matrices,reclustering=None,verbose=False,show_distributions=False):
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
    start_time = time.time()
    
    # start run - for each length of time series (in array xs) correlations are measured for both sigma values
    for T in xs:

        ts.append(T)        
        print("T is {0}".format(T))

        # set up axes to display results
        if show_distributions:
            f,axes = plt.subplots(2,2)
            for ax in axes[1]:
                ax.plot([-2,2],[0,0],'.',alpha=0.2)
                first_run_axes=[axes[0][0],axes[1][0]]
                second_run_axes=[axes[0][0],axes[1][1]]
      
        else:
            first_run_axes=[]
            second_run_axes=[]
        
        # set parameters and instantiate classes
        params_dict = {}
        params_dict['sparse']=True
        disjoint=False
        # use maximum delta value
        delta = int(np.sqrt(T)) 
        params_dict['T'] = T
        params_dict['n'] = number # in case multiple populations have been combined

        td = pc.tweet_data(ts_matrices,params_dict,delta = delta,
                        disjoint_sets=disjoint,verbose=verbose,axes = first_run_axes)
 
        
        print("Running with inferred means (version 1).  Time elapsed: {0}".format(time.time()-start_time))        
        td.params['Use population means']=False
        if td.params['Use population means']:
            version='v2_sigma'
        else:
            version='v1_sigma'
        td.display_Z_vals()
        # measure the spread of z-values across the correlations
        ys.append(np.std(td.results))
        zs.append(np.mean(td.results))
        
        # pass the results to the Louvain algorithm.  Reclustering will be attempted depending on the parameter
        corr_overview_df=Louvain.analyse_raw_results_for_scoring(td,reclustering=reclustering)

        corr_overview_df=corr_overview_df.rename({'mean' : 'Z score mean ({0})'.format(version),
                                                  'std' : 'Z score std ({0})'.format(version)},axis=1)

        
        td.params['Use population means']=True
        if td.params['Use population means']:
            version='v2_sigma'
        else:
            version='v1_sigma'
        td.axes=second_run_axes
        print("Running with population means (version 2). Time elapsed: {0}".format(time.time()-start_time))        
        td.display_Z_vals()
        y1s.append(np.std(td.results))
        z1s.append(np.mean(td.results))
        corr_df=Louvain.analyse_raw_results_for_scoring(td,reclustering=reclustering)
        #corr_df.index=[version]
        corr_df=corr_df.rename({'mean' : 'Z score mean ({0})'.format(version),
                                                  'std' : 'Z score std ({0})'.format(version)},axis=1)
        corr_df[corr_overview_df.columns]=corr_overview_df
        #pd.DataFrame(np.transpose([ts,zs,ys,z1s,y1s])).to_csv("{0}/Accumulating results.csv".format(TEMP_DIR))
    
        if show_distributions:
            axes[0][1].errorbar(ts,zs,ys,color='r',label='v1')
            axes[0][1].errorbar(ts,z1s,y1s,color='b',label='v2')
            axes[0][1].set_title("Errorbars for z-scores")
            axes[0][1].legend()
        
            axes[1][0].legend()
            axes[1][1].legend()
            f.suptitle("Comparison of version 1 and version 2 sigmas for T = {0},delta = {1}".format(T,delta),fontsize=16)
        
            plt.show()
    df=pd.concat([pd.DataFrame(zs,columns=['Z score mean (v1_sigma)']),
                   pd.DataFrame(ys,columns=['Z score std (v1_sigma)']),
                   pd.DataFrame(z1s,columns=['Z score mean (v2_sigma)']),
                   pd.DataFrame(y1s,columns=['Z score std (v2_sigma)'])]
                 ,axis=1)
    
    df.index=['All correlations']
    df=df.append(corr_df)
    
    #df.to_csv("{0}/Sigma_comparison{1}.csv".format(TEMP_DIR,str(xs)[:10]),mode='a')
    return td,df



# copy files from temp directory to Results/dest_root_dir/T_<length>/Run_<i> directory
def copy_from_temp(dest_root_dir):
    src_dir=pc.TEMP_DIR
    # read the time series length from the meta-parameters for this run
    df=pd.read_csv("{0}/meta_params.csv".format(src_dir),index_col=0)
    T=int(df.loc[df.index[0],'T'])
    #target_root_dir="{0}/Multiple_population_correlations".format(pc.RESULTS_DIR)
    
    # try and create the root directory Results/dest_root_dir/T_<length> if not already created
    try:
        os.mkdir("{0}/T_{1}".format(dest_root_dir,str(T)))
        
    except FileExistsError:
        pass
    
    # create a new subdirectory with index time_stamp (representing the run time)
    import datetime
    error=True
    while error:
        error=False
        try:
            x = datetime.datetime.now()
            time_stamp="{0}_{1}_{2}_{3}_{4}_{5}".format(x.year,x.month,x.day,x.hour,x.minute,x.second)
            os.mkdir("{0}/T_{1}/{2}".format(dest_root_dir,T,str(time_stamp)))
            dest_dir="{0}/T_{1}/{2}".format(dest_root_dir,T,str(time_stamp))
        except FileExistsError:
            error=True
    #print(dest_dir)
    #src_dir="C:/Users/owen/Machine learning projects/Luc_tweet project/Results/Temp"
    src_files = os.listdir(src_dir)
    #print(src_files)
    for file_name in src_files:
        full_file_name = os.path.join(src_dir, file_name)
        dest_file=os.path.join(dest_dir,file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest_file)  
            
    return dest_dir

## loads results from given directory and files to separate dataframes.  These are returned formatted ready to be analysed in Louvain method
def load_results_to_dfs(directory,v1_files=['sigma_v1_correlations.csv'],
                      v2_files=['sigma_v2_correlations.csv']):

    v1_df=pd.concat([pd.read_csv("{0}/{1}".format(directory,v1_file),index_col=0) for v1_file in v1_files]).sort_index()
    v2_df=pd.concat([pd.read_csv("{0}/{1}".format(directory,v2_file),index_col=0) for v2_file in v2_files]).sort_index()

    return v1_df.rename(columns={'id1':'object1','id2':'object2'}),v2_df.rename(columns={'id1':'object1','id2':'object2'})



                      
