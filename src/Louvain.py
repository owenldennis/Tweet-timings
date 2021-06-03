# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 15:37:44 2020
Louvain community detection method
Scoring method to see how well the correlations plus Louvain re-create the original populations
@author: owen
"""

import pointwise_correlation as pc

import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import pandas as pd
import numpy as np
import collections
import scipy.stats as stats
pd.set_option("display.precision", 3)
pd.set_option("display.max_rows", 25)
pd.set_option("display.expand_frame_repr", False)


class dodgy_indices_analysis():
    
    def __init__(self,dodgy_indices=[]):
        self.verbose = True
        self.index_df=pd.read_csv("{0}".format(pc.INDEX_FILE),index_col=0)
        if not len(dodgy_indices):
            self.dodgy_indices = [527, 545, 707, 1136, 1308, 1649, 1701, 1803, 1884, 1994, 2493, 2535]
        self.columns = ['index','pop_id','expected_pop_size','actual_pop_size',
                        'expected_comparisons','actual_comparisons',
                        'event_beta', 'length of time series']
        self.results = pd.DataFrame(columns = self.columns)
        self.next_index = 0
        self.iterate_over_dodgy_indices()
        
        
    def analyse_inconsistent_metaparams(self,v1_df,v2_df,params,meta_params):
        # measure number of comparisons within each population from raw data
        # determine which population is missing comparisons
        # look for unusual parameter values for this population
        
        v1_matching = v1_df.loc[v1_df['name1'] == v1_df['name2']]
        
        #print(v1_matching.head(10))
        # iterate through each population in the metaparams index
        for pop in meta_params.index[:-2]:
            n = meta_params.loc[pop,'n']
            expected_comparisons = int(n*(n-1)/2)
            # select only this population from the matching dfs
            # make sure type of population name is going to work in dataframe
            try:
                pop_name = int(pop)
            except ValueError:
                pop_name = pop
            
            pop_comparisons = len(v1_matching.loc[v1_matching['name1'] == pop_name].index)
            if pop_comparisons:
                actual_pop_size = 2
                while actual_pop_size*(actual_pop_size-1)/2 < pop_comparisons:
                    actual_pop_size+=1
            else:
                actual_pop_size = 1
            
            beta = params.loc[pop,'prior process beta']
            length = meta_params.loc[pop,'T']
            
            self.results.loc[self.next_index,self.columns] = [self.current_index,pop,n,actual_pop_size,
                                                                 expected_comparisons,pop_comparisons,
                                                                 beta,length]      
            self.next_index += 1
            if not pop_comparisons == expected_comparisons:
                if self.verbose:
                    print("Expected {0} comparisons from population size {1}".format(expected_comparisons,n))
                    print("Found {0} comparisons for population id {1}".format(pop_comparisons,pop))
                    print(meta_params.loc[pop, 'T'])
                    print(params.loc[pop, 'prior process beta'])
                    print(params.loc[pop+"_noise" , 'betas'])
            
    
    def check_consistency_of_population_sizes(self,raw_data_dir):
        # load in all relevant data in the directory
        meta_params = pd.read_csv("{0}/meta_params.csv".format(raw_data_dir), index_col = 0)
        params = pd.read_csv("{0}/population_parameters.csv".format(raw_data_dir), index_col = 0)
        v1_df = pd.read_csv("{0}/{1}".format(raw_data_dir,'sigma_v1_correlations.csv'),index_col = 0)
        v2_df = pd.read_csv("{0}/{1}".format(raw_data_dir,'sigma_v2_correlations.csv'),index_col = 0)
        
        #print(pd.DataFrame(meta_params))
        pop_sizes = meta_params['n'].dropna()
        #print(meta_params['T'])
        #print(pop_sizes)
        k=sum(pop_sizes)
        #k = sum([pop_sizes.loc[i] for i in pop_sizes.index])
        expected_comparisons = int((k)*(k-1)/2)
        assert len(v1_df.index) == len(v2_df.index)
        self.analyse_inconsistent_metaparams(v1_df,v2_df,params,meta_params)
        
        if not len(v1_df.index) == expected_comparisons:
            return False
        
        return True      
            
    def iterate_over_dodgy_indices(self):
        for self.current_index in self.dodgy_indices:
            data_dir = self.index_df.loc[self.current_index,'raw data directory']
            if self.verbose and not self.check_consistency_of_population_sizes(data_dir):
                print("Inconsistency in index {0} - details above".format(self.current_index))


if __name__ == '__main__':
    dodgy_indices_analysis()


def make_partition_and_score(df_results,test_random_graph=False,pass_weights=True, resolution = 1, show_dfs = False, store_results_dir= "", version = None, verbose=False):
    
    """
    df_results must have the following columns:
        'object1' giving the id of the first object
        'object2' giving the id of the second object
        'name1' giving the name of the population of object1
        'name2' giving the name of the population of object2
    
    """
    # randomise order of results
    df_results=df_results.sample(frac=1)
    # create list of the population names, nodes and dictionary that links them
    names=list(set(df_results['name1']).union(set(df_results['name2'])))
    nodes=list(set(df_results['object1']).union(set(df_results['object2'])))
    if verbose:
        print("There are {0} populations and a total of {1} individuals".format(len(names), len(nodes)))
     
    #name_of_node={df_results.loc[i,'object1']:df_results.loc[i,'name1'] for i in df_results.index}
    name_of_node = {}
    pop_sizes = collections.defaultdict(int)
    for i in df_results.index:
        id1 = df_results.loc[i,'object1']
        name1 = df_results.loc[i,'name1']
        id2 = df_results.loc[i,'object2']
        name2 = df_results.loc[i,'name2']   
        
        # test consistency of populations and names
        if id1 in name_of_node.keys():
            assert(name_of_node[id1] == name1)  
        else:
            pop_sizes[name1] += 1
            name_of_node[id1] = name1
            
        if id2 in name_of_node.keys():
            assert(name_of_node[id2] == name2)  
        else:
            pop_sizes[name2] += 1
            name_of_node[id2] = name2
       
    
    if verbose:
        print(pd.DataFrame.from_dict(pop_sizes, orient = 'index').sort_index())
              
        
    #set up graph with time series as nodes and edge weights given by correlations
    G=nx.Graph()
    G.add_nodes_from(nodes)
    if test_random_graph:
        if verbose:
            print("Testing random graph")
        G.add_edges_from([(df_results.loc[i]['object1'],df_results.loc[i]['object2']) 
                      for i in df_results.index if np.random.random()<0.5])
   
    elif pass_weights:
        G.add_edges_from([(df_results.loc[i]['object1'],df_results.loc[i]['object2'],{'weight': df_results.loc[i]['p-value']}) 
                      for i in df_results.index])
        if verbose:
            print("Passing p-values as weights for each edge")
    else:
        G.add_edges_from([(df_results.loc[i]['object1'],df_results.loc[i]['object2'])
                         for i in df_results.index if np.random.random()<df_results.loc[i]['p-value']])
        if verbose:
            print("Assigning each edge with probability given by its p-value")

    # compute the best partition
    partition = community_louvain.best_partition(G,weight='weight',resolution = resolution)
    clusters=len(set(partition.values()))
    if verbose:
        print("Divided into {0} clusters".format(clusters))
    # analyse which cluster contains which members of each population
    cross_reference_dict=collections.defaultdict(lambda: collections.defaultdict(int))
    for node_id in partition.keys():
        cross_reference_dict["cluster {0}".format(partition[node_id])][name_of_node[node_id]]+=1
    
    cross_ref_df = pd.DataFrame.from_dict(cross_reference_dict,orient = 'index')
    if show_dfs:
        display(cross_ref_df)
    if len(store_results_dir):
        cross_ref_df.to_csv("{0}/{1}confusion_matrix.csv".format(store_results_dir,version))
        
        
        
    
    # make f-score based on true positive edges between ids in the same population
    TP=collections.defaultdict(int)
    tp=0
    FP=collections.defaultdict(int)
    fp=0
    FN=collections.defaultdict(int)
    fn=0
    for index in df_results.index:
        id1=df_results.loc[index]['object1']
        pop1=df_results.loc[index]['name1']
        id2=df_results.loc[index]['object2']
        pop2=df_results.loc[index]['name2']
        if pop1==pop2:
            if partition[id1]==partition[id2]:
                TP[pop1]+=1 # true positive
                tp+=1
            else:
                FN[pop1]+=1 # false negative
                fn+=1
        elif partition[id1]==partition[id2]:
                FP[pop1]+=1 # false positive - not interested in true negatives
                FP[pop2]+=1 # NB false positive edges double counted, once for each node (population object)
                fp+=1
    
    FP_overall=sum([FP[name] for name in names])/2 # halved as double counted within populations
    TP_overall=sum([TP[name] for name in names])
    FN_overall=sum([FN[name] for name in names])
    assert FP_overall==fp
    assert TP_overall==tp
    assert FN_overall==fn
    
    recalls = {name :TP[name]/(TP[name]+FP[name]) if (TP[name] + FP[name]) else 0 for name in names}
    #print(recalls)
    precisions = {name : TP[name]/(TP[name]+FN[name]) if (TP[name] + FN[name]) else 0 for name in names}
    f_scores={name : 2*recalls[name]*precisions[name]/(recalls[name]+precisions[name]) if (recalls[name]+precisions[name]) else 0 for name in names}
    scores_df = pd.DataFrame([recalls,precisions,f_scores],index=['Recall','Precison','F-score'],columns=[name for name in names])
    R=TP_overall/(FP_overall+TP_overall)
    P=TP_overall/(TP_overall+FN_overall)
    f_score=2*R*P/(R+P)            
    scores_df['Overall'] = [R,P,f_score]
    if show_dfs:
        display(scores_df)
    if store_results_dir:
        scores_df.to_csv("{0}/{1}recall_precision_fscores.csv".format(store_results_dir,version))

    return {'{0}clusters'.format(version) :[clusters],'{0}recall'.format(version) :[R],'{0}precision'.format(version) : [P],'{0}f_score'.format(version) : [f_score]}





def analyse_raw_results_for_scoring(td_object,reclustering=None,test_random_graph=False,
                                    repeats=1,pass_weights=True,verbose=False):
    raw_results=td_object.raw_results#[:400]
    if td_object.params.get("Use population means"):
        sigma_version='sigma_v2_'
    else:
        sigma_version='sigma_v1_'
    
    
    
    df_all = pd.DataFrame(raw_results,columns=['Z-score','object1','object2'])    
    # remove infinite/nan values
    df_all = df_all.replace(np.inf,np.nan)
    df_all=df_all.dropna(inplace=False)    
    # find p-values for each z-score
    df_all['p-value']=df_all['Z-score'].map(lambda x: stats.norm.cdf(x))
    
    # Extract the population names, object ids and time series for each time series object
    df_all['name1']=[d.name for d in df_all['object1']]
    df_all['name2']=[d.name for d in df_all['object2']]
    df_all['time series 1']=[d.t_series for d in df_all['object1']]
    df_all['time series 2']=[d.t_series for d in df_all['object2']]
    df_all['id1']=[id(d) for d in df_all['object1']]
    df_all['id2']=[id(d) for d in df_all['object2']]
    names=np.sort(list(set(df_all['name1']).union(set(df_all['name2']))))
    
    df_to_store=df_all.loc[:,['p-value','Z-score','name1','name2','id1','id2']]
            
    # create a group for each population
    df_matching=df_to_store.loc[df_to_store['name1'] == df_to_store['name2']]
    grouped_matching=df_matching.groupby(['name1'],as_index=False)
    df_new=pd.DataFrame()
    for group,frame in grouped_matching:
        #print(frame)
        df_new[group]=frame.mean()
    
    # store non-matching population results in a separate df
    df_non_matching=df_to_store.drop(labels=df_matching.index)   
    names_dict={name:i for i,name in enumerate(names)} # arbitrary order for names
    swap_dict=collections.defaultdict(int)
    for i in df_non_matching.index:
        name1=df_non_matching.loc[i]['name1']
        name2=df_non_matching.loc[i]['name2']
        if names_dict[name1]>names_dict[name2]:
            #swap names so that, for any pair, the order of names is always the same
            swap_dict[name1]+=1
            df_non_matching.at[i,'name1']=name2
            df_non_matching.at[i,'name2']=name1
    # group by both names
    group_non_matching=df_non_matching.groupby(['name1','name2'],as_index=False)
    if verbose:
        assert(len(group_non_matching)==len(names)*(len(names)-1)/2)
    df_non_matching_mean=pd.DataFrame()
    for group,frame in group_non_matching:
        df_non_matching_mean[group]=frame.mean()
    
    
    if verbose:
        print("Z score and corresponding p-value mean results within each population")
        print(df_new)
        print("Z score and corresponding p-value mean results across populations")
        print(df_non_matching_mean.head(2))
    
    z_across=df_non_matching['Z-score']
    z_within=df_matching['Z-score']
    if verbose:
        print("Mean non-matching stats is {0}".format([np.mean(z_across),np.std(z_across)]))
        print("Overall mean Z-score is {0}".format(np.mean(df_all['Z-score'])))
        print("Mean matching stats is {0}".format([np.mean(z_within),np.std(z_within)]))
    
    
    # store dataframes of results in csv files
    
    #df_non_matching.to_csv("{0}\{1}correlations_across_populations.csv".format(pc.TEMP_DIR,sigma_version))
    df_to_store.to_csv("{0}\{1}correlations.csv".format(pc.TEMP_DIR,sigma_version))
    
    if reclustering=='greedy Louvain':
        scores=[]
        j=0
        for i in range(repeats):
            if repeats%(int(repeats/100+1)):
                print("{0}%".format(j),end=',')
            j+=1
            print("\n")
            scores.append(make_partition_and_score(df_all,test_random_graph=test_random_graph,pass_weights=pass_weights,verbose=verbose))
        if repeats:
            print("Over {0} runs of graph and partition, scores are {1}".format(repeats,scores))
    
    return pd.DataFrame.from_dict({"matching": {'mean':np.mean(z_within),
                                                  "std": np.std(z_within)},
                                  "not matching" : {'mean': np.mean(z_across),
                                                       "std" : np.std(z_across)}
                                   },orient='index')

