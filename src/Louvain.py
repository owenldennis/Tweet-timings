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

def make_partition_and_score(df_results,test_random_graph=False,pass_weights=True,verbose=False):
    
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
    name_of_node={df_results.loc[i,'object1']:df_results.loc[i,'name1'] for i in df_results.index}
    for i in df_results.index:
        name_of_node[df_results.loc[i,'object2']]=df_results.loc[i,'name2']
        
    #set up graph with time series as nodes and edge weights given by correlations
    G=nx.Graph()
    G.add_nodes_from(list(set(df_results['object1']).union(set(df_results['object2']))))
    if test_random_graph:
        print("Testing random graph")
        G.add_edges_from([(df_results.loc[i]['object1'],df_results.loc[i]['object2']) 
                      for i in df_results.index if np.random.random()<0.5])
   
    elif pass_weights:
        G.add_edges_from([(df_results.loc[i]['object1'],df_results.loc[i]['object2'],{'weight': df_results.loc[i]['p-value']}) 
                      for i in df_results.index])
        print("Passing p-values as weights for each edge")
    else:
        G.add_edges_from([(df_results.loc[i]['object1'],df_results.loc[i]['object2'])
                         for i in df_results.index if np.random.random()<df_results.loc[i]['p-value']])
        print("Assigning each edge with probability given by its p-value")

    # compute the best partition
    partition = community_louvain.best_partition(G,weight='weight',resolution = 1)
    clusters=len(set(partition.values()))
    print("Divided into {0} clusters".format(clusters))
    # analyse which cluster contains which members of each population
    cross_reference_dict=collections.defaultdict(lambda: collections.defaultdict(int))
    for node_id in partition.keys():
        cross_reference_dict[partition[node_id]][name_of_node[node_id]]+=1
    
    print(pd.DataFrame.from_dict(cross_reference_dict,orient='index'))
        
        
    
    # make f-score based on true positive egdes between ids in the same population
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
    
    recalls = {name :TP[name]/(TP[name]+FP[name]) for name in names}
    #print(recalls)
    precisions = {name : TP[name]/(TP[name]+FN[name]) for name in names}
    f_scores={name : 2*recalls[name]*precisions[name]/(recalls[name]+precisions[name]) for name in names}
    print(pd.DataFrame([recalls,precisions,f_scores],index=['Recall','Precison','F-score'],columns=[name for name in names]))
    R=TP_overall/(FP_overall+TP_overall)
    P=TP_overall/(TP_overall+FN_overall)
    f_score=2*R*P/(R+P)            
    return {'clusters':clusters,'recall':R,'precision':P,'f_score':f_score}

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
    
    return pd.DataFrame.from_dict({"Within population correlation mean": [np.mean(z_within)],
                        "Within population correlation std": [np.std(z_within)],
                        "Across populations correlation mean" : [np.mean(z_across)],
                        "Across populations correlation std" : [np.std(z_across)]}
                        )

