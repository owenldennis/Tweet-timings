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
    G=nx.Graph()
    names=list(set(df_results['name1']))
    # randomise order of results
    df_results=df_results.sample(frac=1)
    #print(df_results.loc[:20,['p-value','Z-score','name1','name2']])
    #for result in results:
    #    edge_weight=result['p-value']
    #    node1=result['node1']
    #    node2=result['node2']
    G.add_nodes_from(list(set(df_results['object1'])))
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
    # if np.random.random()<df_results.loc[i]['p-value']])
    #pos = nx.spring_layout(G)
    #nx.draw_networkx_nodes(G,pos)
    #nx.draw_networkx_edges(G,pos)
    # compute the best partition
    partition = community_louvain.best_partition(G,weight='weight',resolution = 1)
    clusters=len(set(partition.values()))
    print("Divided into {0} clusters".format(clusters))
    TP=collections.defaultdict(int)
    tp=0
    FP=collections.defaultdict(int)
    fp=0
    FN=collections.defaultdict(int)
    fn=0
    for index in df_results.index:
        obj1=df_results.loc[index]['object1']
        pop1=obj1.event_t_series
        obj2=df_results.loc[index]['object2']
        pop2=obj2.event_t_series
        if pop1==pop2:
            if partition[obj1]==partition[obj2]:
                TP[obj1.name]+=1 # true positive
                tp+=1
            else:
                FN[obj1.name]+=1 # false negative
                fn+=1
        elif partition[obj1]==partition[obj2]:
                FP[obj1.name]+=1 # false positive - not interested in true negatives
                FP[obj2.name]+=1 # NB false positive edges double counted, once for each node (population object)
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
    return R,P,f_score

def analyse_raw_results_for_scoring(td_object,reclustering=None,test_random_graph=False,repeats=1,pass_weights=True,verbose=False):
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
    
    df=df_all.loc[:,['p-value','Z-score','name1','name2','id1','id2']]#,'time series 1','time series 2']]
#    print(df.head(10))
    #grouped=df.groupby(['name1','name2'],as_index=False)
    #print([(name,grouped.get_group((name,name))['Z-score'].mean()) for name in set(df['name1'])])
    #grouped_again.get_group('Luke')
    #print(grouped["Z-score"].sum())
        
    
    # create a group for each population
    df_matching=df.loc[df['name1'] == df['name2']]
    grouped_matching=df_matching.groupby(['name1'],as_index=False)
    df_new=pd.DataFrame()
    for group,frame in grouped_matching:
        print(frame)
        #df_new.loc['p-value',group]=frame['p-value'].mean()
        #df_new.loc['Z-score',group]=frame['Z-score'].mean()
        df_new[group]=frame.mean()
    
    # store non-matching population results in a separate df
    df_non_matching=df.drop(labels=df_matching.index)   
    names_dict={name:i for i,name in enumerate(names)} # random order for names
    #print(names_dict)
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
        
        #print(group)
        #df_non_matching_mean.loc['p-value',group]=frame['p-value'].mean()
        #df_non_matching_mean.loc['Z-score',group]=frame['Z-score'].mean()
        df_non_matching_mean[group]=frame.mean()
    
    
    print("Z score and corresponding p-value mean results within each population")
    print(df_new)
    print("Z score and corresponding p-value mean results across populations")
    print(df_non_matching_mean.head(2))
    
    print("Mean non-matching Z-score is {0}".format(np.mean(df_non_matching['Z-score'])))
    print("Overall mean Z-score is {0}".format(np.mean(df_all['Z-score'])))
    
    
    # store dataframes of matching and non-matching populations in csv files
    
    df_non_matching.to_csv("{0}\{1}correlations_across_populations.csv".format(pc.TEMP_DIR,sigma_version))
    df_matching.to_csv("{0}\{1}correlations_within_populations.csv".format(pc.TEMP_DIR,sigma_version))
    
    if reclustering=='greedy Louvain':
        scores=[]
        j=0
        #df_all=df_all.sample(frac=1)
        for i in range(repeats):
            if repeats%(int(repeats/100+1)):
                print("{0}%".format(j),end=',')
            j+=1
            print("\n")
            scores.append(make_partition_and_score(df_all,test_random_graph=test_random_graph,pass_weights=pass_weights,verbose=verbose))
        if repeats:
            print("Over {0} runs of graph and partition, mean/std score is {1}/{2}".format(repeats,np.mean(scores,axis=0),np.std(scores,axis=0)))
    


