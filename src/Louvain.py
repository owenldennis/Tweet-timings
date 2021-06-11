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


class Louvain_methods():
    
    def __init__(self,df_results, version = 'Unknown', p_values_graph_setup_option = 'weights', resolution = 1, recursion_level = 0,
                 show_dfs = False, store_results_dir= "", Louvain_version = '1', verbose=False):
        """
        Parameters
        ----------
        df_results : TYPE DataFrame
            DESCRIPTION. df_results must have the following columns:
            'object1' giving the id of the first object
            'object2' giving the id of the second object
            'name1' giving the name of the population of object1
            'name2' giving the name of the population of object2
            'p_value' giving the probabilty of correlation between the two objects
        p_values_graph_setup_option : TYPE string
            DESCRIPTION options are 'random' to initalise complete graph with random edge weights;
                                    'weights' to initialise complete graph with correlation p-values as edge weights
                                    'edges' to selecte each edge (unweighted) with probability given by correlation p-value
        resolution : TYPE float, optional
            DESCRIPTION. The default is 1. Passed directly to community_Louvain methods
        show_dfs : TYPE boolean, optional
            DESCRIPTION. The default is False.
        store_results_dir : TYPE string, optional
            DESCRIPTION. The default is "". If passed, f-scores and confusion matrices are stored
        version : TYPE string, optional
            DESCRIPTION. The default is None. Options are 'v1' for sigma version 1 or 'v2' for sigma version 2
        Louvain_version : TYPE string, optional
            DESCRIPTION. The default is '1', referring to standard Louvain method maximising modularity
                         If this is set to '2', the level of the dendrogram giving the nearest number of clusters to the number of populations is selected
                         If this is set to '3', the standard Louvain method is run; then on each partition found, Louvain is run again
                         If this is set to '4', Louvain is run recursively on each partition found until there is no change in the total number of partitions
                         
        verbose : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
    
        self.graph_setup = p_values_graph_setup_option
        self.verbose = verbose
        self.Louvain_version = Louvain_version
        self.sigma_version = version
        self.show_dfs = show_dfs
        self.store_results_dir = store_results_dir
        self.resolution = resolution
        self.recursion_level = recursion_level
        
        # randomise order of results
        self.df_input_correlations = df_results.sample(frac=1)
        # create lists of the population names and the object ids (nodes)
        self.names=list(set(self.df_input_correlations['name1']).union(set(self.df_input_correlations['name2'])))
        self.nodes=list(set(self.df_input_correlations['object1']).union(set(self.df_input_correlations['object2'])))
        if verbose:
            print("There are {0} populations and a total of {1} individuals".format(len(self.names), len(self.nodes)))
         
        # create dictionary to look up the population name for any given object id (node)
        #name_of_node={self.df_input_correlations.loc[i,'object1']:self.df_input_correlations.loc[i,'name1'] for i in self.df_input_correlations.index}
        self.name_of_node = {}
        pop_sizes = collections.defaultdict(int)
        for i in self.df_input_correlations.index:
            id1 = self.df_input_correlations.loc[i,'object1']
            name1 = self.df_input_correlations.loc[i,'name1']
            id2 = self.df_input_correlations.loc[i,'object2']
            name2 = self.df_input_correlations.loc[i,'name2']   
            
            # test consistency of populations and names
            if id1 in self.name_of_node.keys():
                assert(self.name_of_node[id1] == name1)  
            else:
                pop_sizes[name1] += 1
                self.name_of_node[id1] = name1
                
            if id2 in self.name_of_node.keys():
                assert(self.name_of_node[id2] == name2)  
            else:
                pop_sizes[name2] += 1
                self.name_of_node[id2] = name2
           
        
        if self.verbose:
            print("Population sizes based on names in dataframe are {0}".format(pd.DataFrame.from_dict(pop_sizes, orient = 'index').sort_index()))
        
        self.initialise_graph()
        self.make_partition()
                  
            
    def initialise_graph(self):
        #set up graph with time series as nodes and edge weights given by correlations
        self.graph=nx.Graph()
        self.graph.add_nodes_from(self.nodes)
        if self.graph_setup == 'random':
            if self.verbose:
                print("Testing random graph")
            self.graph.add_edges_from([(self.df_input_correlations.loc[i]['object1'],self.df_input_correlations.loc[i]['object2'],
                               {'weight': np.random.random()}) 
                          for i in self.df_input_correlations.index])
       
        elif self.graph_setup == 'weights':
            self.graph.add_edges_from([(self.df_input_correlations.loc[i]['object1'],self.df_input_correlations.loc[i]['object2'],
                               {'weight': self.df_input_correlations.loc[i]['p-value']}) 
                          for i in self.df_input_correlations.index])
            if self.verbose:
                print("Passing p-values as weights for each edge")
        elif self.graph_setup == 'edges':
            self.graph.add_edges_from([(self.df_input_correlations.loc[i]['object1'],self.df_input_correlations.loc[i]['object2'])
                             for i in self.df_input_correlations.index if np.random.random()<self.df_input_correlations.loc[i]['p-value']])
            if self.verbose:
                print("Assigning each edge with probability given by corresponding correlation p-value")

    def make_partition(self):
        if self.Louvain_version == '1':
            # maximise modularity using greedy algorithm
            self.partition = community_louvain.best_partition(self.graph,weight='weight',resolution = self.resolution)
        elif self.Louvain_version == '2':
            # choose the partition with the best number of clusters
            dendro = community_louvain.generate_dendrogram(self.graph)
            # default to the lowest level (largest number of clusters)
            level = 0
            self.partition = community_louvain.partition_at_level(dendro,level)
            
            # if a better partition (closer to the correct number of populations) can be found then use it instead
            for level in range(len(dendro)-1):
                current_partition = community_louvain.partition_at_level(dendro,level)
                #print("Partition at level {0} is {1}".format(level,current_partition))
                clusters = len(set(current_partition.values()))
                if clusters<len(self.names):
                    break
                    
            if level:
                last_partition = community_louvain.partition_at_level(dendro,level-1)
                last_clusters = len(set(last_partition.values()))
                if abs(len(self.names)-clusters)<abs(last_clusters - len(self.names)):
                    self.partition = current_partition
                else:
                    self.partition = last_partition
        
        elif self.Louvain_version == '3':
            # find best partition
            self.partition = community_louvain.best_partition(self.graph,weight='weight',resolution = self.resolution)
            node_groupings = self.split_partition_into_nodes()
            recursively_analyse_subgraphs(self, to_depth = 1)
        
        elif self.Louvain_version == '4':
            # find best partition
            self.partition = community_louvain.best_partition(self.graph,weight='weight',resolution = self.resolution)
            self.node_groupings = self.split_partition_into_nodes()
            self.recursively_analyse_subgraphs(self)

                
    def split_partition_into_nodes(self):
        if self.verbose:
            print("The set of partitions is {0}".format(set(self.partition.values())))
        # create a list of each set of nodes that corresponds to a separate partitions
        node_groupings = [[node for node in self.partition.keys() if self.partition[node] == subgraph_number]
                                 for subgraph_number in set(self.partition.values())]
        if self.verbose:
            print("The partition is {0}".format(self.format_partition(self.partition)))
            print("The node-groupings are therefore {0}".format(self.format_partition(node_groupings)))
        assert(len(self.nodes) == sum([len(nodes) for nodes in node_groupings]))
        
        return node_groupings
    
    def recursively_analyse_subgraphs(self, to_depth = None):
        if self.verbose:
            print("Starting recursion level {0}".format(self.recursion_level))
        # if the graph has only one partition, or the desired recursion level has been reached,break out of the recursion
        if to_depth == self.recursion_level or len(self.node_groupings) == 1:
            if self.verbose:
                print("Returning from recursion level {0}".format(self.recursion_level))
            return None    
        
        # iterate through each group of nodes and instantiate a new Louvain_methods object for each
        new_partition = {}
        partition_number = 0
        for i,nodes in enumerate(self.node_groupings):
            df = self.df_input_correlations.loc[(self.df_input_correlations['object1'].isin(nodes))
                                                    & (self.df_input_correlations['object2'].isin(nodes))]
                
            sub_Louvain = Louvain_methods(df,p_values_graph_setup_option = self.graph_setup,resolution = self.resolution, 
                                          recursion_level = self.recursion_level + 1, version = self.sigma_version, 
                                          Louvain_version = self.Louvain_version, verbose = self.verbose)
            if self.verbose:
                print("Sub-partition {1} found : {0}".format(self.format_partition(sub_Louvain.partition), i))
                
            # update new partition
            for individual in sub_Louvain.partition.keys():
                new_partition[individual] = sub_Louvain.partition[individual] + partition_number
                
                
            partition_number += len(set(sub_Louvain.partition.values()))
                
            if self.verbose:
                print("New partition now looks like: {0}".format(self.format_partition(new_partition)))
                
        self.partition = new_partition
                
        
    def format_partition(self, partition):
        if type(partition) == dict:
            return {str(self.name_of_node[id_code]) + "_" + str(id_code)[-4:] : partition[id_code] for id_code in partition.keys()}
        if type(partition) == list:
            return [[str(self.name_of_node[id_code]) + "_" + str(id_code)[-4:] for id_code in node_group] for node_group in partition]
        
            
        
            
                
    #def make_sub_graph(self, nodes = []):
    #    if not len(nodes):
    #        nodes = self.nodes
        
                
    def score_partition(self): 
        
        clusters=len(set(self.partition.values()))
        if self.verbose:
            print("Divided into {0} clusters".format(clusters))
        # analyse which cluster contains which members of each population
        cross_reference_dict=collections.defaultdict(lambda: collections.defaultdict(int))
        for node_id in self.partition.keys():
            cross_reference_dict["cluster {0}".format(self.partition[node_id])][self.name_of_node[node_id]]+=1
        
        cross_ref_df = pd.DataFrame.from_dict(cross_reference_dict,orient = 'index')
        if self.show_dfs:
            print(cross_ref_df)
        if len(self.store_results_dir):
            cross_ref_df.to_csv("{0}/{1}confusion_matrix.csv".format(self.store_results_dir,self.sigma_version))
            
            
            
        
        # make f-score based on true positive edges between ids in the same population
        TP=collections.defaultdict(int)
        tp=0
        FP=collections.defaultdict(int)
        fp=0
        FN=collections.defaultdict(int)
        fn=0
        for index in self.df_input_correlations.index:
            id1=self.df_input_correlations.loc[index]['object1']
            pop1=self.df_input_correlations.loc[index]['name1']
            id2=self.df_input_correlations.loc[index]['object2']
            pop2=self.df_input_correlations.loc[index]['name2']
            if pop1==pop2:
                if self.partition[id1]==self.partition[id2]:
                    TP[pop1]+=1 # true positive
                    tp+=1
                else:
                    FN[pop1]+=1 # false negative
                    fn+=1
            elif self.partition[id1]==self.partition[id2]:
                    FP[pop1]+=1 # false positive - not interested in true negatives
                    FP[pop2]+=1 # NB false positive edges double counted, once for each node (population object)
                    fp+=1
        
        FP_overall=sum([FP[name] for name in self.names])/2 # halved as double counted within populations
        TP_overall=sum([TP[name] for name in self.names])
        FN_overall=sum([FN[name] for name in self.names])
        assert FP_overall==fp
        assert TP_overall==tp
        assert FN_overall==fn
        
        recalls = {name :TP[name]/(TP[name]+FP[name]) if (TP[name] + FP[name]) else 0 for name in self.names}
        #print(recalls)
        precisions = {name : TP[name]/(TP[name]+FN[name]) if (TP[name] + FN[name]) else 0 for name in self.names}
        f_scores={name : 2*recalls[name]*precisions[name]/(recalls[name]+precisions[name]) if (recalls[name]+precisions[name]) else 0 for name in self.names}
        scores_df = pd.DataFrame([recalls,precisions,f_scores],index=['Recall','Precison','F-score'],columns=[name for name in self.names])
        R=TP_overall/(FP_overall+TP_overall)
        P=TP_overall/(TP_overall+FN_overall)
        f_score=2*R*P/(R+P)            
        scores_df['Overall'] = [R,P,f_score]
        if self.show_dfs:
            print(scores_df)
        if self.store_results_dir:
            scores_df.to_csv("{0}/{1}recall_precision_fscores.csv".format(self.store_results_dir,self.sigma_version))
    
        return {'{0}clusters'.format(self.sigma_version) :[clusters],'{0}recall'.format(self.sigma_version) :[R],'{0}precision'.format(self.sigma_version) : [P],'{0}f_score'.format(self.sigma_version) : [f_score]}




"""
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
"""
