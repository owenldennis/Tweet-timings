# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:54:14 2021
Functions to analyse and nicely display sensitivity of Z-score results to parameters
@author: owen
"""

from sklearn import feature_selection
import numpy as np
import pandas as pd
from IPython.display import display
#import pointwise_correlation as pc
#import testing_and_analysis_functions as analysis_func
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


pd.options.display.float_format = '{:,.2f}'.format

# columns containing relevant parameters
PARAMETER_COLUMNS_DICT={'length of time series' : True, 
                            'number of time series compared' : True,
                            'incidence mean' : False,
                            'incidence std' : False,
                            'expected noise:event ratio' : False,
                            'max lag' : True, 
                            'number of populations' : True,
                            'individuals within populations similar' : True,
                            'individuals across populations similar' : True,
                            'log ratio' : False,
                            'number_of_comparisons' : True,
                            'sqrt number_of_comparisons' : True,
                            }

EXPLANATIONS_DICT = {'length of time series' : 'length of each time series compared', 
                     'number of time series compared' : 'total number of time series in all populations combined',
                     'incidence mean' : 'mean proportion of 1s across all time series' ,
                     'incidence std' : ' standard deviation of the number of 1s across all time series',
                     'expected noise:event ratio' : 'the ratio of uncorrelated to correlated 1s within populations' ,
                     'max lag' : 'the maximum lag for the correlated 1s within populations' , 
                     'number of populations' : 'number of distinct populations',
                     'individuals within populations similar' : 
                         'if True, this means that individuals within a population all have the same expected number of 1s - this is necessary for meaningful v2 sigma calculations', 
                     'individuals across populations similar' : 'if True, each population has the same overall mean expected number of 1s',
                     'log ratio' : 'natural log of the noise ratio - used for linear regression',
                     'number_of_comparisons' : 'the number of comparisons made within populations (where correlation can be expected to be seen)',
                     'sqrt number_of_comparisons' : 'square root of the number of comparisons - used to create weighted z-scores',
                     'Z score mean (v1_sigma)' : 'The mean z-score for the number of marks within populations (using version 1 sigma)',
                     'Z score std (v1_sigma)' : 'The standard deviation of the z-score for the number of marks within populations (using version 1 sigma)', 
                     'Z score mean (v2_sigma)': 'The mean z-score for the number of marks within populations (using version 2 sigma)',
                     'Z score std (v2_sigma)' : 'The standard deviation of the z-score for the number of marks within populations (using version 1 sigma)',
                     'v1_null_mean' : 'The mean z-score using v1 sigma for the number of marks across populations, where no correlation is possible',
                     'v1_null_std' : 'Standard deviation of the v1 null z-scores', 
                     'v2_null_mean' : 'The mean z-score using v2 sigma for the number of marks across populations where no correlation is possible',
                     'v2_null_std' : 'Standard deviation of the v2 null z-scores',
                     'v1 weighted Z-score' : 'Attempt to create statistically valid z-score based on the number of comparisons and the measured null hypothesis values', 
                     'v1 p-value' : 'p-value based on the weighted z-score for v1 sigma',
                     'v2 weighted Z-score' : 'Attempt to create statistically valid z-score based on the number of comparisons and the measured null hypothesis values', 
                     'v2 p-value' : 'p-value based on the weighted z-score for v2 sigma'}

def mutual_information_analysis(df,verbose=False,all_results=False,target_results_cols = [],cutoff=0.1,display_scattergraphs=False):
     
    
    df=df.dropna()
    ### only results that mean anything for v2 sigma have individuals within a population drawn from same incidence
    df_v2=df[df['individuals within populations similar']==True]
    
    # for reference only
    columns=['length of time series', 'number of time series compared',
           'incidence mean', 'incidence std', 'expected noise:event ratio',
           'max lag', 'number of populations',
           'individuals within populations similar',
           'individuals across populations similar', 'Z score mean (v1_sigma)',
           'Z score std (v1_sigma)', 'Z score mean (v2_sigma)',
           'Z score std (v2_sigma)', 'raw data directory', 'log ratio',
           'number_of_comparisons', 'sqrt number_of_comparisons', 'v1_null_mean', 'v1_null_std', 'v2_null_mean',
           'v2_null_std', 'v1 weighted Z-score', 'v1 p-value',
           'v2 weighted Z-score', 'v2 p-value']
    
    # determine which columns containing results are to be used in MI comparisons with parameters
    # defaults to the correlation results for v1 and v2 sigmas if nothing passed in function parameters
    if all_results:
        results_columns=['Z score mean (v1_sigma)',
                     'Z score std (v1_sigma)', 
                     'Z score mean (v2_sigma)',
                     'Z score std (v2_sigma)',
                     'v1_null_mean', 'v1_null_std', 'v2_null_mean',
                     'v2_null_std', 'v1 weighted Z-score', 'v1 p-value',
                     'v2 weighted Z-score', 'v2 p-value'
                    ]
        print("Analysing MI between parameters and every results column...")
    elif len(target_results_cols):
        results_columns = target_results_cols
        print("Mutual information between parameters and column(s) {0}".format(results_columns))
    else: 
        # default results columns are the correlation results for v1 and v2 sigma
        results_columns=['Z score mean (v1_sigma)',
                         'Z score mean (v2_sigma)']
        print("Mutual information between parameters and mean z-scores for comparisons within populations using version 1/version 2 sigma")
            

    ordered_headings=np.array(list(PARAMETER_COLUMNS_DICT.keys()))
    
    
    # Mutual information analysis
    results={}
    for col in results_columns:   
        if 'v2' in col:
            y=df_v2[col].to_numpy()
            X=df_v2[ordered_headings].to_numpy()
        else:
            y=df[col].to_numpy()
            X=df[ordered_headings].to_numpy()
        bool_mask=[PARAMETER_COLUMNS_DICT[key] for key in ordered_headings]
        results[col]=feature_selection.mutual_info_regression(X, y, discrete_features=bool_mask, n_neighbors=3, copy=True, random_state=None)
    
    results_df=pd.DataFrame.from_dict(results,orient='columns')
    results_df['feature']=ordered_headings
    results_df.set_index('feature',inplace=True,drop=True)
    
    
    # sort by v1 MI results
    results_df = results_df.sort_values(by=results_columns[0],ascending=False)
    # remove very low values from display
    MI_cut_off = cutoff
    mask = results_df > MI_cut_off
    results_df = results_df.where(mask,'<{0}'.format(cutoff))
    # remove meaningless entries for v2 sigma
    v2_cols = [col for col in results_columns if 'v2' in col]
    for col in v2_cols:
        results_df.loc['individuals within populations similar',col] = 'NA'
    
    display(results_df)
    pd.set_option("display.max_colwidth", None)
    if verbose:
        print("Explanation of parameters (index of table)")
        d={}
        for feature in results_df.index:
            d[feature] = EXPLANATIONS_DICT[feature]          
        display(pd.DataFrame.from_dict(d,orient='index',columns=['Explanation']))
        d={}
        print("Explanation of results headings")
        for res in results_df.columns:
            d[res] = EXPLANATIONS_DICT[res]
        display(pd.DataFrame.from_dict(d,orient='index',columns=['Explanation']))
    pd.set_option("display.max_colwidth", 50)        
    
    # if displaying scattergraphs for parameter and result types with MI higher than cut-off, first make a list of all relevant parameter/result pairs
    if display_scattergraphs:
        # correlations - clean up first
        df_clean1 = df.drop(['raw data directory'],axis=1)
        df_clean1.drop(['individuals within populations similar', 'individuals across populations similar'],axis=1,inplace=True)
        df_corr1=df_clean1.corr()
        df_clean2 = df_v2.drop(['raw data directory'],axis=1)
        df_clean2.drop(['individuals within populations similar', 'individuals across populations similar'],axis=1,inplace=True)
        df_corr2=df_clean2.corr()
        
        # select all the parameter/result pairs that have MI above cutoff
        v1_scatter_cols=[]
        v2_scatter_cols=[]
        # these columns have only True/False entries and are not suitable for scattergraphs
        boolean_cols = ['individuals within populations similar', 'individuals across populations similar']
        for ind in results_df.index:
            for col in results_df.columns:
                if mask.loc[ind,col] and ind not in boolean_cols:
                    if 'v1' in col:
                        v1_scatter_cols.append((ind,col))
                    else:
                        v2_scatter_cols.append((ind,col))
                        
        # display all these in scattergraphs
        for version,scatter_cols in enumerate([v1_scatter_cols,v2_scatter_cols]):
            
            if 'v1' in scatter_cols:
                df_corr = df_corr1
                df_version = df
            else:
                df_corr = df_corr2
                df_version = df_v2
                
            if len(scatter_cols):

                if not len(scatter_cols)%4:
                    fig_rows = int(len(scatter_cols)/4)
                else:
                    fig_rows = int(len(scatter_cols)/4)+1

                fig_cols = min(4,len(scatter_cols))
                                    
                plt.rcParams["figure.figsize"]=[20,5*fig_rows]
                
                f,ax=plt.subplots(fig_rows,fig_cols)
                if fig_rows==1:
                    for i in range(fig_cols):
                        param=scatter_cols[i][0]
                        res=scatter_cols[i][1]
                        correlation = df_corr.loc[param,res]
                        df_version.plot.scatter(ax=ax[i],x=param,y=res,title='r = {0:.3f}'.format(correlation))
                else:
                    for i in range(fig_rows):
                        for j in range(fig_cols):
                            if 4*i+j == len(scatter_cols):
                                break
                            param=scatter_cols[4*i+j][0]
                            res=scatter_cols[4*i+j][1]
                            correlation = df_corr.loc[param,res]
                            df_version.plot.scatter(ax=ax[i][j],x=param,y=res,title='r = {0:.3f}'.format(correlation))
                print("Scattergraphs for associations between parameters and version {0:.3f} sigma results".format(version+1))
                plt.show()
                

                
                
                        
                
 
def linear_regression_analysis(df,regression_feature = 'log score', regression_on_cols = ['Z score mean (v1_sigma)','Z score mean (v2_sigma)'], only_similar_individuals = True, color_map_col = None,verbose=False):
    df=df.dropna()
    ### only results that mean anything for v2 sigma have individuals within a population drawn from same incidence
    if only_similar_individuals:
        df=df[df['individuals within populations similar']==True]
    
    df=df.replace(to_replace = {True : 1, False : 0})
     
    # check linear correlation coefficients
    ordered_headings=np.array(list(PARAMETER_COLUMNS_DICT.keys()))
    new_headings=np.append(ordered_headings,regression_on_cols)
    df = df[new_headings]
    #df_clean2 = df_v2[new_headings]
    df_corr = df.corr()
    #df_clean2 = df_clean2.corr()
    df_corr = df_corr.sort_values(by=regression_on_cols[0])

    display(df_corr[regression_on_cols])
    
    # linear regression
 
    df_train=df.tail(2990)
    df_test=df.tail(2990)
    log_mean=np.mean(df_train[regression_feature])
    scores={}
    plt.rcParams['figure.figsize']=[20,10]
    f,ax = plt.subplots(1,2)
    for i,col in enumerate(regression_on_cols):
        lin_reg=LinearRegression().fit(np.array(df_train[regression_feature]).reshape(len(df_train.index),1), \
                                       np.array(df_train[col]).reshape(len(df_train.index),1))
        score = lin_reg.score(np.array(df_test[regression_feature]).reshape(len(df_test.index),1), \
                              np.array(df_test[col]).reshape(len(df_test.index),1))

        scores['R2_v{0}'.format(i+1)] = score
        scores['Parameters_v{0}'.format(i+1)] = lin_reg.coef_
        
        
        #print(df.columns)
        df.plot.scatter(ax=ax[i],x=regression_feature,y=col,c=color_map_col, colormap='plasma')
        
        z_score_mean = np.mean(df_train[col])
        m = lin_reg.coef_[0][0]
        c = z_score_mean-log_mean*m
        xs=np.array([min(df[regression_feature]),max(df[regression_feature])])
        ys = xs*m + c
        ax[i].plot(xs, ys, color = 'r')
        ax[i].set_title("Linear regression y={1:.2f}x+{2:.2f}. R2 score {0:.2f}".format(score,m,c))
        
    plt.show()
    
    
def lin_reg_two_features(df):
    df_train=df.head(3000)
    df_test=df.tail(3000)
    lin_reg=LinearRegression().fit(list(zip(df_train['log ratio'],df_train['sqrt number_of_comparisons'])), \
                                   np.array(df_train['Z score mean (v1_sigma)']).reshape(len(df_train.index),1))
    score_v1 = lin_reg.score(list(zip(df_test['log ratio'],df_test['sqrt number_of_comparisons'])), \
                          np.array(df_test['Z score mean (v1_sigma)']).reshape(len(df_test.index),1))
    df.plot.scatter(x='sqrt number_of_comparisons',y='Z score mean (v1_sigma)')
    print("Explained variance using log (noise ratio) and sqrt(number of comparisons) to predict the mean sigma_1 z-score is {0:.3f}".format(score_v1))


# the three functions below are used for appending new info to index file
# shouldn't need to run these again
# analyses the raw data in order to:
# - create a new column showing the number of comparisons on which stored mean p-values and z-scores were based
# - create columns storing mean p-values and z-scores weighted by the number of comparisons

def calculate_no_of_comparisons(row):
    v1_df,v2_df=analysis_func.load_results_to_dfs(row['raw data directory'])
    # set up dataframes for matching populations
    v1=v1_df[v1_df['name1']==v1_df['name2']]
    v2=v2_df[v2_df['name1']==v2_df['name2']]
    
    # set up null hypothesis mean and std for each sigma measure (based on all null comparisons across populations)
    v2_across_pops=v2_df[v2_df['name1'] != v2_df['name2']]
    v2_null_mean=v2_across_pops['Z-score'].mean()
    v2_null_std=v2_across_pops['Z-score'].std()
    
    v1_across_pops=v1_df[v1_df['name1'] != v1_df['name2']]
    v1_null_mean=v1_across_pops['Z-score'].mean()
    v1_null_std=v1_across_pops['Z-score'].std()

    if np.random.random()<0.003:
        print("Re-calculated mean v1 score is {0} and stored score is {1}".format(v1['Z-score'].mean(), \
                                                                                  row['Z score mean (v1_sigma)']))
        v2_across_pops=v2_df[v2_df['name1'] != v2_df['name2']]
        v1_across_pops=v1_df[v1_df['name1'] != v1_df['name2']]
        print("v1 sigma gives mean/std across non-matching populations of {0}/{1}" \
              .format(v1_across_pops['Z-score'].mean(),v1_across_pops['Z-score'].std()))
        v1_across_pops.hist(column='Z-score',bins=100)
        v2_across_pops.hist(column='Z-score',bins=100)

    assert(len(v1.index)==len(v2.index))
    return pd.Series([len(v1.index),v1_null_mean,v1_null_std,v2_null_mean,v2_null_std])

def calculate_p_values(row):
    x1=row['Z score mean (v1_sigma)']
    mu1=row.v1_null_mean
    sig1=row.v1_null_std
    x2=row['Z score mean (v2_sigma)']
    mu2=row.v2_null_mean
    sig2=row.v2_null_std
    n=row.number_of_comparisons
    Z1 = np.sqrt(n)*(x1-mu1)/sig1
    Z2 = np.sqrt(n)*(x2-mu2)/sig2
    return pd.Series([Z1,stats.norm.cdf(Z1),Z2,stats.norm.cdf(Z2)])

def append_new_columns():
    df=pd.read_csv("{0}".format(pc.INDEX_FILE),index_col=0)
    print("Total length of dataframe {0}".format(len(df.index)))
    
    # create column for log of noise ratio
    df['log ratio']=df['expected noise:event ratio'].apply(lambda x: np.log(x))
    df[['number_of_comparisons','v1_null_mean','v1_null_std','v2_null_mean','v2_null_std']] = df.apply(lambda x: calculate_no_of_comparisons(x),axis=1)
    
    # create columns of comparable Z-scores with corresponding columns for
    # p-values for v1 and v2 mean z-scores based on the number of comparisons represented by each mean Z-score
    df[['v1 weighted Z-score','v1 p-value','v2 weighted Z-score','v2 p-value']] \
        =df.apply(lambda x: calculate_p_values(x),axis=1)
    
    df['sqrt number_of_comparisons'] = np.sqrt(df['number_of_comparisons'])
    
    df.to_csv("Index_file_test_with_extras")
    

if __name__ == '__main__':
    df=pd.read_csv("..\Results\multiple_pop_correlations_index.csv",index_col=0)   
    linear_regression_analysis(df,regression_feature = 'log ratio', only_similar_individuals = True,
                               color_map_col = 'incidence mean')