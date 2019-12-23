# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 08:17:16 2019

@author: owen
"""


import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import pointwise_correlation as pc
import time

def test_pop():
    ts = np.array([1,0,0,1,1,0])
    ts1 = np.array([1,1,0,0,0,1])
    a = ts[1:]
    b=ts1[:-1]
    print(a)
    print(b)

def test_prob():
    t=[]
    for i in range(10000):
        #t.append(np.random.chisquare(6)/30)
        t.append(np.random.uniform())
    print(np.mean(t))
    print(min(t))
    print(max(t))
    print(np.std(t))
    plt.hist(t,bins=100)
    print(len(np.where(np.array(t)>1.)[0]))


fibs = {1:1,2:1}
def fib(n):
    if n in fibs.keys():
        return fibs[n]
    if n-1 in fibs.keys():
        fibs[n] = fibs[n-1] + fibs[n-2]
    return fib(n-1)+fib(n-2)

def do_fib():
    fib(1000)
    for i in range(2,100):
        print("Ratio of next fibonacci numbers {0} and {1} is {2} ".format(fibs[i],fibs[i-1],fibs[i]/fibs[i-1]))

def seeds_test():
    np.random.seed(2)
    seeds = np.random.choice(100,100)

def test_np_inf():
    a = np.inf
    b=1
    print(b<a)

def display_results():
    df = pd.read_csv("{0}/Accumulating results.csv".format(pc.TEMP_DIR))
    df=pd.read_csv("{0}/Sigma_comparison100_repeated_runs_100_time_series.csv".format(pc.RESULTS_DIR))
    print(df)

    f,axes=plt.subplots(1,1)
    axes.errorbar(df['T'],df['Z score mean (inferred means)'],df['Z score std dev (inferred means)'],color='r',label='Conditional version')
    axes.errorbar(df['T'],df['Z score mean (known means)'],df['Z score std dev (known means)'],color='b',label='Using population means')
    axes.set_title("Mean z-score and errorbars against T (each datapoint represents 10000 comparisons)",fontsize=24)
    axes.set_xlabel("T",fontsize=20)
    axes.set_ylabel("z-score",fontsize=20)
    axes.legend(fontsize=15)
    
def test_initialisation(n,T):
    start=time.time()
    array = np.random.choice([0,0,0,0,0,0,0,0,0,1],size=[n,T])
    t1=time.time()-start
    print("Time to initialise as 2-array : {0}".format(t1))
    matrix=[]
    for i in range(n):
        step=int(n/10)
        if not i%step:
            print("{0}% first initialisation complete".format(i/step))
        array = []
        seeds = np.random.choice(10*T,T)
        for seed in seeds:
            np.random.seed(seed=seed)
            if np.random.uniform()<0.2:
                array.append(1)
            else:
                array.append(0)
        matrix.append(array)
        matrix=[]
    
        
    t2=time.time()-start-t1
    print("{0} time series length {1}".format(n,T))
    print("Time to initialise using loop : {0}".format(t2))
    return t1,t2

def timing_test():
    for i in range(1,10):    
        try:
            t1,t2 = test_initialisation(10**i,1000)
        except MemoryError:
            print("Memory overflow for array size [{0},{1}]".format(10**i,1000))
        plt.plot(i,np.log(t1),'x',color='r',markersize=10)
        plt.plot(i,np.log(t2),'x',color='b',markersize=10)

def test_init(ps=[0.1,0.5],n=10,T=10):
    zeros=np.array([[0]*int(10*(1-p)) for p in ps])
    ones=np.array([[1]*int(10*p) for p in ps])
    choices=[np.append(zeros[i],ones[i]) for i in range(len(ps))]
    #print(choices)
    p_arrays = [np.random.choice(choices[i],size=[n,T]) for i in range(len(ps))]
    #print(p_arrays)
    props=[sum(r)/len(r) for a in p_arrays for r in a]
    plt.hist(props,bins=100)

def test_init1(ps=[0.1,0.5],n=10,T=10):
    p_arrays = [np.random.choice([0,1],p=[1-ps[i],ps[i]],size=[n,T]) for i in range(len(ps))]
    #print(p_arrays)
    props=[sum(r)/len(r) for a in p_arrays for r in a]
    plt.hist(props,bins=100)

def test_init_random(n,T):
    ps=[np.random.chisquare(6)/30 for i in range(n)]
    p_array = [np.random.choice([0,1],p=[1-ps[i],ps[i]],size=T) for i in range(len(ps))for j in range(2)]
    props=[sum(r)/len(r) for r in p_array]
    plt.hist(props,bins=100)

n=10000
T=1000
ps=[0.1,0.5]
start= time.time()
test_init(ps=ps,n=n,T=T)
print(time.time()-start)
test_init1(ps=ps,n=n,T=T)
print(time.time()-start)
test_init_random(n,T)
print(time.time()-start)
    
    


   