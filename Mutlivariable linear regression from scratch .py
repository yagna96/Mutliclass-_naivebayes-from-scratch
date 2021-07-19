#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[28]:





# In[29]:


def normalisation(df):
    l=list(df.columns)
    n=len(l)
   
    for i in range(n):
        pmax=max(df[i])
        pmin=min(df[i])
        df[i]= np.array((df[i]-pmin)/(pmax-pmin))
   
    
    return df


# In[30]:


# multivariable gradient descent with dataframe as input 
def multivariable_regression(df,iteration,learning_rate):
    l=list(df.columns)
    n=len(l)
    p=l[:n-1]
    x=np.array(df[p])
    y=np.array(df[n-1])
    dw=[0]*(x.shape[1])
    alpha=learning_rate
    j_store=[]
    w=[0]*(x.shape[1])
    w0=0
    iteration=iteration 
    iteration_store=[]
    for i in range(iteration):
        y_pre = np.dot(x,w)+w0
        j=(.5)*np.sum((y_pre -y)**2)
        j_store.append(j)
        iteration_store.append(i)
        dz=(y_pre-y)
        db=np.sum(dz)
        dw=(np.dot(dz,x))
        w= w-alpha*np.array(dw)
        w0 = w0-alpha*np.array(db)
   
        
        
        sum1=0
        for d in dz:
            sum1=sum1+d**2
        mse1= sum1/len(dz)
    std1=np.std(dz)
   
    plt.plot(j_store,iteration_store)
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.show()
   

   
    print('mse is', mse1)
    print('standard deviation ',std1)
   
    return w,w0,y_pre,mse1,std1
    


# In[ ]:





# In[8]:


def multivariable_regression_stoch(df,learning_rate):
    l=list(df.columns)
    n=len(l)
    p=l[:n-1]
    x=np.array(df[p])
    y=np.array(df[n-1])
    dw=[0]*(x.shape[1])
    w_store=[]
    w0_store=[]
    alpha=learning_rate
    j_store=[]
    w=[0]*(x.shape[1])
    w0=0
   
    iteration_store=[]
    for i in range(n):
        y_pre = np.dot(x,w)+w0
        j=(.5)*np.sum((y_pre -y)**2)
        j_store.append(j)
        iteration_store.append(i)
        dz=(y_pre-y)
        db=np.sum(dz)
        dw=(np.dot(dz,x))
        w= w-alpha*np.array(dw)
        w_store.append(w)
        w0 = w0-alpha*np.array(db)
        w0_store.append(w0)
   
        
        
        sum1=0
        for d in dz:
            sum1=sum1+d**2
        mse1= sum1/len(dz)
    std1=np.std(dz)
   
    plt.plot(j_store,iteration_store)
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.show()
    
    #plt.plot(j_store,w_store)
    #plt.plot(j_store,w0_store)

   
    print('mse is', mse1)
    print('standard deviation ',std1)
   
    return w,w0,y_pre,mse1,std1
    


# In[3]:


def holdout_method(df,iteration,alpha): # holdout method algorithm ,spliting the 80% data into training and iterating it 30 times
    msecollect=[]
    iteration=iteration
    learning_rate=alpha
    
    for i in range(30):
            df=df.sample(frac=1)
            df1=df[0:int(0.8*len(df))]
            w,w0,y_pre,mse,std=multivariable_regression(df1,iteration,learning_rate)
            msecollect.append(mse)
    sum2=np.sum(msecollect) 
    avgmse=sum2/len(msecollect)
    print(avgmse)    
        


# In[ ]:





# In[ ]:




