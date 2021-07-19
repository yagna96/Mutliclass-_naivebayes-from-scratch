#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[5]:


import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# ## SIGMOID FUNCTION to convert CONTINOUS TO PROBABILITIES

# In[9]:


def cont2prob(x):
    return 1/(1 + np.exp(-x))


# ### LOGISTIC REGRESSION TRAIN METHOD using GRADIENT DESCENT ALGORITHM

# In[10]:


def Logistic_gradient_descent_TRAIN(x,y,iteration,lr):
    k=list(x.columns)
    
    
    
   
    
    global w0
    w0 = 0
    
    global w
    w = [0]*len(k)
    
    global iterationlist
    iterationlist=[]
    
    
    global wlist
    wlist=[]
    
    global w0list
    w0list=[]
    
    global jwlist
    jwlist=[]
 
    global mse_list
    mse_list=[]
 
    
    for i in range(iteration):
        
        iterationlist.append(i)
        
        global y_pred_conti_TRAIN
        y_pred_conti_TRAIN = np.dot(x,w) + w0
        
        global y_pred_prob_TRAIN
        y_pred_prob_TRAIN = cont2prob(y_pred_conti_TRAIN)
        
        jw = - np.sum( np.dot(y,np.log(y_pred_prob_TRAIN)) + np.dot( (1 - y), np.log(1 - y_pred_prob_TRAIN) ) )
        
        jwlist.append(jw)
        
        w0d =  np.sum(y_pred_prob_TRAIN-y)
        
        wd =   np.dot((y_pred_prob_TRAIN-y),x)
        
        
        
        w = w - lr*np.array(wd)
        
        wlist.append(w)
        
        
        
        w0 = w0 - lr*np.array(w0d)
        
        w0list.append(w0)
        
        
        
        if i== (iteration-1):
            #print("w0:{} w:{} mse:{}".format(w0,w,mse))
            return w0,w,w0d,wd


# ### FUNCTION CALL TO LOGISTIC REGRESSION USING GRADIENT DESCENT
# #### Logistic_gradient_descent_TRAIN(X_train , y_train , iteration , lr)

# ### LOGISTIC REGRESSION TEST METHOD using GRADIENT DESCENT ALGORITHM

# In[11]:


def Logistic_gradient_descent_TEST(x,y,decision_boundary_probability):
    
    global y_pred_conti_TEST
    y_pred_conti_TEST = np.dot(x,w) + w0
        
    global y_pred_prob_TEST
    y_pred_prob_TEST = cont2prob(y_pred_conti_TEST)
    
    y_pred_class = [1 if i>decision_boundary_probability else 0 for i in y_pred_prob_TEST]
    
    C = confusion_matrix(y_test,y_pred_class)
    global TN,FN,TP,FP
    TN = C[0][0]
    FN = C[1][0]
    TP = C[1][1]
    FP = C[0][1]
    print(C)
    

