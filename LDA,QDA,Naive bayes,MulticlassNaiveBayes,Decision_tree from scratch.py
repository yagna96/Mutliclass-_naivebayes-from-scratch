#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cmath
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve


# 
# ## TRAIN TEST SPLIT METHOD

# In[20]:


def Train_Test_Split(df):
    n = len(df.columns)
    x = df.drop(n-1,axis=1)
    y = df[n-1]
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2)
    #print(" lenght of DATA FRAME :{} ".format(len(df)))
    #print(" lenght of TRAIN DATA :{} ".format(len(X_train)))
    #print(" lenght of TEST DATA  :{} ".format(len(X_test)))


# ## LDA TRAIN METHOD

# In[21]:


def LDA_Train(X_train,y_train):
    
    global attributes
    attributes = len(X_train.columns)
    
    global class_labels
    class_labels = np.unique(y_train)
    
    global P
    P = []
    
    global weighted_covariance
    
    global inverse_weighted_covariance
    
    global mean_class
    mean_class = []
    
    weighted_covariance = np.zeros((attributes,attributes))
    
    for c in range(len(class_labels)):
        
        X_c = X_train[y_train == class_labels[c]]
        
        Probability = (len(X_c)) / (len(X_train))
        P.append(Probability)
        
        mean_c = np.mean(X_c , axis = 0)
        mean_class.append(list(mean_c))
        
        covariance_matrix = ((X_c - mean_c).T.dot(X_c - mean_c))
        
        weighted_covariance = weighted_covariance + covariance_matrix
        
    weighted_covariance = weighted_covariance/(len(X_train)-2)
    
    inverse_weighted_covariance = np.linalg.inv(weighted_covariance)
    
    print("Probabilities of Classes")
    print(P)
    
    print("")
    
    print("mean_class:")
    print(mean_class)
    
    
    print("")
    
    print("weighted_covariance:")
    print(weighted_covariance)
    
    print("")
    
    print("Weighted Covariance Matrix Inverse:")
    print(inverse_weighted_covariance)
    


# ## LDA TEST METHOD

# In[22]:


def LDA_Test(X_test,y_test):
    
    global y_pred_list
    y_pred_list = []
    
    for i in range(len(X_test)):
        X = np.array(X_test.iloc[i]).reshape(len(X_test.columns),1)
        
        global del_x_List
        del_x_List = []
    
        for c in range(len(class_labels)):
            
            mean_class[c] = np.array(mean_class[c])
            
            del_x = np.linalg.multi_dot([mean_class[c],inverse_weighted_covariance,X]) + np.log(P[c]) -0.5 * np.linalg.multi_dot([mean_class[c],inverse_weighted_covariance,mean_class[c].T])
            del_x_List.append(del_x)
        
        c = np.argmax(del_x_List)
        y_pred_class = class_labels[c]
        y_pred_list.append(y_pred_class)
       
    C = confusion_matrix(y_test,y_pred_list)
    global TN,FN,TP,FP
    TN = C[0][0]
    FN = C[1][0]
    TP = C[1][1]
    FP = C[0][1]
    print(C)
  
        


# ## QDA TRAIN METHOD

# In[23]:


def QDA_Train(X_train,y_train):
    
    attributes = len(X_train.columns)
    
    global class_labels
    class_labels = np.unique(y_train)
    
    global P
    P = []
    
    global weighted_covariance
    
    global inverse_weighted_covariance
    
    global mean_class
    mean_class = []
    
    global inv_matrix
    inv_matrix = []
    
    weighted_covariance = np.zeros((attributes,attributes))
    
    for c in range(len(class_labels)):
        
        X_c = X_train[y_train == class_labels[c]]
        
        Probability = (len(X_c)) / (len(X_train))
        P.append(Probability)
        
        mean_c = np.mean(X_c , axis = 0)
        mean_class.append(list(mean_c))
        
        covariance_matrix = (1 / (len(X_c) - 1))*((X_c - mean_c).T.dot(X_c - mean_c))
        inv_cov_mat = np.linalg.inv(covariance_matrix)
        inv_matrix.append(inv_cov_mat)
        
    print("Probabilities of Classes")
    print(P)
    
    print("")
    
    print("mean_class:")
    print(mean_class)
    
    print("")
    
    for c in class_labels:
        print("")
        print("inv_cov_mat",c)
        print(inv_matrix[c])
    
    print("")
      


# ## QDA TEST METHOD

# In[24]:


def QDA_Test(X_test,y_test):
    
    global del_max_list
    del_max_list = []
    
    global y_pred_list
    y_pred_list = []
    
    for i in range(len(X_test)):
        X = np.array(X_test.iloc[i]).reshape(len(X_test.columns),1)
        
        global del_x_List
        del_x_List = []
    
        for c in range(len(class_labels)):
              
            mean_class[c] = np.array(mean_class[c])
            
            
            del_x = np.linalg.multi_dot([mean_class[c],inv_matrix[c],X]) + np.log(P[c]) -0.5 * np.linalg.multi_dot([mean_class[c],inv_matrix[c],mean_class[c].T])-0.5*np.linalg.multi_dot([X.T,inv_matrix[c],X])+0.5*np.log(np.linalg.det([inv_matrix[c]]) )
            
            del_x_List.append(del_x)
       
        c = np.argmax(del_x_List)
        y_pred = class_labels[c]
        y_pred_list.append(y_pred)
        
    C = confusion_matrix(y_test,y_pred_list)
    global TN,FN,TP,FP
    TN = C[0][0]
    FN = C[1][0]
    TP = C[1][1]
    FP = C[0][1]
    print(C)
  
        


# ## PERFORMANCE MEASURE

# In[25]:


def performance_measure(TP,FP,TN,FN):
    global sensitivity,specificity,precision,F_measure
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision   = TP / (TP + FP)
    F_measure   = (2)* (precision * sensitivity) / (precision + sensitivity)
    Score       = (TP+TN)/(TP+TN+FP+FN)
    print("""    sensitivity : {}
    specificity : {}
    precision   : {}
    F_measure   : {}""" .format(sensitivity,specificity,precision,F_measure) )
    print(" ")
    print("Score :",Score)


# ### PLOT ROC 

# In[26]:


def ROC(y_test,y_pred_list):
        tpr,fpr,threshold = roc_curve(y_test ,y_pred_list,drop_intermediate=False)
        plt.plot(tpr,fpr,color='red',lw=5)
        plt.show()


# ### CROSS VALIDATION

# In[27]:


def Cross_validation(df,iterations):
    for i in range(iterations):
        Train_Test_Split(df)
        LDA_Test(X_test,y_test)
        accuracy = (TP+TN)/len(X_test)
        print("accuracy",accuracy)
        
        


# # NAVIE BAYES TRAIN METHOD

# In[28]:


def NavieBayes_Train(X_train,y_train):
    
    global P
    P = []
    
    global class_labels
    class_labels = np.unique(y_train)
    
    A = list(X_train.columns)
    
    global P_a_au_c
    P_a_au_c = []
    
    for c in class_labels:
        
        X_c = X_train[y_train == class_labels[c]]
        
        Probability = (len(X_c)) / (len(X_train))
        P.append(Probability)    
        
    for j in A:
            
        X_a = np.unique(X_train[j])
            
        global P_au_c
        P_au_c = []
            
        for i in X_a:
        
            global P_c
            P_c = []
            
            for c in range(len(class_labels)):
                
                Y = X_train[((X_train[j] == i) & (y_train == class_labels[c]))]
                
                P_Xa_given_c = (len(Y) + 1) / (len(X_train[y_train == class_labels[c]]) + len( X_a) )
                
                P_c.append(P_Xa_given_c)
            
            P_au_c.append(P_c)
            
        P_a_au_c.append(P_au_c)
    #print(np.array(P_a_au_c))
    return np.array(P_a_au_c)
    


# # NAVIE BAYES TEST METHOD

# In[29]:


def att(x):
    l = [ ]
    for i in x.columns:
        a = list(np.unique(X_train[i]))
        l.append(a)
    l = np.array(l)
    return l


# In[30]:


def NavieBayes_Test(X_test):
    global class_labels
    
    global P
    P = []
    
    for i in class_labels:
        X_c = X_train[y_train == class_labels[i]]
        Probability = (len(X_c)) / (len(X_train))
        P.append(Probability)    
        
    P = np.array(P)
    a=att(X_test)
    mat=np.array(P_a_au_c)
    class_labels = np.unique(y_train)
    cls=class_labels
    global c
    global y_pred
    y_pred = [ ]
    data = np.array(X_test)
    for x in data:
        k = 0 
        d = [ ]
        for c in cls:
            p = P[k]
            for  idx in range(len(x)):
                p *= mat[idx][np.where(a[idx]==x[idx])][0][np.where(cls==c)]
            k+=1
            d.append(p)
        d = np.array(d)
        y_pred.append(cls[np.argmax(d)])
    c = np.array(confusion_matrix(y_test,y_pred))
    return c


# # MULTINOMIAL NAVIE BAYES

# ## MULTINOMIAL NAVIE BAYES TRAIN METHOD

# In[44]:


def Multinomial_Navie_bayes_train(X_train,y_train):
    
    global class_labels
    
    class_labels = list(np.unique(y_train))
    
    A = list(X_train.columns)
    
    global X_class_sum
    X_class_sum = []
    
    global  P
    P = [] 
    
    for c in range(len(class_labels)):
        
        X_c = X_train[y_train == class_labels[c]]
        
        Probability = (len(X_c)) / (len(X_train))
        P.append(Probability)
    
    
    for i in range(len(class_labels)):
        
        x = np.array(X_train[y_train == class_labels[i] ].sum())
        X_class_summ = x.sum()
        X_class_sum.append(X_class_summ)
        
    global Psum_c_a
    Psum_c_a = []
    
    
    for j in class_labels:
        
        Psum_a = []
        
        for i in A :
            
            X_sum_a_c = X_train[i][y_train == j].sum()
            
            P_sum = np.log10((X_sum_a_c + 1)/(X_class_sum[j] + len(X_train.columns)))
            
            Psum_a.append(P_sum)
            
        Psum_c_a.append(Psum_a)
    
    Psum_c_a = np.array(Psum_c_a)
    
    print("Probability of y == class :")
    print(P)
    print("")
    print("log of Probability of The Attributes for class y == c :")
    print(Psum_c_a)
        


# ## MULTINOMAIAL NAVIE BAYES TEST METHOD

# In[45]:


def Multinomial_Navie_bayes_test(X_test,y_test):
    global Del_X_List
    
    
    
    global y_pred_list
    y_pred_list = []
        
    for i in range(len(X_test)):
        X = np.array(X_test.iloc[i])
        
        Del_X_List = []
        
        for j in range(len(class_labels)):
            Del_X = P[j]
            
            
            
            for k in range(len(X_test.columns)):
                
                Del_X += X[k] * Psum_c_a[class_labels[j]][k]
            
            Del_X_List.append(Del_X)
        
        c = np.argmax(Del_X_List)
        y_pred = class_labels[c]
        y_pred_list.append( y_pred)
    
    C = confusion_matrix(y_test,y_pred_list)
    global TN,FN,TP,FP
    TN = C[0][0]
    FN = C[1][0]
    TP = C[1][1]
    FP = C[0][1]
    print(C)
        

        
    


# # DECISION TREE

# In[52]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import random


# In[53]:


## data purity
def check_purity(data):
    
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False
    
    


# In[54]:


## classification
def classify_data(data):
    
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification


# In[55]:


## splits
def get_potential_splits(data):
    
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):        # excluding the last column which is the label
        potential_splits[column_index] = []
        values = data[:, column_index]
        unique_values = np.unique(values)

        for index in range(len(unique_values)):
            if index != 0:
                current_value = unique_values[index]
                previous_value = unique_values[index - 1]
                potential_split = (current_value + previous_value) / 2
                
                potential_splits[column_index].append(potential_split)
    
    return potential_splits


# In[56]:


### split data
def split_data(data, split_column, split_value):
    
    split_column_values = data[:, split_column]

    data_below = data[split_column_values <= split_value]
    data_above = data[split_column_values >  split_value]
    
    return data_below, data_above


# In[57]:


### entropy calculation 
def calculate_entropy(data):
    
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy


# In[58]:


def calculate_overall_entropy(data_below, data_above):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy =  (p_data_below * calculate_entropy(data_below) 
                      + p_data_above * calculate_entropy(data_above))
    
    return overall_entropy


# In[59]:


def determine_best_split(data, potential_splits):
    
    overall_entropy = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value


# In[60]:


### decision tree algorithm 
def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5):
    
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS
        COLUMN_HEADERS = df.columns
        data = df.values
    else:
        data = df           
    
    
    # base cases
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
        
        return classification

    
    # recursive part
    else:    
        counter += 1

        # helper functions 
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)
        
        # instantiate sub-tree
        feature_name = COLUMN_HEADERS[split_column]
        question = "{} <= {}".format(feature_name, split_value)
        sub_tree = {question: []}
        
        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth)
        
        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base cases).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
            
        
        return sub_tree
    
    


# In[61]:


### classification
def classify_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split()

    # ask question
    if example[feature_name] <= float(value):
        answer = tree[question][0]
    else:
        answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)


# In[62]:


### accuracy 
def accuracy(y_prelist,df1):
    y=list(df1.label)
    return accuracy_score(y,y_prelist)
    


# In[157]:


def pruning(steps,train,test):
    tree = decision_tree_algorithm(train, max_depth=steps)
    y_prelist=[]
    for i in range(len(test)):
            example=test.iloc[i]
            y_pre = classify_example(example,tree)
            y_prelist.append(y_pre)
    df1=test
    ac=accuracy(y_prelist,df1)
    print('for no of steps =',steps)
    print('accuracy is =',ac)


# In[ ]:




