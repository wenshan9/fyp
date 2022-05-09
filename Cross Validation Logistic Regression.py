#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#import sklearn

get_ipython().system(' pip install sklearn --user')
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import random
from sklearn.model_selection import StratifiedKFold


# In[2]:


#upregulated genes
#read dataset
upgenes = pd.read_csv(r"C:\Users\wensh\Desktop\wen shan\FYP\upgenes.csv")
upgenes.head()
upgenes.set_index('Gene_Name', inplace=True)
#remove the column type for X
X = upgenes.drop(['Type'], axis="columns")
X.head()
#create Y
Y = upgenes.Type
Y


# In[3]:


#perform feature scailing for input
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# In[4]:


#function of LogReg to fit data, find mean 
def LogReg():
    #instantiate the model
    logreg = LogisticRegression()
    
    #create stratifiedkfold object
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    accuracy_outcomes = []
    auc_outcomes = []
    fpr_outcomes = []
    tpr_outcomes = []
    
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        logreg.fit(X_test, Y_test)
        Y_pred = logreg.predict_proba(X_test)
        Y_pred = Y_pred[:,1]
        Y_pred = np.rint(Y_pred)
        
        auc = roc_auc_score (Y_test, Y_pred)
        auc_outcomes.append(auc)
        
        fpr,tpr,_ = roc_curve(Y_test, Y_pred)
        fpr_outcomes.append(fpr)
        tpr_outcomes.append(tpr)

        con_matrix = confusion_matrix(Y_test, Y_pred)
        
        accuracy = accuracy_score(Y_test, Y_pred)
        accuracy_outcomes.append(accuracy)
        print("AUC score for every run", auc)
        print("Accuracy for every run",accuracy)
    
    mean_accuracy = np.mean(accuracy_outcomes)
    mean_auc = np.mean(auc_outcomes)
    
    print("Confusion matrix of Logistic Regression is",con_matrix)
    print("Mean of accuracy for Logistic Regression is", mean_accuracy)
    print("Mean of AUROC for Logistic Regression is", mean_auc)
    
    plt.plot(fpr,tpr,label = "data 1, auc = "+str(auc))
    plt.legend(loc=4)
    plt.show
    
LogReg()


# In[5]:


#downregulated genes
#read dataset
downgenes = pd.read_csv(r"C:\Users\wensh\Desktop\wen shan\FYP\downgenes.csv")
downgenes.head()
downgenes.set_index('Gene_Name', inplace = True)
#remove the column type for X
X1 = downgenes.drop(['Type'], axis="columns")
X1.head()
#create Y1
Y1 = downgenes.Type
Y1


# In[6]:


#perform feature scailing for input
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X1 = sc.fit_transform(X1)


# In[7]:


#function of LogReg to fit data, find mean 
def LogReg():
    #instantiate the model
    logreg = LogisticRegression()
    
    #create stratifiedkfold object
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    accuracy_outcomes = []
    auc_outcomes = []
    fpr_outcomes = []
    tpr_outcomes = []
    
    for train_index, test_index in skf.split(X1, Y1):
        X1_train, X1_test = X1[train_index], X1[test_index]
        Y1_train, Y1_test = Y1[train_index], Y1[test_index]
        logreg.fit(X1_test, Y1_test)
        Y1_pred = logreg.predict_proba(X1_test)
        Y1_pred = Y1_pred[:,1]
        Y1_pred = np.rint(Y1_pred)
        
        auc = roc_auc_score (Y1_test, Y1_pred)
        auc_outcomes.append(auc)
        
        fpr,tpr,_ = roc_curve(Y1_test, Y1_pred)
        fpr_outcomes.append(fpr)
        tpr_outcomes.append(tpr)

        con_matrix = confusion_matrix(Y1_test, Y1_pred)
        
        accuracy = accuracy_score(Y1_test, Y1_pred)
        accuracy_outcomes.append(accuracy)
        print("AUC score for every run", auc)
        print("Accuracy for every run",accuracy)
    
    mean_accuracy = np.mean(accuracy_outcomes)
    mean_auc = np.mean(auc_outcomes)
    
    print("Confusion matrix of Logistic Regression is",con_matrix)
    print("Mean of accuracy for Logistic Regression is", mean_accuracy)
    print("Mean of AUROC for Logistic Regression is", mean_auc)
    
    plt.plot(fpr,tpr,label = "data 1, auc = "+str(auc))
    plt.legend(loc=4)
    plt.show
    
LogReg()


# In[ ]:




