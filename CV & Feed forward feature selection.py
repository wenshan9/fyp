#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd
import numpy as np
#import sklearn

get_ipython().system(' pip install sklearn --user')
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import mlxtend
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
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


#instantiate the model
logreg = LinearRegression()
    
sfs1 = sfs(logreg, k_features = 40, forward = True, verbose = 2, scoring = 'neg_mean_squared_error')
sfs1 = sfs1.fit(X,Y)
    
#get genes name
genes_names = list(sfs1.k_feature_names_)


# In[4]:


print(genes_names)


# In[5]:


#create selected upregulated genes
new_upgenes = upgenes[genes_names]
#create new dataset
X = new_upgenes
X.head()
Y = upgenes.Type
Y


# In[6]:


#perform feature scailing for input
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# In[7]:


#function of Random forest
def RanFor():
    #instantiate the model
    rf = RandomForestClassifier()
    
    #create stratifiedkfold object
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    accuracy_outcomes = []
    auc_outcomes = []
    fpr_outcomes = []
    tpr_outcomes = []
    
    for train_index, test_index in skf.split(X, Y):
        rf = RandomForestClassifier(n_estimators=7)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        rf.fit(X_test, Y_test)
        rf_probs = rf.predict_proba(X_test)
        rf_probs = rf_probs[:,1]
        rf_probs = np.rint(rf_probs)  #round off the rf_probs 
        
        auc = roc_auc_score (Y_test, rf_probs)
        auc_outcomes.append(auc)
        
        fpr, tpr, _ = roc_curve(Y_test, rf_probs)
        fpr_outcomes.append(fpr)
        tpr_outcomes.append(tpr)
        
        con_matrix = confusion_matrix(Y_test, rf_probs)
    
        accuracy = accuracy_score(Y_test, rf_probs)
        accuracy_outcomes.append(accuracy)
        print("AUC score for every run", auc)
        print("Accuracy for every run",accuracy)
    
    mean_accuracy = np.mean(accuracy_outcomes)
    mean_auc = np.mean(auc_outcomes)
    print("Confusion matrix of Random Forest is", con_matrix)
    print("Mean of accuracy for Random Forest is", mean_accuracy)
    print("Mean of AUROC for Random Forest is", mean_auc)
    
    plt.plot(fpr, tpr, marker='.', label='Random Forest (AUROC = %0.3f)' % auc)
    plt.legend() 
    plt.show()
    
RanFor()


# In[8]:


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


# In[9]:


#instantiate the model
logreg = LinearRegression()
    
sfs1 = sfs(logreg, k_features = 40, forward = True, verbose = 2, scoring = 'neg_mean_squared_error')
sfs1 = sfs1.fit(X1,Y1)
    
#get genes name
genes_names1 = list(sfs1.k_feature_names_)


# In[10]:


print(genes_names1)


# In[11]:


#create selected downregulated genes
new_downgenes = downgenes[genes_names1]

#create new dataset
X1 = new_downgenes
X1.head()
Y1 = downgenes.Type
Y1


# In[12]:


#perform feature scailing for input
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X1 = sc.fit_transform(X1)


# In[14]:


#function of Random forest
def RanFor():
    #instantiate the model
    rf = RandomForestClassifier()
    
    #create stratifiedkfold object
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    accuracy_outcomes = []
    auc_outcomes = []
    fpr_outcomes = []
    tpr_outcomes = []
    
    for train_index, test_index in skf.split(X1, Y1):
        rf = RandomForestClassifier(n_estimators=7)
        X1_train, X1_test = X1[train_index], X1[test_index]
        Y1_train, Y1_test = Y1[train_index], Y1[test_index]
        rf.fit(X1_test, Y1_test)
        rf_probs1 = rf.predict_proba(X1_test)
        rf_probs1 = rf_probs1[:,1]
        rf_probs1 = np.rint(rf_probs1)  #round off the rf_probs 
        
        auc = roc_auc_score (Y1_test, rf_probs1)
        auc_outcomes.append(auc)
        
        fpr, tpr, _ = roc_curve(Y1_test, rf_probs1)
        fpr_outcomes.append(fpr)
        tpr_outcomes.append(tpr)
        
        con_matrix = confusion_matrix(Y1_test, rf_probs1)
    
        accuracy = accuracy_score(Y1_test, rf_probs1)
        accuracy_outcomes.append(accuracy)
        print("AUC score for every run", auc)
        print("Accuracy for every run",accuracy)
    
    mean_accuracy = np.mean(accuracy_outcomes)
    mean_auc = np.mean(auc_outcomes)
    print("Confusion matrix of Random Forest is", con_matrix)
    print("Mean of accuracy for Random Forest is", mean_accuracy)
    print("Mean of AUROC for Random Forest is", mean_auc)
    
    plt.plot(fpr, tpr, marker='.', label='Random Forest (AUROC = %0.3f)' % auc)
    plt.legend() 
    plt.show()
    
RanFor()


# In[ ]:




