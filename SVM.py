#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd
import numpy as np
#import sklearn

get_ipython().system(' pip install sklearn --user')
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import random


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


#spilt dataset into training and testing dataset
from sklearn import model_selection
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, stratify = Y)
#perform feature scailing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[4]:


Y_train


# In[5]:


X_train


# In[6]:


Y_test


# In[10]:


#function of Random forest
def SVM():
    #instantiate the model
    svm = SVC()

    accuracy_outcomes = []
    auc_outcomes = []
    fpr_outcomes = []
    tpr_outcomes = []
    
    for i in range(0,10):
        svm = SVC(kernel = 'rbf')
        svm.fit(X_test, Y_test)
        Y_pred = svm.predict(X_test)
        Y_pred = np.rint(Y_pred)
        
        auc = roc_auc_score(Y_test, Y_pred)
        auc_outcomes.append(auc)
        
        fpr, tpr, _ = roc_curve(Y_test, Y_pred)
        fpr_outcomes.append(fpr)
        tpr_outcomes.append(tpr)
        
        con_matrix = confusion_matrix(Y_test, Y_pred)
        
        accuracy = accuracy_score(Y_test,Y_pred)
        accuracy_outcomes.append(accuracy)
        print("AUC score for every run", auc)
        print("Accuracy for every run",accuracy)
        
    
    mean_accuracy = np.mean(accuracy_outcomes)
    mean_auc = np.mean(auc_outcomes)
    print("Confusion matrix of SVM is", con_matrix)
    print("Mean of accuracy for SVM is", mean_accuracy)
    print("Mean of AUROC for SVM is", mean_auc)
    
    plt.plot(fpr, tpr, marker='.', label='SVM (AUROC = %0.3f)' % auc)
    plt.legend() 
    plt.show()

SVM()


# In[11]:


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


# In[12]:


#spilt dataset into training and testing dataset
from sklearn import model_selection
from sklearn.model_selection import train_test_split
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size = 0.20, stratify = Y1)
#perform feature scailing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X1_train = sc.fit_transform(X1_train)
X1_test = sc.transform(X1_test)


# In[13]:


#function of Random forest
def SVM():
    #instantiate the model
    svm = SVC()

    accuracy_outcomes = []
    auc_outcomes = []
    fpr_outcomes = []
    tpr_outcomes = []
    
    for i in range(0,10):
        svm = SVC(kernel = 'rbf')
        svm.fit(X1_test, Y1_test)
        Y1_pred = svm.predict(X1_test)
        Y1_pred = np.rint(Y1_pred)
        
        auc = roc_auc_score(Y1_test, Y1_pred)
        auc_outcomes.append(auc)
        
        fpr, tpr, _ = roc_curve(Y1_test, Y1_pred)
        fpr_outcomes.append(fpr)
        tpr_outcomes.append(tpr)
        
        con_matrix = confusion_matrix(Y1_test, Y1_pred)
        
        accuracy = accuracy_score(Y1_test,Y1_pred)
        accuracy_outcomes.append(accuracy)
        print("AUC score for every run", auc)
        print("Accuracy for every run",accuracy)
        
    
    mean_accuracy = np.mean(accuracy_outcomes)
    mean_auc = np.mean(auc_outcomes)
    print("Confusion matrix of SVM is", con_matrix)
    print("Mean of accuracy for SVM is", mean_accuracy)
    print("Mean of AUROC for SVM is", mean_auc)
    
    plt.plot(fpr, tpr, marker='.', label='SVM (AUROC = %0.3f)' % auc)
    plt.legend() 
    plt.show()

SVM()


# In[1]:


get_ipython().run_line_magic('reset', '')


# In[ ]:





# In[ ]:




