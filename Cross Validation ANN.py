#!/usr/bin/env python
# coding: utf-8

# In[11]:


#import libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
#import tensorflow

get_ipython().system(' pip install tensorflow --user')
import tensorflow as tf
#import sklearn

get_ipython().system(' pip install sklearn --user')
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, accuracy_score
import random
from sklearn.model_selection import StratifiedKFold


# In[12]:


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


# In[13]:


#perform feature scailing for input
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# In[14]:


def ANN():
    ann = tf.keras.models.Sequential()
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    accuracy_outcomes = []
    auc_outcomes = []
    fpr_outcomes = []
    tpr_outcomes = []
    
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        ann.add(tf.keras.layers.Dense(units=6, activation ="relu"))
        ann.add(tf.keras.layers.Dense(units=6, activation ="relu"))
        ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
        ann.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])
        batch_size = 32
        epochs = 100
        ann.fit(X_test, Y_test, batch_size = batch_size, epochs = 100, verbose = 1)
        Y_pred = ann.predict(X_test)
        Y_pred = np.rint(Y_pred)
        
        ANN_auc = roc_auc_score (Y_test, Y_pred)
        auc_outcomes.append(ANN_auc)
        
        ANN_fpr, ANN_tpr, _ = roc_curve(Y_test, Y_pred)
        fpr_outcomes.append(ANN_fpr)
        tpr_outcomes.append(ANN_tpr)
        
        con_matrix = confusion_matrix(Y_test, Y_pred)
        
        accuracy = accuracy_score(Y_test, Y_pred)
        accuracy_outcomes.append(accuracy)
        print("AUC score for every run", ANN_auc)
        print("Accuracy for every run",accuracy)
        
    mean_accuracy = np.mean(accuracy_outcomes)
    mean_auc = np.mean(auc_outcomes)
    print("Confusion matrix of ANN is", con_matrix)
    print("Mean of accuracy for ANN is", mean_accuracy)
    print("Mean of AUROC for ANN is",mean_auc)

    plt.plot(ANN_fpr, ANN_tpr, marker='.', label='ANN (AUROC = %0.3f)' % ANN_auc)
    # Title
    plt.title('ROC Plot')
    # Axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # Show legend
    plt.legend() 
    # Show plot
    plt.show()
        
ANN()


# In[15]:


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


# In[16]:


#perform feature scailing for input
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X1 = sc.fit_transform(X1)


# In[17]:


def ANN():
    ann = tf.keras.models.Sequential()
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    accuracy_outcomes = []
    auc_outcomes = []
    fpr_outcomes = []
    tpr_outcomes = []
    
    for train_index, test_index in skf.split(X1, Y1):
        X1_train, X1_test = X1[train_index], X1[test_index]
        Y1_train, Y1_test = Y1[train_index], Y1[test_index]
        ann.add(tf.keras.layers.Dense(units=6, activation ="relu"))
        ann.add(tf.keras.layers.Dense(units=6, activation ="relu"))
        ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
        ann.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])
        batch_size = 32
        epochs = 100
        ann.fit(X1_test, Y1_test, batch_size = batch_size, epochs = 100, verbose = 1)
        Y1_pred = ann.predict(X1_test)
        Y1_pred = np.rint(Y1_pred)
        
        ANN_auc = roc_auc_score (Y1_test, Y1_pred)
        auc_outcomes.append(ANN_auc)
        
        ANN_fpr, ANN_tpr, _ = roc_curve(Y1_test, Y1_pred)
        fpr_outcomes.append(ANN_fpr)
        tpr_outcomes.append(ANN_tpr)
        
        con_matrix = confusion_matrix(Y1_test, Y1_pred)
        
        accuracy = accuracy_score(Y1_test, Y1_pred)
        accuracy_outcomes.append(accuracy)
        print("AUC score for every run", ANN_auc)
        print("Accuracy for every run",accuracy)
        
    mean_accuracy = np.mean(accuracy_outcomes)
    mean_auc = np.mean(auc_outcomes)
    print("Confusion matrix of ANN is", con_matrix)
    print("Mean of accuracy for ANN is", mean_accuracy)
    print("Mean of AUROC for ANN is",mean_auc)

    plt.plot(ANN_fpr, ANN_tpr, marker='.', label='ANN (AUROC = %0.3f)' % ANN_auc)
    # Title
    plt.title('ROC Plot')
    # Axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # Show legend
    plt.legend() 
    # Show plot
    plt.show()
        
ANN()


# In[10]:


get_ipython().run_line_magic('reset', '')


# In[ ]:




