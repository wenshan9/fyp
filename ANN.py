#!/usr/bin/env python
# coding: utf-8

# In[9]:


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


# In[10]:


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


# In[11]:


#spilt dataset into training and testing dataset
from sklearn import model_selection
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, stratify = Y)
#perform feature scailing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[12]:


def ANN():
    ann = tf.keras.models.Sequential()     
    accuracy_outcomes = []
    auc_outcomes = []
    fpr_outcomes = []
    tpr_outcomes = []
    
    for i in range(0,10):
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


# In[13]:


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


# In[14]:


#spilt dataset into training and testing dataset
from sklearn import model_selection
from sklearn.model_selection import train_test_split
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size = 0.20, stratify = Y1)
#perform feature scailing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X1_train = sc.fit_transform(X1_train)
X1_test = sc.transform(X1_test)


# In[15]:


def ANN():
    ann = tf.keras.models.Sequential()     
    accuracy_outcomes = []
    auc_outcomes = []
    fpr_outcomes = []
    tpr_outcomes = []
    
    for i in range(0,10):
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


# In[8]:


get_ipython().run_line_magic('reset', '')


# In[ ]:




