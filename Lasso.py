#!/usr/bin/env python
# coding: utf-8

# In[115]:


#import libraries
import pandas as pd
import numpy as np
#import sklearn

get_ipython().system(' pip install sklearn --user')
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import random


# In[116]:


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


# In[117]:


genes_names = X.columns[0:79]
print(genes_names)


# In[119]:


#create object
pipeline = Pipeline([('model',Lasso())])
search = GridSearchCV(pipeline,{'model__alpha':np.arange(0.1,10,0.1)},scoring="neg_mean_squared_error",verbose = 3)
search.fit(X,Y)
search.best_params_
coefficients = search.best_estimator_.named_steps['model'].coef_
importance = np.abs(coefficients)
print(importance)


# In[113]:


genes_names = np.array(genes_names)[importance > 0]
print(genes_names)


# In[120]:


new_upgenes = upgenes[genes_names]


# In[121]:


print(new_upgenes)


# In[136]:


#create new dataset
X = new_upgenes
X.head()
Y = upgenes.Type
Y


# In[123]:


#spilt dataset into training and testing dataset
from sklearn import model_selection
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, stratify = Y1)
#perform feature scailing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[55]:


#function of Random forest
def RanFor():
    #instantiate the model
    rf = RandomForestClassifier()

    accuracy_outcomes = []
    auc_outcomes = []
    fpr_outcomes = []
    tpr_outcomes = []
    
    for i in range(0,10):
        rf = RandomForestClassifier(n_estimators=7)
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


# In[124]:


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


# In[125]:


X1.shape


# In[130]:


genes_names1 = X1.columns[0:57]
genes_names1


# In[127]:


#create object
pipeline = Pipeline([('model',Lasso())])
search = GridSearchCV(pipeline,{'model__alpha':np.arange(0.1,10,0.1)},scoring="neg_mean_squared_error",verbose = 3)
search.fit(X1,Y1)
search.best_params_
coefficients = search.best_estimator_.named_steps['model'].coef_
importance = np.abs(coefficients)
print(importance)


# In[131]:


genes_names1 = np.array(genes_names1)[importance > 0]
print(genes_names1)


# In[132]:


new_downgenes = downgenes[genes_names1]


# In[133]:


#create new dataset
X1 = new_downgenes
X1.head()
Y1 = downgenes.Type
Y1


# In[134]:


#spilt dataset into training and testing dataset
from sklearn import model_selection
from sklearn.model_selection import train_test_split
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size = 0.20, stratify = Y1)
#perform feature scailing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X1_train = sc.fit_transform(X1_train)
X1_test = sc.transform(X1_test)


# In[135]:


#function of Random forest
def RanFor():
    #instantiate the model
    rf = RandomForestClassifier()

    accuracy_outcomes = []
    auc_outcomes = []
    fpr_outcomes = []
    tpr_outcomes = []
    
    for i in range(0,10):
        rf = RandomForestClassifier(n_estimators=7)
        rf.fit(X1_test, Y1_test)
        rf_probs1 = rf.predict_proba(X1_test)
        rf_probs1 = rf_probs1[:,1]
        rf_probs1 = np.rint(rf_probs1)
        
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


# In[114]:


get_ipython().run_line_magic('reset', '')


# In[ ]:




