#!/usr/bin/env python
# coding: utf-8

# # Decision tree
# sources: 
# - https://www.kaggle.com/code/funxexcel/p2-decision-tree-hyperpratameter-tuning-python/notebook 
# 
# - https://www.projectpro.io/recipes/find-optimal-parameters-using-randomizedsearchcv-for-regression
# 
# - https://www.kaggle.com/code/cdabakoglu/heart-disease-classifications-machine-learning
# - https://stackoverflow.com/questions/35062665/how-can-i-analyze-a-confusion-matrix

# In[346]:


#Load libraries
import pandas as pd # data processing
import numpy as np # working with arrays
import matplotlib.pyplot as plt # visualization
from matplotlib import rcParams # figure size
from termcolor import colored as cl # text customization

from sklearn.tree import DecisionTreeClassifier as dtc # tree algorithm
from sklearn.model_selection import train_test_split # splitting the data
from sklearn.metrics import accuracy_score # model precision
from sklearn.tree import plot_tree # tree diagram

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer


# # **`1. Loading data`**

# # **Training data and Test data:**

# In[347]:


#Loading train data
with np.load('/Users/kafiaelmi/Downloads/train_data_label.npz') as data:
    train_data = data['train_data']
    train_label = data['train_label']
    
#Loading test data    
with np.load('/Users/kafiaelmi/Downloads/test_data_label.npz') as data:
    test_data = data['test_data']
    test_label = data['test_label']


# # Preprocessing: preparing input data

# **Reshape for decision tree**

# In[350]:


X_train = train_data
X_train = X_train.reshape(-1,28,28,1)
X_train = train_data/ 255 # we need to normalize the pictures
y_train = train_label


# **Split data**

# In[323]:



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=999) # 0.25 x 0.8 = 0.2


# **Shapes of variables**

# In[324]:


print(cl('X_train shape : {}'.format(X_train.shape), attrs = ['bold'], color = 'yellow'))
print(cl('X_test shape : {}'.format(X_val.shape), attrs = ['bold'], color = 'yellow'))
print(cl('y_train shape : {}'.format(y_train.shape), attrs = ['bold'], color = 'yellow'))
print(cl('y_test shape : {}'.format(y_val.shape), attrs = ['bold'], color = 'yellow'))


# **Build model without hyperparameter tuning and feature extraction**

# In[351]:


model = dtc(criterion = 'entropy', max_depth = 4)
model.fit(X_train, y_train)

pred_model = model.predict(X_val)

print(cl('Accuracy of the model is {:.0%}'.format(accuracy_score(y_val, pred_model))))


# # Hyper parameter tuning

# **Using Decision tree as a model to train the data and setting its parameters**

# In[326]:


#Here we are using Decision tree as a model to train the data and setting its parameters(max_depth etc.) for which we have to use RandomizedSearchCV to get the best set of parameters.
parameters = {'max_depth' : (3,5,7,9,10,15,20,25)
              , 'criterion' : ('gini', 'entropy')
              , 'max_features' : ('auto', 'sqrt', 'log2')
              , 'min_samples_split' : (2,4,6)
             }

DT_grid  = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions = parameters, cv = 5, verbose = True)

DT_grid.fit(X_train,y_train)


# **Check Accuracy and best parameters**

# In[327]:


#Check Accuracy (Not Overfitting anymore)
DT_grid.best_estimator_
print(" Results from Random Search " )
print("\n The best estimator across ALL searched params:\n", DT_grid.best_estimator_)
print("\n The best score across ALL searched params:\n", DT_grid.best_score_)
print("\n The best parameters across ALL searched params:\n", DT_grid.best_params_)


# In[328]:


DT_grid.best_estimator_


# **Now rebuild model with best Estimators**

# In[329]:


DT_Model = DecisionTreeClassifier(criterion='entropy', max_depth=15, max_features='sqrt', min_samples_split= 2, random_state=123)

DT_Model.fit(X_train,y_train)


# # Evaluation

# **Accuracy after hyper parameter tuning**

# In[330]:


print (f'Train Accuracy - : {DT_Model.score(X_train,y_train):.3f}')
print (f'Test Accuracy - : {DT_Model.score(X_val,y_val):.3f}')


# **Confusion matrix**

# In[331]:


# If you don't recognize the 'pattern' in the confusion_matrix() function above, try Brute-force approach first:
def confusion_matrix(y_true, y_pred):
    """A function which takes two lists and returns a confusion matrix with the cells (TN, TP, FP, FN)"""
    #............................................
    # Create empty Matrix
    M = [[0, 0], 
         [0, 0]]
    # Loop over each sample
    for true, pred in zip(y_true, y_pred):
        # Write literally every combination of possible outputs for true and pred by hand
        if true == 0 and pred == 0:
            # Increase the corresponding cell TN
            M[0][0] += 1
        elif true == 0 and pred == 1:
            # Increase FP
            M[0][1] += 1
        elif true == 1 and pred == 0:
            # Increase FN
            M[1][0] += 1
        elif true == 1 and pred == 1:
            # Increase TP
            M[1][1] += 1
    return M

# Now notice how the true and pred values happen to be the same as the index values we increase by 1, 
# thus you can also use the true and pred variables as indices, and that's basically the pattern to recognize here and then write shorter like done above.           
confusion_matrix(y_val, pred_model)
M = np.array(confusion_matrix(y_val, pred_model))
print(M)


# **Precision and recall**

# In[332]:


def precision(M):
    """A function which takes a 2x2 confusion matrix and returns the precision"""
    #...........................................
    TP = M[1, 1]
    FP = M[0, 1]
    return TP/(TP+FP)

    
def recall(M):
    """A function which takes a 2x2 confusion matrix and returns the recall"""
    #...............................................
    TP = M[1, 1]
    FN = M[1, 0]
    return TP/(TP+FN)
    
print('Precision:',round(precision(M),2))
print('Recall:',round(recall(M), 3))


# **Table to find easiest letter**

# In[334]:


pd.crosstab(y_val, pred_model, rownames=['True'], colnames=['Predicted'], margins=True)


# **Average precision**

# In[349]:


from sklearn.metrics import precision_score
precisionScore_sklearn_microavg = precision_score(y_val, pred_model, average='macro')
precisionScore_sklearn_microavg

