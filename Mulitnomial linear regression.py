#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import all the necessary packages.
import pandas as pd 
import numpy as np 
import scipy as scp
import sklearn
import statsmodels.api as sm
import matplotlib.pyplot as plt  

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics 
from sklearn.metrics import confusion_matrix

# Upload the CSV file.
oysters = pd.read_csv('oysters.csv')  

# Print the columns.
print(oysters.info())
oysters.head()


# In[2]:


# Apply the value_counts() method, and assign the results to a new DataFrame.
oysters_sex = oysters['sex'].value_counts()

# View the output.
oysters_sex


# In[3]:


# Set the independent variable.  
X = oysters.drop(['sex'], axis=1) 
# Set the dependent variable. 
y = oysters['sex']   

# Print to check that the sex column has been dropped.
print(list(X.columns.values))  

# Specify the train and test data sets and 
# use 30% as the 'test_size' and a random_state of one.
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.30, random_state = 1, stratify=y) 


# In[4]:


# Import the MinMaxScaler to normalise the data.
from sklearn.preprocessing import MinMaxScaler  

# Create a function, and set values.
scaler = MinMaxScaler(feature_range = (0, 1))  

# Add the X_train data set to the 'scaler' function.
scaler.fit(X_train)

# Specify X_train data set.
X_train = scaler.transform(X_train) 

# Specify X_test data set. 
X_test = scaler.transform(X_test)

# Define the MLR model, and set predictions and parameters.
MLR = LogisticRegression(random_state=0, 
                         multi_class='multinomial', 
                         penalty='none', 
                         solver='newton-cg').fit(X_train, y_train)

# Set the predictions equal to the 'MLR' function and 
# specify the DataFrame.
preds = MLR.predict(X_test) 

# Set the parameters equal to the DataFrame and 
# add the 'get_params' function. 
params = MLR.get_params() 

# Print the parameters, intercept, and coefficients.
print(params)  
print("Intercept: \n", MLR.intercept_)
print("Coefficients: \n", MLR.coef_)


# In[5]:


# Name the model, and set the model to the function.
# Use the add_constant() function to determine the value of y when X=0.
logit_model = sm.MNLogit(y_train, sm.add_constant(X_train))

# Specify how the function returns the results.
result = logit_model.fit()  

# Print the report as a result.summary() function: 
print("Summary for Sex:I/M :\n ", result.summary())


# In[6]:


# Define confusion matrix.
cm = confusion_matrix(y_test, preds)  

# Create visualisation for the MLR.
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1, 2),
             ticklabels=('female', 'infant', 'male'))
ax.yaxis.set(ticks=(0, 1, 2),
             ticklabels=('female', 'infant', 'male'))

# ax.set_ylim(1.5, -0.5)
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j],
                ha='center',
                va='center',
                color='white',
                size='xx-large')
        
# Sets the labels.
plt.xlabel('Predictions', fontsize=16)
plt.ylabel('Actuals', fontsize=16)
plt.title('Confusion Matrix', fontsize=15)

plt.show()


# In[7]:


# Create and print a confusion matrix:
# y_test as the first argument and the predictions as the second argument. 
confusion_matrix(y_test, preds)

# Transform confusion matrix into an array.
cmatrix = np.array(confusion_matrix(y_test, preds))

# Create the DataFrame from cmatrix array. 
pd.DataFrame(cmatrix, index=['female','infant', 'male'],
columns=['predicted_female', 'predicted_infant', 'predicted_male'])


# In[8]:


# Determine accuracy statistics.
print('Accuracy score:', metrics.accuracy_score(y_test, preds))  

# Create a classification report.
class_report=classification_report(y_test, preds)

print(class_report)


# In[ ]:




