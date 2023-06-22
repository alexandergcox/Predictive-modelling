#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import all the necessary packages.
import pandas as pd
import numpy as np


# In[3]:


# Read the provided CSV file/data set.
df = pd.read_csv('customer_data.csv') 

# Print the output.
print(df.info())
df.head()


# In[4]:


# Specify the DataFrame column, and add/determine the values.
df['Edu'].value_counts() 


# In[5]:


# Create two lists: one with initial and one with new values.
intial_vals = ['illiterate', 'unknown', 'basic',
               'high', 'university', 'professional']

new_vals = ['other', 'other', 'pre-school',
            'high-school', 'uni', 'masters']

# Create a for loop to replace the values.
for old_val, new_val in zip(intial_vals, new_vals):
    df.loc[df['Edu'].str.contains(old_val),'Edu' ] = new_val
# Specify the DataFrame column, and add/determine the values.
# Display all the unique values/check changes.
df['Edu'].unique()df['Edu'].value_counts() 


# In[6]:


# Specify the DataFrame column, and add/determine the values.
df['Edu'].value_counts() 


# In[10]:


#The order of the Edu column is meaningful and the order matters, so apply LabelEncoder to this column. 
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


# In[13]:


# Create a class and write a user-defined function.
class MyLabelEncoder(LabelEncoder):
    def fit(self, y):
        y = column_or_1d (y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self


# In[14]:


# Order lists of the values for the Edu column.
Edu_order = ['other', 'pre-school', 'high-school', 'uni', 'masters']

# Create an instance of MyLabelEncoder.
le = MyLabelEncoder()

# Fit the encoder with the ordered values.
le.fit(Edu_order)

# Apply the LabelEncoder to the Edu column in the DataFrame.
df['Edu'] = df['Edu'].apply(lambda x: x if x in Edu_order else 'other')
df['Edu'] = le.transform(df['Edu'])

# View the DataFrame
print(df.head())


# In[15]:


# Create dummy variables for 'Comm' column
comm_dummies = pd.get_dummies(df['Comm'], prefix='Comm', drop_first = True)
df = pd.concat([df, comm_dummies.astype(int)], axis=1)

# Create dummy variables for 'Last_out' column
last_out_dummies = pd.get_dummies(df['Last_out'], prefix='Last_out', drop_first = True)

# Join the new columns to the DataFrame.
df = pd.concat([df, last_out_dummies.astype(int)], axis=1)

# Drop the original string columns
df.drop(['Comm', 'Last_out'], axis=1, inplace=True)

# View the updated DataFrame
print(df.head())


# In[16]:


# Set the variables.
X = df.drop('Target', axis = 1)
y = df['Target']

# Import the VIF package.
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a VIF dataframe.
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
  
# Calculate VIF for each feature.
vif_data['VIF'] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]

# View the DataFrame.
vif_data.sort_values('VIF',ascending=True)


# In[17]:


# Drop the columns with VIF > 10 to avoid multicollinearity problems.
X = X.drop(['Last_out_success', 'Last_out_nonexistent', 'Age', 'Var_rate', 'PosDays', 'Conf_idx',
              'Month_rate', 'Price_idx', 'Quarterly_emp'],
             axis = 1)

# View the DataFrame.
print(X.shape)
X.head()


# In[18]:


# Recalculate VIF after removal of columns.
# VIF dataframe.
vif_data2 = pd.DataFrame()
vif_data2['feature'] = X.columns
  
# Calculate VIF for each feature.
vif_data2['VIF'] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]

# View the DataFrame.
vif_data2.sort_values('VIF',ascending=True)


# In[19]:


# Determine whether values in a column are balanced.
df['Target'].value_counts()  


# In[20]:


# Create a plot with Seaborn.
import seaborn as sns

sns.set_theme(style='darkgrid')
ax = sns.countplot(x='Target', data=df)
ax.set_title('Target Imbalance')


# In[21]:


# Handles unbalanced data (scikit-learn needed)
get_ipython().system('pip install imblearn  ')

# Optimised linear, algebra, integrations (scientific)
get_ipython().system('pip install scipy  ')

# Simple tools for predictive data analytics
get_ipython().system('pip install scikit-learn  ')

# Oversampling technique; creates new samples from data
get_ipython().system('pip install SMOTE ')


# In[22]:


# Import all the necessary packages:
import statsmodels.api as sm   
import imblearn
from imblearn.over_sampling import SMOTE  
from sklearn.model_selection import train_test_split 

# Apply SMOTE class as the target variable is not balanced.
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Balance the training data.
os_data_X, os_data_y = os.fit_resample(X_train, y_train)  

# Create two DataFrames for X and one for y:
os_data_X = pd.DataFrame(data = os_data_X, columns = X.columns) 
os_data_y = pd.DataFrame(data = os_data_y, columns = ['Target'])

# View DataFrame.
print(os_data_X.head())
os_data_y.head()


# In[23]:


# Determine whether values in a column are balanced by counting the values.
os_data_y['Target'].value_counts()


# In[24]:


sns.set_theme(style ='darkgrid')
ax = sns.countplot(x ='Target', data = os_data_y)
ax.set_title("New Balanced Target")


# In[25]:


dur = sns.regplot(x = 'Duration',
                  y= 'Target',
                  data= df,
                  logistic= True).set_title("Duration Log Odds Linear Plot")


# In[26]:


# Name the new DataFrame, and specify all the columns for BLR.
nec_cols = nec_cols = os_data_X.columns

# Set the independent variable.
X = os_data_X[nec_cols]  

# Set the dependent variable.
y = os_data_y['Target']  

# Set the logit() to accept y and X as parameters, and return the logit object.
logit_model=sm.Logit(y, X)

# Indicate result = logit_model.fit() function.
result = logit_model.fit()  

# Print the results.
result.summary()


# In[27]:


# Import the necessary packages.
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Set LogisticRegression() to logreg.
logreg = LogisticRegression(max_iter=1000) 

# Fit the X_train and y_train data sets to logreg. 
logreg.fit(os_data_X, os_data_y.values.ravel()) 


# In[28]:


# Determine BLR model's accuracy.
y_pred = logreg.predict(X_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print('Accuracy of logistic regression classifier on test set: {:.2f}'      .format(acc))


# In[29]:


# Create the confusion matrix to test classification accuracy in BLR.
# Import the necessary package to create the confusion matrix. 
from sklearn.metrics import confusion_matrix  

# Indicate the confusion matrix needs to be created.
confusion_matrix = confusion_matrix(y_test, y_pred)  

# Plot the confusion_matrix.
sns.heatmap(confusion_matrix, annot=True, fmt='g')


# In[30]:


# Create a DataFrame to display the confusion matrix. 
pd.DataFrame(confusion_matrix, index=['observed_notchurn','observed_churn'],
columns=['predicted_notchurn', 'predicted_churn'])


# In[31]:


# Import the necessary package.
from sklearn.metrics import classification_report  

# Print a report on the model's accuracy.
print(classification_report(y_test, y_pred))  


# In[ ]:




