# Building a BLR model

# 
# We’ve learned how to build a binary logistic regression (BLR) model and how to test its accuracy, which included checking key assumptions in logistic regression. Now it’s time for you to build a BLR model as a practical activity.
# Shen Lee, a senior manager at Westside Hospital in Australia, has to write an article on the prediction and diagnosis of breast cancer. She decides to build a BLR model as part of classification predictive analysis. 
# Shen has to compile a report for her manager by the end of the business day indicating the following:
# 
# - descriptive statistics of the data set
# - if the data set is balanced or not
# - if cancer could be predicted based on some kind of detection measurement
# - indicate the accuracy of the BLR model.

# ## 1. Prepare your workstation

# In[1]:


# Import all the necessary packages.
import numpy as np
import pandas as pd
import statsmodels.api as sm   
from statsmodels.stats.outliers_influence import variance_inflation_factor
import imblearn
from imblearn.over_sampling import SMOTE  
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

# Import the data set.
df = pd.read_csv('breast_cancer_data.csv',
                index_col='id')

# View the DataFrame.
print(df.info())
df.head()


# ## 2. .Explore the data

# In[2]:


# Determine if there are any null values.
df.isnull().sum()


# In[3]:


# Determine the descriptive statistics.
df.describe()


# ## 3. Manipulate the data

# In[4]:


# All values are null. We'll drop them.
df.drop(labels='Unnamed: 32', axis=1, inplace=True)

# View DataFrame.
df.head()


# In[5]:


# Determine if data set is balanced.
df['diagnosis'].value_counts()


# ## 4. Prepare the data

# In[6]:


# Set the target variable
y = df['diagnosis']


# In[7]:


# Set input features
X_data = df.drop('diagnosis', axis = 1)

# Normalise the data with the min-max feature scale.
X = (X_data -np.min(X_data))/(np.max(X_data)-np.min(X_data)).values

# View the values.
X.head()


# ## 5. Eliminate multicollinearity and balance the target variable

# In[8]:


# Import the necessary package.

# VIF dataframe
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
  
# calculating VIF for each feature
vif_data['VIF'] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
 
# View the DataFrame.
vif_data.sort_values('VIF',ascending=True)


# In[9]:


# Repeat VIF calculation using the five smallest values to avoid multicollinearity problems.
X2 = X[['symmetry_se', 'concavity_se', 'fractal_dimension_se', 'smoothness_se', 'texture_se']]

# VIF dataframe
vif_data = pd.DataFrame()
vif_data['feature'] = X2.columns

# calculating VIF for each feature
vif_data['VIF'] = [variance_inflation_factor(X2.values, i)
                          for i in range(len(X2.columns))]
 
# View the DataFrame.
vif_data.sort_values('VIF',ascending=True)


# While the VIF factors are still relatively high, they are at least below 10 and we will use X2 for the rest of the activity.

# In[10]:


# View the DataFrame.
print(X2.shape)
X2.head()


# In[11]:


# Apply SMOTE class as the target variable is not balanced.
os = SMOTE(random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X2, y,
                                                    test_size=0.3,
                                                    random_state=42)

# Specify the new data sets.
os_data_X, os_data_y = os.fit_resample(X_train, y_train)  

# Create two DataFrames for X and one for y:
os_data_X = pd.DataFrame(data = os_data_X, columns = X2.columns) 
os_data_y = pd.DataFrame(data = os_data_y, columns = ['diagnosis'])

# View DataFrame.
print(os_data_X.head())
os_data_y.head()


# In[12]:


# Determine if data set is balanced.
print("Original data: ")
print(df['diagnosis'].value_counts())
print("Balanced data:")
print(os_data_y.value_counts())


# ## 6. Create the BLR model

# In[13]:


# Import necessary libraries, packages, and classes.
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

# Create train and test data sets.
#X_train, X_test, y_train, y_test = train_test_split(X2, y,
#                                                    test_size=0.3,
#                                                    random_state=42)

# Specify and fit the model.
logreg_model = LogisticRegression()
#logreg_model.fit(X_train, y_train)
logreg_model.fit(os_data_X, os_data_y) # Using balanced training data


# ## 7. Calculate accuracy of the model

# In[14]:


# Calculate the predicted labels and predicted probabilities on the test set.

# Predict test class:
y_pred = logreg_model.predict(X_test)


# In[15]:


# Indicate the confusion matrix needs to be created.
confusion_matrix = confusion_matrix(y_test, y_pred)

confusion = pd.DataFrame(confusion_matrix, index=['is_healthy', 'is_cancer'],
                         columns=['predicted_healthy', 'predicted_cancer'])

# View the output.
confusion


# In[16]:


# import Seaborn library.
import seaborn as sns

# Plot the confusion_matrix.
sns.heatmap(confusion_matrix, annot=True, fmt='g')


# In[17]:


# Print the accuracy and classification report.
print(metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# While we have a reasonable model, there is significant room for improvement when using this dataset and it is a would be a good idea to consider alternatives.

# In[ ]:




