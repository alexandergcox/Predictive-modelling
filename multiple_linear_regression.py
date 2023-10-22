# Conducting multiple linear regression using Python

# **This is the solution to the activity.**
# 
# In MLR you are adding another variable (or two or three or more!) to the calculation when you run your regression. Most likely, in the real world, you’ll have more than two variables to deal with, so MLR allows you to handle this and find predictive results that can help your business grow. This activity will build on the simple linear regression practical exercise from earlier, but this time, there will be another variable to work with. 
# 
# The main objective is to run multiple linear regression on three variables to predict future median business values. You’ll need to divide the data into training and testing subsets and use these to test the model with OLS. You’ll also check for multicollinearity and homoscedasticity. 

# ## 1. Prepare your workstation

# In[1]:


# Import all the necessary packages.
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
import sklearn
import matplotlib.pyplot as plt

from sklearn import linear_model
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols

import warnings  
warnings.filterwarnings('ignore')  


# ## 2. Import data set

# In[2]:


# Import the data set.
df_ecom = pd.read_csv('ecommerce_data.csv')

# View the DataFrame.
df_ecom.head()


# In[3]:


# View the metadata.
df_ecom.info()


# ## 3 Define variables

# In[4]:


# Define the dependent variable.
y = df_ecom['Median_s'] 

# Define the independent variable.
X = df_ecom[['avg_no_it', 'tax']] 


# In[5]:


# Specify the model.
multi = LinearRegression()  

# Fit the model.
multi.fit(X, y)


# In[6]:


# Call the predictions for X (array).
multi.predict(X)


# In[7]:


# Checking the value of R-squared, intercept and coefficients.
print("R-squared: ", multi.score(X, y))
print("Intercept: ", multi.intercept_)
print("Coefficients:")

list(zip(X, multi.coef_))


# In[8]:


# Make predictions.
New_Value1 = 5.75
New_Value2 = 15.2
print ('Predicted Value: \n', multi.predict([[New_Value1 ,New_Value2]]))  


# ## 4. Training and testing subsets with MLR

# In[9]:


# Create train and test data sets.
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)


# In[10]:


# Training the model using the 'statsmodel' OLS library.
# Fit the model with the added constant.
model = sm.OLS(y_train, sm.add_constant(x_train)).fit()

# Set the predicted response vector.
Y_pred = model.predict(sm.add_constant(x_test)) 

# Call a summary of the model.
print_model = model.summary()

# Print the summary.
print(print_model)  


# In[11]:


print(multi.score(x_train, y_train)*100)


# ## 5. Run a regression test

# In[12]:


# Run regression on the train subset.
mlr = LinearRegression()  

mlr.fit(x_train, y_train)


# In[13]:


# Call the predictions for X in the train set.
y_pred_mlr = mlr.predict(x_train)  

# Print the predictions.
print("Prediction for test set: {}".format(y_pred_mlr)) 


# In[14]:


# Print the R-squared value.
print(mlr.score(x_test, y_test)*100)  


# # 6. Check for multicollinearity

# In[15]:


# Check multicollinearity.
x_temp = sm.add_constant(x_train)

# Create an empty DataFrame. 
vif = pd.DataFrame()

# Calculate the VIF for each value.
vif['VIF Factor'] = [variance_inflation_factor(x_temp.values,
                                               i) for i in range(x_temp.values.shape[1])]

# Create the feature columns.
vif['features'] = x_temp.columns

# Print the values to one decimal points.
print(vif.round(1))


# In[16]:


# Determine heteroscedasticity.
model = sms.het_breuschpagan(model.resid, model.model.exog) 


# In[17]:


terms = ['LM stat', 'LM Test p-value', 'F-stat', 'F-test p-value']
print(dict(zip(terms, model)))


# `Note:` We always fit the model to train data and evaluate the performance of the model using the test data. We predict the test data and compare the predictions with actual test values.
# - rerun the model on the test data and jot down your observation.

# # 7. Evaluate the model

# In[18]:


# Call the metrics.mean_absolute_error function.  
print('Mean Absolute Error (Final):', metrics.mean_absolute_error(y_test, Y_pred))  

# Call the metrics.mean_squared_error function.
print('Mean Square Error (Final):', metrics.mean_squared_error(y_test, Y_pred))  


# In[ ]:




