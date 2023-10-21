#Finding the best fit line

# **This is the solution to the activity.**
# 
# The best fit line is the line that best describes the relationship between variables in a linear regression model. In this activity, you will use Python to find the line of best fit using data from e-commerce companies. 
# 
# You’ve been hired by a venture capitalist firm to evaluate e-commerce investment opportunities. They have provided you with a data set of existing companies they see as potential future investments. They need you to predict the future median value of the business using simple linear regression analysis.
# 
# Using historical data, predict the median value of the seller's business based on the average number of items sold. 

# ## 1. Prepare your workstation

# In[1]:


# Import all the necessary packages.
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sklearn
import matplotlib.pyplot as plt

from sklearn import datasets 
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ## 2. Import the data set

# In[2]:


# Import the data set.
df = pd.read_csv('ecommerce_data.csv')

# View the DataFrame.
df.head()


# In[3]:


# Replace the missing values with 0.
df.fillna(0, inplace=True)

# Determine the number of missing values.
df.isna().sum()


# In[4]:


# View the metadata.
df.info()


# ## 3. Define the variables

# In[5]:


# Choose your variables.
X = df['avg_no_it'].values.reshape(-1, 1) 
y = df['Median_s'].values.reshape(-1, 1) 


# ## 4. Split the data set

# In[6]:


# Split the data into training = 0.7 and testing = 0.3 subsets.
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)


# ## 5. Run a linear equation

# In[7]:


# Run linear regression model.
lr = LinearRegression()

# Fit the model on the training data.
lr.fit(x_train, y_train)

# Predict is used for predicting on the x_test.
y_pred = lr.predict(x_test)

# View the output.
y_pred


# ## 6. Plot the regression

# In[8]:


# Visualise the training set.
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, lr.predict(x_train), color = 'green')
plt.title("Avg no of items vs Median of Seller Business(Training Data)")
plt.xlabel("Avg no of items")
plt.ylabel("Median of Seller Business")

plt.show()


# In[9]:


# Visualise the test set.
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, lr.predict(x_train), color = 'green')
plt.title("Avg no of items vs Median of Seller Business(Testing Data)")
plt.xlabel("Avg no of items")
plt.ylabel("Median of Seller Business")

plt.show()


# # 7. Print the values

# In[10]:


# Print the R-squared, intercept and coefficient value.
print("R-squared value: ", lr.score(x_test, y_test))

print("Intercept value: ", lr.intercept_)
print("Coefficient value: ", lr.coef_)


# #### Notes:
# - The R-squared tells us that the model is explaining 45.85% of the model.
# - The coefficient value of 9.12 tells us that as the `lowstat` variable increases by 1, the predicted value of `Median_s` increases by 9.12.

# In[ ]:




