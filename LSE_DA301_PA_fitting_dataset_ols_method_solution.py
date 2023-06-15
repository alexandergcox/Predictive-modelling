#!/usr/bin/env python
# coding: utf-8

# ### LSE Data Analytics Online Career Accelerator 
# # Course 301: Advanced Analytics for Organisational Impact

# ## Practical activity: Fitting a data set using the OLS method

# **This is the solution to the activity.**
# 
# In all regression models, the aim is to minimise residuals by finding the line of best fit. OLS is a common and straightforward way to estimate a linear regression model and to fit a linear equation to observed data. You’ll now have a chance to fit a data set using OLS in Python. 
# 
# An e-commerce store has asked you to analyse its online shopping experience. To help its marketing team optimise its campaign efforts to pre-existing users, the company wants to know how much money customers spend relative to how long they've been a member.
# 
# Remember, the marketing team needs to find a way to predict as accurately as possible how much money customers will spend, based on how long they have been a member of the shopping platform. Using the data set provided, you’ll need to decide on the independent and dependent variables and minimise residuals by finding the line of best fit. 

# ## 1. Prepare your workstation

# In[1]:


# Import the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm 
from statsmodels.formula.api import ols


# ## 2. Import the data set

# In[2]:


# Import the data set.
df_test = pd.read_csv('loyalty_club.csv')

# View the DataFrame.
df_test.head()


# ## 3. Define the variables

# In[3]:


# Dependent variable.
y = df_test["Yearly Amount Spent"] 

# Independent variable.
X = df_test["Length of Membership"]

# Check for linearity.
plt.scatter(X,y)


# ## 4. Run an OLS test

# In[4]:


# Run the OLS test.
f = 'y ~ X'
test = ols(f, data = df_test).fit()

# View the output.
test.summary()


# > What does the summary indicates?
# > - $R^2$: 65% of the total variability of `y` (money spent by customers), is explained by the variability of `X` (how long they have been a member).  
# > - F-stat: If the probability of F stat. is smaller than a threshold (usually 0.05), the set of variables of the regression model are significant, else, the regression is not good. For simple regression model, the `t`-statistic is equivalent.
# > - `X`: The coefficient of `X` describes the slope of the regression line, in other words, how much the response variable `y` change when `X` changes by 1 unit. In this activity, if the length that the customer has been a member (`X`) changes by 1 unit (please check units used) the money spent (`y`) will change by 64.2187 units. 
# >  - The `t`-value tests the hypothesis that the slope is significant or not. If the corresponding probability is small (typically smaller than 0.05) the slope is significant. In this case, the probability of the t-value is zero, thus the estimated slope is significant. 
# >   - The last two numbers describe the 95% confidence interval of the true xcoefficient, i.e. the true slope. For instance, if you take a different sample, the estimated slope will be slightly different. If you take 100 random samples each of 500 observations of `X` and `y`, then 95 out of the 100 samples will derive a slope that is within the interval (60.112 , 68.326).
# >   - In case of a multivariate regression model, each explanatory variable will have a separate row with the above information. So we will need to check which of the variables are significant, remove the ones that are not significant and then re-run the new regression model.

# ## 5. Create linear equation

# In[5]:


# x coef: 64.2187.
# Constant coef: 272.3998.
# Create the linear equation.
y_pred = 272.3998 + 64.2187 * X

# View the output.
y_pred


# ## 6. Plot the regression

# In[6]:


# Plot the data points.
plt.scatter(X, y)

# Plot the line.
plt.plot(X, y_pred, color='black')


# In[ ]:




