#!/usr/bin/env python
# coding: utf-8

# ### LSE Data Analytics Online Career Accelerator 
# # Course 301: Advanced Analytics for Organisational Impact

# ## Practical activity: Conduct A/B testing in Python

# **This is the solution to the activity.**
# 
# An online bicycle store has changed its home page interface to encourage visitors to click through to its loyalty programme sign-up page. It is hoping the new interface will encourage more visitors to access the loyalty programme page, see what benefits the programme brings, and hopefully then sign up. The current click-through rate (CTR) sits at around 50% annually, and the company hopes the new design will push this to at least 55%. 
# 
# This analysis uses the `bike_shop.csv` data set. Using your Python and data wrangling skills, you will run an A/B test on the data to measure the significance of the interface change based on CTR to the loyalty programme page. 

# ## 1. Prepare your workstation

# In[1]:


# Import the necessary libraries.
import statsmodels.stats.api as sms
from statsmodels.stats.power import TTestIndPower


# ## 2. Perform power analysis

# In[2]:


# Perform the power analysis to determine sample size.
effect = sms.proportion_effectsize(0.50, 0.55)   
 
effect,
alpha = 0.05
power = 0.8

analysis = TTestIndPower()
result = analysis.solve_power(effect, power=power,
                              nobs1=None, ratio=1.0,
                              alpha=alpha)

print('Sample Size: %.3f' % result)


# ## 3. Import data set

# In[3]:


# Import the necessary libraries.
import pandas as pd
import math
import numpy as np
import statsmodels.stats.api as sms
import scipy.stats as st
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[5]:


# Read the data set with Pandas.
df = pd.read_csv('bike_shop.csv')

# Print the DataFrame.
print(df.shape)
df.head()


# In[6]:


# View the DataFrame.
df.info()


# ## 4. Clean the data

# In[7]:


# Rename the columns.
df_new = df.rename(columns={'IP Address': 'IPAddress',
                            'LoggedInFlag': 'LoyaltyPage'})

# View the DataFrame.
print(df_new.shape)
print(df_new.head())
df_new.info()


# In[8]:


# Drop duplicate values.
df_new.drop_duplicates(subset ='IPAddress',
                       keep = False,
                       inplace = True)


# Drop duplicate columns.
df_final = df_new.drop(['RecordID', 'VisitPageFlag'],
                       axis=1)


# View the DataFrame.
print(df_final.shape)
print(df_final.head())
df_final.info()


# ## 5. Subset the DataFrame

# In[9]:


# Split the data set into ID1 as treatment and ID2 & ID3 as control groups.
df_final['Group'] = df_final['ServerID'].map({1:'Treatment',
                                              2:'Control',
                                              3:'Control'})

# View the DataFrame.
print(df_final.shape)
df_final.head()


# In[10]:


# Count the values.
df_final['Group'].value_counts()


# In[11]:


# Create two DataFrames.
# You can use any random_state.
c_sample = df_final[df_final['Group'] == 'Control'].sample(n=1566,
                                                           random_state=42) 

t_sample = df_final[df_final['Group'] == 'Treatment'].sample(n=1566,
                                                             random_state=42)

# View the DataFrames.
print(c_sample.head())
t_sample.head()


# ## 6. Perform A/B testing

# In[12]:


# Perform A/B testing.
# Create variable and merge DataFrames.
ab_test = pd.concat([c_sample, t_sample], axis=0)

ab_test.reset_index(drop=True, inplace=True)

# View the output.
ab_test.head()


# In[13]:


# Calculate the conversion rates.
conversion_rates = ab_test.groupby('Group')['LoyaltyPage']


# Standard deviation of the proportion.
STD_p = lambda x: np.std(x, ddof=0)    
# Standard error of the proportion.
SE_p = lambda x: st.sem(x, ddof=0)     

conversion_rates = conversion_rates.agg([np.mean, STD_p, SE_p])

conversion_rates.columns = ['conversion_rate',
                            'std_deviation',
                            'std_error']

# Convert output into a Pandas DataFrame.
cr = pd.DataFrame(conversion_rates)

# View output.
cr


# In[14]:


from statsmodels.stats.proportion import proportions_ztest, proportion_confint

control_results = ab_test[ab_test['Group'] == 'Control']['LoyaltyPage']
treatment_results = ab_test[ab_test['Group'] == 'Treatment']['LoyaltyPage']

n_con = control_results.count()
n_treat = treatment_results.count()

successes = [control_results.sum(), treatment_results.sum()]

nobs = [n_con, n_treat]

z_stat, pval = proportions_ztest(successes, nobs=nobs)
(lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint(successes,
                                                                        nobs=nobs,
                                                                        alpha=0.05)

print(f'Z test stat: {z_stat:.2f}')
print(f'P-value: {pval:.3f}')
print(f'Confidence Interval of 95% for control group: [{lower_con:.3f}, {upper_con:.3f}]')
print(f'Confidence Interval of 95% for treatment group: [{lower_treat:.3f}, {upper_treat:.3f}]')


# ## 7. Summarise results and explain your answers

# The change to the homepage slightly decreased the click through to the login page. 
# 
# The `p`-value is smaller than the Alpha value of 0.05, meaning we reject the $H_0$. 

# In[ ]:




