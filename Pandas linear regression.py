#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the necessary libraries..
# Creatre the linear regression.
import numpy as np
import pandas as pd

# Visualise the linear regression.
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Creatre a DataFrame to store the data.
# Pass the key value as a list.
data = {'Engine size' : [0.9, 1.0, 1.1, 1.2, 1.4,
                        1.6, 1.8, 2, 2.2, 2.4],
       'Selling Price': [20000, 22000, 23500, 26000,
                        25000, 28250, 29300, 33000,
                        34255, 45000]}

# Create a DataFrame
df = pd.DataFrame(data)

# Print the dataframe.
df


# In[3]:


# Apply pandas correlation method.
df.corr()


# In[4]:


# Start with a visualisation before running linear regression
sns.scatterplot(data=df,
               x = 'Engine size',
               y = 'Selling Price')

# Set axis values.
plt.ylim(0, 45000)
plt.xlim(0, 2.5)


# In[6]:


# Fit the linear model.
# Polyfit() good to use for simple linear regression (only one variable)
# See whether we can predict the selling price based on the engine size.
reg = np.polyfit(df['Engine size'],
                 df['Selling Price'],
                # Degree = 1, degree of polynomium, for SLR always 1.
                deg= 1)

#View the output
reg


# In[8]:


# Add a trendline to visualise the linear regression.
# Use the NumPy polyval method, and specify the regression and the independent variable.
trend = np.polyval(reg, df['Engine size'])

# View the previous scatterplot
sns.scatterplot(data=df,
               x = 'Engine size',
               y = 'Selling Price')

# Set axis values.
plt.ylim(0, 45000)
plt.xlim(0, 2.5)

# Add the trendline.
plt.plot(df['Engine size'],
        trend,
        color = 'red')


# In[12]:


# Install the statsmodel package.
get_ipython().system('pip install statsmodels')

# Import the necessary libraries.
import numpy as np
import pandas as pd

# Visualisation
import seaborn as sns
import matplotlib.pyplot as plt

# The statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols

#import and read the data file (car4u.csv)
cars = pd.read_csv('car4u.csv')

# View the DataFrame.
print(cars.head())
print(cars.info())
cars.describe()


# In[13]:


# Define the dependent variable.
y = cars['Selling price']

# Define the independent variable.
x = cars['Engine size']

# Check for linearity with matplotlib
plt.scatter(x,y)


# In[14]:


# Create formula and pass through OLD methods.

f = 'y ~ x'
test = ols(f, data= cars).fit()

# Print the regression table.
test.summary()


# In[15]:


# Extract the estimated parameter.
print("Parameters: ", test.params)

# Extract the standard errors.
print("Standard errors: ", test.bse)

# Extract the predicted values.
print("Predicted values: ", test.predict())


# In[16]:


# Create the linear regression model.
# Set the coefficient to 1.0143 and the constant to -0.4618.
y_pred = (-0.4618) + 1.0143 * cars['Engine size']

# View the output
y_pred


# In[17]:


# Plot the data points with a scatterplot.
plt.scatter(x,y)

# Plot the regression line (in black).
plt.plot(x, y_pred, color='black')

# Set the x and y limits on the axes.
plt.xlim(0)
plt.ylim(0)

# View the plot.
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




