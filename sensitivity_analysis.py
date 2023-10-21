# Sensitivity scenario and what-if analysis combined

# You’ve created a retirement model for the wealth management company and plotted the results in a DataFrame, let’s see how the model responds to different outputs and scenarios to solve a specific business challenge. In this demonstration, we will combine a sensitivity scenario and a what-if analysis.

# # 

# # Sensitivity scenario

# ## Prepare your workstation

# > The code snippets and data are based on the previous model.

# ### Import the necessary libraries.

# In[2]:


# Define classes to contain and encapsulate data.
from dataclasses import dataclass 
import pandas as pd
# Import in-built module for generating random numbers. 
import random 
# Display output inline.
get_ipython().run_line_magic('matplotlib', 'inline')
# Import to replicate a nested loop over the input values.
from sensitivity import SensitivityAnalyzer 

# Import warnings and disable.
import warnings
warnings.filterwarnings('ignore')


# ### Specify the inputs

# In[3]:


# Create a DataFrame consisting of various classes using Python's 'dataclass()'
# module and Object Oriented Programming (OPP).
@dataclass

class ModelInputs: 
    # Define the class and specify the default inputs. 
    starting_salary: int = 30000
    promos_every_n_years: int = 3
    cost_of_living_raise: float = 0.025
    promo_raise: float = 0.15
    savings_rate: float = 0.20
    interest_rate: float = 0.07
    desired_cash: int = 1000000

# Create an instance of the new class with the default inputs.
model_data = ModelInputs() 

# Print the results.
model_data 


# ### Calculate wage

# In[4]:


# Get the wage at a given year from the start of the model based 
# on the cost of living raises and regular promotions.
def wages_year(data: ModelInputs, year):
    # Every n years we have a promotion, so dividing the years and
    # taking out the decimals gets the number of promotions.
    num_promos = int(year / data.promos_every_n_years)  
    
   # This is the formula above implemented in Python.
    salary_t = data.starting_salary * (1 + data.cost_of_living_raise)    ** year * (1 + data.promo_raise) ** num_promos
    return salary_t

# Show the first four salaries in the range and 
# print the results using the f-string.
for i in range(4):
    year = i + 1
    salary = wages_year(model_data, year)
    print(f'The wage at year {year} is £{salary:,.0f}.')


# ### Calculate wealth

# In[5]:


# Calculate the cash saved within a given year by first 
# calculating the salary at that year then applying the savings rate.
def cash_saved_during_year(data: ModelInputs, year):
    salary = wages_year(data, year)
    cash_saved = salary * data.savings_rate
    return cash_saved

# Calculate the accumulated wealth for a given year based
# on previous wealth, the investment rate, and cash saved during the year.
def wealth_year(data: ModelInputs, year, prior_wealth):
                cash_saved = cash_saved_during_year(data, year)
                wealth = prior_wealth * (1 + data.interest_rate) + cash_saved
                return wealth

# Start with no cash saved.
prior_wealth = 0  
for i in range(4):
    year = i + 1
    wealth = wealth_year(model_data, year, prior_wealth)
    print(f'The wealth at year {year} is £{wealth:,.0f}.')
    
    # Set next year's prior wealth to this year's wealth:
    prior_wealth = wealth           


# ### Calculate years to retirement 

# In[6]:


def years_to_retirement(data: ModelInputs, print_output=True):
    # Start with no cash saved.
    prior_wealth = 0  
    wealth = 0
    # The ‘year’ becomes ‘1’ on the first loop.
    year = 0  
   
    if print_output:
        print('Wealth over time:')
    while wealth < data.desired_cash:
        year = year + 1
        wealth = wealth_year(data, year, prior_wealth)
        if print_output:
            print(f'The accumulated wealth at year {year} is £{wealth:,.0f}.')
            # Set next year's prior wealth to this year's wealth.
        prior_wealth = wealth  
       
    # Now we have run the while loop, the wealth must be >= desired_cash 
    # (whatever last year was set is the years to retirement), which we can print.
    if print_output:
        # \n makes a blank line in the output.
        print(f'\nRetirement:\nIt will take {year} years to retire.')  
    return year

years_to_retirement(model_data)


# # 

# ## 1. Defining functions for calculating the values for sensitivity analysis

# In[7]:


# Define the function that accepts the individual parameters.
# Note parameters are pecified sparatel.
def years_to_retirement_separate_args(
    # List the parameters and set their values.
    starting_salary=20000, 
    promos_every_n_years=5, 
    cost_of_living_raise=0.02,
    promo_raise= 0.15, 
    savings_rate=0.25, 
    interest_rate=0.05, 
    desired_cash=1000000): 
 
    # Update the values of the parameters:
    data = ModelInputs(
        starting_salary=starting_salary, 
        promos_every_n_years=promos_every_n_years, 
        cost_of_living_raise=cost_of_living_raise, 
        promo_raise=promo_raise, 
        savings_rate=savings_rate, 
        interest_rate=interest_rate, 
        desired_cash=desired_cash)
       
    return years_to_retirement(data, print_output=False)

# Call the function.
years_to_retirement_separate_args()


# In[ ]:





# ## 2. Generate random values for the input variables

# In[8]:


# Use Python's 'list comprehensions' syntax to make it easier to adjust the inputs. 
# Use 'i' as a temporary variable to store the value's position in the range:
sensitivity_values = {
    'starting_salary': [i * 10000 for i in range(2, 6)],
    'promos_every_n_years': [i for i in range(2, 6)],
    'cost_of_living_raise': [i/100 for i in range(1, 4)],
    'promo_raise': [i/100 for i in range(10, 25, 5)],
    'savings_rate': [i/100 for i in range(10, 50, 10)],
    'interest_rate': [i/100 for i in range(3, 8)],
    'desired_cash': [i * 100000 for i in range(10, 26, 5)]}


# In[ ]:





# ## 3. Running the sensitivity analyse module

# In[9]:


# Run the Python’s SensitivityAnalyzer with the all the assigned inputs:
sa = SensitivityAnalyzer(
    sensitivity_values,
    years_to_retirement_separate_args,
    result_name="Years to retirement",
    reverse_colors=True,
    grid_size=3)


# In[ ]:





# ## 4. Display the results

# In[10]:


# Display the results using a DataFrame.
styled_dict = sa.styled_dfs(num_fmt='{:.1f}') 


# In[ ]:





# # 

# # What-if analysis

# ### Specify the good and bad economies

# In[13]:



#  The function to calculate bad economy:
bad_economy_data = ModelInputs(
    starting_salary=10000,
    promos_every_n_years=8,
    cost_of_living_raise=0.01,
    promo_raise=0.07,
    savings_rate=0.15,
    interest_rate=0.03)

# The function for good economy:
good_economy_data = ModelInputs(
    starting_salary=40000,
    promos_every_n_years=2,
    cost_of_living_raise=0.03,
    promo_raise=0.20,
    savings_rate=0.35,
    interest_rate=0.06)

cases = {
    'Bad': bad_economy_data,
    # Original inputs were set to assume a 'normal' economy
    'Normal': model_data, 
    'Good': good_economy_data}


# In[14]:


# Run the model with the three scenarios and print the results:
for case_type, case_inputs in cases.items():
    ytr = years_to_retirement(case_inputs, print_output=False)
    # How long to retire in a good economy?
    print(f"It would take {ytr} years to retire in a {case_type} economy.")


# # 

# # Assigning probabilities

# In[15]:


# These values are arbitrary and are only used for demonstration. 
case_probabilities = {
    'Bad': 0.2,
    'Normal': 0.5,
    'Good': 0.3}

# Run the model by taking the expected value over the three cases;
# print the results with a text string:
expected_ytr = 0
for case_type, case_inputs in cases.items():
    ytr = years_to_retirement(case_inputs, print_output=False)
    weighted_ytr = ytr * case_probabilities[case_type]
    expected_ytr += weighted_ytr
    
print(f"It would take {expected_ytr:.0f} years to retire given a {case_probabilities['Bad']:.0%} chance of a bad economy and {case_probabilities['Good']:.0%} chance of a good economy.")


# In[ ]:




