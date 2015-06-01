# -*- coding: utf-8 -*-
"""
@author: Daniel
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np 
import pandas as pd
import statsmodels.api as sm

from ggplot import *
import matplotlib.pyplot as plt


file1 = 'C:\Users\Daniel\Downloads\improved-dataset\\turnstile_weather_v2.csv'
file2 = 'C:\Users\Daniel\Downloads\\turnstile_data_master_with_weather.csv'
turnstile_weather = pd.read_csv(file1)



### Data Wangling


### Data Visualization


### Data Analyzing

def linear_regression(features, values):
    """
    Perform linear regression given a data set with an arbitrary number of features.
    
    This can be the same code as in the lesson #3 exercise.
    """
    
    #features = sm.add_constant(features) 
    n = len(features[:,0])
    ones = np.ones((n,1))
    features = np.hstack((ones, features))
    
    model = sm.OLS(values, features)
    results = model.fit()
    intercept = results.params[0]
    params = results.params[1:len(results.params)]
       
    return intercept, params

def predictions(features, params, intercept):
    '''
    The NYC turnstile data is stored in a pandas dataframe called weather_turnstile.
    Using the information stored in the dataframe, let's predict the ridership of
    the NYC subway using linear regression with gradient descent.
    
    You can download the complete turnstile weather dataframe here:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv    
    
    Your prediction should have a R^2 value of 0.40 or better.
    You need to experiment using various input features contained in the dataframe. 
    We recommend that you don't use the EXITSn_hourly feature as an input to the 
    linear model because we cannot use it as a predictor: we cannot use exits 
    counts as a way to predict entry counts. 
    
    Note: Due to the memory and CPU limitation of our Amazon EC2 instance, we will
    give you a random subet (~10%) of the data contained in 
    turnstile_data_master_with_weather.csv. You are encouraged to experiment with 
    this exercise on your own computer, locally. If you do, you may want to complete Exercise
    8 using gradient descent, or limit your number of features to 10 or so, since ordinary
    least squares can be very slow for a large number of features.

    '''
    
    predictions = intercept + np.dot(features_array, params)
    return predictions
    
def compute_r_squared(data, predictions):
    r_squared = 1 - (np.square(data - predictions)).sum() / (np.square(data - data.mean())).sum()
    return r_squared

##### prepare data
turnstile_weather_lessData = turnstile_weather[0:len(turnstile_weather['rain'])]
# Select Features (try different features!)
features = turnstile_weather_lessData[['rain', 'fog', \
        'meanprecipi', 'pressurei', 'meantempi', 'wspdi', \
        'day_week', 'weekday']]
# Add UNIT to features using dummy variables. Why?
dummy_units = pd.get_dummies(turnstile_weather_lessData['UNIT'], prefix='unit')
features = features.join(dummy_units)

dummy_units = pd.get_dummies(turnstile_weather_lessData['conds'], prefix='conds')
features = features.join(dummy_units)

features_array = features.values
values = turnstile_weather_lessData['ENTRIESn']
values_array = values.values

##### Perform linear regression
intercept, params = linear_regression(features_array, values_array)
    
##### do prediction
predictions = predictions(features, params, intercept)
print compute_r_squared(values, predictions)

print intercept
print params[0:9]

import scipy
import matplotlib.pyplot as plt

def plot_residuals(values, predictions):
    plt.figure()
    (predictions - values).hist(bins=1000)
    plt.xlabel('Residuals')
    plt.ylabel('Freq.')
    return plt
plot_residuals(values, predictions)
print (abs(predictions - values)).describe()