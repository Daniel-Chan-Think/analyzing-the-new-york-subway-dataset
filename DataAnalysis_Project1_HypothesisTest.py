# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file. 
"""

import pandas as pd
import numpy as np
#import pandasql
import scipy
import scipy.stats

from ggplot import *
import matplotlib.pyplot as plt


file1 = 'C:\Users\Daniel\Downloads\improved-dataset\\turnstile_weather_v2.csv'
file2 = 'C:\Users\Daniel\Downloads\\turnstile_data_master_with_weather.csv'
turnstile_weather_v2 = pd.read_csv(file1)
turnstile_noRain = turnstile_weather_v2[turnstile_weather_v2['rain'] == 0]
turnstile_rain = turnstile_weather_v2[turnstile_weather_v2['rain'] == 1]
entry_noRain = turnstile_noRain['ENTRIESn']
entry_rain = turnstile_rain['ENTRIESn']


### Data Wangling
# print turnstile_weather_v2.describe()
# print turnstile_weather_v2.head()
# print np.where(pd.isnull(turnstile_weather_v2))

### Data Visualization

plt.figure()
entry_noRain.plot(color='blue', kind='hist', bins=100, label='No Rain')
entry_rain.plot(color='green', kind='hist', bins=100, label='Rain')
#plt.axis([0,4000,0,20000])
#plt.axis([0,5e7,0,400])
plt.xlabel('ENTRIESn')
plt.ylabel('Freq.')
plt.title('Histogram of ENTRIESn')
plt.legend()


'''
print ggplot(turnstile_noRain, aes('Hour', 'ENTRIESn_hourly')) + \
    geom_point(color='blue') + geom_line() + \
    ggtitle('Histogram of ENTRIESn_hourly') + xlab('ENTRIESn_hourly') + ylab('Freq.') 
'''


### Data Analyzing
'''
From above figure, the samples do not follow normal distrubution, so T-test is not
applicable, and thus I decide to use Mann-Whitney U test which does not assume
normal distibution.
'''
entry_noRain_mean = np.mean(entry_noRain)
entry_rain_mean = np.mean(entry_rain)
print entry_noRain_mean, entry_rain_mean
U,p_oneTail = scipy.stats.mannwhitneyu(entry_rain.values, entry_noRain.values)
print U,p_oneTail
p_twoTail = p_oneTail * 2
print p_twoTail
alpha = 0.1
if p_twoTail < alpha:
    print "Reject the null hypothesis since the two-tail p-value < alpha=0.1"
else:
    print "Fail to reject the null hypothesis since the two-tail p-value >= alpha=0.1"



def plot_weather_data(turnstile_weather_v2):
    plot = ggplot(turnstile_weather_v2, aes('day_week', 'ENTRIESn')) + \
    geom_histogram() 
    #+ stat_smooth(color='red')
    return plot
print plot_weather_data(turnstile_weather_v2)