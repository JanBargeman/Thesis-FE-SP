#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 09:40:01 2021

@author: Jan
"""


# Python example - Fourier transform using numpy.fft method

import numpy as np
import matplotlib.pyplot as plotter
import pandas as pd


#%% plotting for understanding

data = dataYear
    
data_for_plot = data.balance   
# data_for_plot = data.amount 
fft = scipy.fft.fft(data_for_plot.values)/len(data_for_plot)
fft = fft[range(int(len(data)/2))]    

figure, axis = plotter.subplots(2, 1)
plotter.subplots_adjust(hspace=1)
axis[0].set_title('Balance data')
axis[0].plot(data.date, data_for_plot)
axis[0].set_xlabel('Time')
axis[0].set_ylabel('Amplitude')

samplingFrequency   = 365
tpCount     = len(data)
values      = np.arange(int(tpCount/2))
timePeriod  = tpCount/samplingFrequency
frequencies = values/timePeriod

axis[1].set_title('Fourier transform depicting the frequency components')
axis[1].plot(frequencies, abs(fft))
axis[1].set_xlabel('Frequency (1/y)')
axis[1].set_ylabel('Amplitude')


#%%
data = dataYear
    
# data_for_plot = data.balance   
data_for_plot = data.amount 
fft = scipy.fft.fft(data_for_plot.values)/len(data_for_plot)
fft = fft[range(int(len(data)/2))]    

figure, axis = plotter.subplots(2, 1)
plotter.subplots_adjust(hspace=1)
axis[0].set_title('Transaction data')
axis[0].plot(data.date, data_for_plot)
axis[0].set_xlabel('Time')
axis[0].set_ylabel('Amplitude')

samplingFrequency   = 365
tpCount     = len(data)
values      = np.arange(int(tpCount/2))
timePeriod  = tpCount/samplingFrequency
frequencies = values/timePeriod

axis[1].set_title('Fourier transform depicting the frequency components')
axis[1].plot(frequencies, abs(fft))
axis[1].set_xlabel('Frequency (1/y)')
axis[1].set_ylabel('Amplitude')


#%%

# How many time points are needed i,e., Sampling Frequency
samplingFrequency   = 365; #1       #365

# At what intervals time points are sampled
samplingInterval       = 1 / samplingFrequency; #1      #1/365
 
# Begin time period of the signals
beginTime           = 0; #0         #0
 
# End time period of the signals
endTime             = 1;  #365     #1

# Frequency of the signals
signal1Frequency     = 30;
signal2Frequency     = 7;

 
# Time points
time        = np.arange(beginTime, endTime, samplingInterval); #0-365   #0-1

amount = 10000

test = pd.DataFrame(time)
test['amount'] = 0
test.amount.loc[test.index % 7 == 0] = amount
test.amount.loc[(test.index+3) % 7 == 0] = -amount


data_for_test = data.amount + test.amount

fft = scipy.fft.fft(data_for_test.values)/len(data_for_test)
fft = fft[range(int(len(data)/2))]    

figure, axis = plotter.subplots(2, 1)
plotter.subplots_adjust(hspace=1)
axis[0].set_title('Transaction data')
axis[0].plot(data.date, data_for_test)
axis[0].set_xlabel('Time')
axis[0].set_ylabel('Amplitude')

samplingFrequency   = 365
tpCount     = len(data)
values      = np.arange(int(tpCount/2))
timePeriod  = tpCount/samplingFrequency
frequencies = values/timePeriod

axis[1].set_title('Fourier transform depicting the frequency components')
axis[1].plot(frequencies, abs(fft))
axis[1].set_xlabel('Frequency (1/y)')
axis[1].set_ylabel('Amplitude')