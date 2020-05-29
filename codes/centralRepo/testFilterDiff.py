#!/usr/bin/env python
# coding: utf-8
"""
Created on Sun May  3 15:41:46 2020

@author: ravi
"""

import copy
import numpy as np
import scipy.signal as signal
import torch

from eegDataset import eegDataset
import matplotlib.pyplot as plt

def bandpassFilter( data, bandFiltCutF,  fs, filtAllowance=2, axis=1, filtType='filtfilt'):
        """
         Filter a signal using cheby2 iir zero phase filtering.

        Parameters
        ----------
        data: 2d/ 3d np array
            trial x channels x time
        bandFiltCutF: two element list containing the low and high cut off frequency in herts.
            if any value is specified as None then only one sided filtering will be performed
        fs: sampling frequency
        filtAllowance: transition bandwidth in herts
        filtType: string, available options are 'filtfilt' and 'filter'

        Returns
        -------
        dataOut: 2d/ 3d np array after filtering
            Data after applying bandpass filter.
        """
        aStop = 30 # stopband attenuation
        aPass = 3 # passband attenuation
        nFreq= fs/2 # niquest frequency
        
        if (bandFiltCutF[0] == 0 or bandFiltCutF[0] is None) and (bandFiltCutF[1] == None or bandFiltCutF[1] >= fs / 2.0):
            # no filter
            print("Not doing any filtering. Invalid cut-off specifications")
            return data
        
        elif bandFiltCutF[0] == 0 or bandFiltCutF[0] is None:
            # low-pass filter
            print("Using lowpass filter since low cut hz is 0 or None")
            fPass =  bandFiltCutF[1]/ nFreq
            fStop =  (bandFiltCutF[1]+filtAllowance)/ nFreq
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'lowpass')
        
        elif (bandFiltCutF[1] is None) or (bandFiltCutF[1] == fs / 2.0):
            # high-pass filter
            print("Using highpass filter since high cut hz is None or nyquist freq")
            fPass =  bandFiltCutF[0]/ nFreq
            fStop =  (bandFiltCutF[0]-filtAllowance)/ nFreq
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'highpass')
        
        else:
            # band-pass filter
            print("Using bandpass filter")
            fPass =  (np.array(bandFiltCutF)/ nFreq).tolist()
            fStop =  [(bandFiltCutF[0]-filtAllowance)/ nFreq, (bandFiltCutF[1]+filtAllowance)/ nFreq]
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'bandpass')

        if filtType == 'filtfilt':
            dataOut = signal.filtfilt(b, a, data, axis=axis, padlen = 3 * (max(len(a), len(b))-1))
        else:
            dataOut = signal.lfilter(b, a, data, axis=axis)
        return dataOut


dPath = '/home/ravi/FBCNetToolbox/data/bci42a/'

dataPython = eegDataset(dPath+'multiviewPython3', dPath+'multiviewPython3/dataLabels.csv')
dataMat = eegDataset(dPath+'csvFilterBand', dPath+'csvFilterBand/dataLabels.csv')
# dataMat = eegDataset(dPath+'multiviewPython', dPath+'multiviewPython/dataLabels.csv')

#%%
i= 1
j =10
band = 5

plt.figure()
plt.plot(range(1000), dataPython[i]['data'][j,:,band], label="python")
plt.plot(range(1000), dataMat[i]['data'][j,:,band], label="matlab")
plt.legend(loc='upper right')
plt.show()

# psd plots
f, psdPy = signal.welch(dataPython[i]['data'][j,:,band], 250)
f, psdMat = signal.welch(dataMat[i]['data'][j,:,band], 250) 

plt.figure()
plt.semilogy(f, psdPy, label="python")
plt.semilogy(f, psdMat, label="mat")
plt.legend(loc='upper right')
plt.show()