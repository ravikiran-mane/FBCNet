#!/usr/bin/env python
# coding: utf-8
"""
A custom dataset to handle and load the epoched EEG files
@author: Ravikiran Mane
"""

from torch.utils.data import Dataset
import os
import pickle 
import csv


#%%
class eegDataset(Dataset):
    '''
   A custom dataset to handle and load the epoched EEG files. 
    
    This Dataset will load the EEG dataset saved in the following format
        1 file per trial in a dictionary formate with following fields: 
            id: unique key in 00001 formate
            data: a data matrix
            label: class of the data
            subject: subject number of the data
    
    At the initialization, it will first load the dataLabels.csv file 
    which contains the data labels and ids. Later the entire data can be
    loaded on the fly when it is necessary.
    
    The dataLabels file will be in the following formate:
        There will be one entry for every data file and will be stored as a 2D array and in csv file. 
        The column names are as follows:
            id, relativeFileName, label -> these should always be present. 
            Optional fields -> subject, session. -> they will be used in data sorting.
    
    Input Argument:
        dataPath : Path to the folder which contains all the data and dataLabels file.
        dataLabelsPath: absolute path to the dataLabels.csv.
        preloadData: bool -> whether to load the entire data or not. default: false
    
    For More Info, check this site:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    '''
    
    def __init__(self, dataPath, dataLabelsPath, transform = None, preloadData = False):
        '''
        Initialize EEG dataset

        Parameters
        ----------
        dataPath : str
            Path to the folder which contains all the data and dataLabels file.
        dataLabelsPath : str
            Path to the datalabels file.
        transform : transform, optional
            any transforms that will be applied on the data at loading. The default is None.
        preloadData : bool, optional
            whether to load the entire data in the memory or not. The default is False.

        Returns
        -------
        None.

        '''
        
        self.labels = []
        self.data = []
        self.dataPath = dataPath
        self.dataLabelsPath = dataLabelsPath
        self.preloadData = preloadData
        self.transform = transform
        
        # Load the labels file
        with open(self.dataLabelsPath, "r") as f:
            eegReader = csv.reader(f, delimiter = ',')
            for row in eegReader:
                self.labels.append(row)
            
            # remove the first header row
            del self.labels[0]
        
        # convert the labels to int
        for i, label in enumerate(self.labels):
            self.labels[i][2] = int(self.labels[i][2])
        
        # if preload data is true then load all the data and apply the transforms as well
        if self.preloadData:
            for i, trial in enumerate(self.labels):
                with open(os.path.join(self.dataPath,trial[1]), 'rb') as fp:
                    d = pickle.load(fp)
                    if self.transform:
                        d= self.transform(d)
                    self.data.append(d)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        '''Load and provide the data and label'''
        
        if self.preloadData:
            data  = self.data[idx]
        
        else:
            with open(os.path.join(self.dataPath,self.labels[idx][1]), 'rb') as fp:
                data = pickle.load(fp)
                if self.transform:
                    data = self.transform(data) 
                
        d = {'data': data['data'], 'label': data['label']}
        
        return d
    
    def createPartialDataset(self, idx, loadNonLoadedData = False):
        '''
        Create a partial dataset from the existing dataset.

        Parameters
        ----------
        idx : list
            The partial dataset will contain only the data at the indexes specified in the list idx.
        loadNonLoadedData : bool, optional
            Setting this flag will load the data in the memory 
            if the original dataset has not done it.
            The default is False.

        Returns
        -------
        None.

        '''
        self.labels = [self.labels[i] for i in idx]
        
        if self.preloadData:
            self.data  = [self.data[i] for i in idx]
        elif loadNonLoadedData:
            for i, trial in enumerate(self.labels):
                with open(os.path.join(self.dataPath,trial[1]), 'rb') as fp:
                    d = pickle.load(fp)
                    if self.transform:
                        d= self.transform(d)
                    self.data.append(d)
            self.preloadData = True
    
    def combineDataset(self, otherDataset, loadNonLoadedData = False):
        '''
        Combine two datasets which were generated from the same dataset by splitting.
        The possible use case for this function is to combine the train and validation set
        for continued training after early stop.
        
        This is a primitive function so please make sure that there is no overlap between the datasets.
        
        Parameters
        ----------
        otherDataset : eegdataset
            eegdataset to combine.
        loadNonLoadedData : bool, optional
            Setting this flag will load the data in the memory 
            if the original dataset has not done it.
            The default is False.

        Returns
        -------
        None.

        '''
        self.labels.extend(otherDataset.labels)
        if self.preloadData or loadNonLoadedData:
            self.data = []
            for i, trial in enumerate(self.labels):
                with open(os.path.join(self.dataPath,trial[1]), 'rb') as fp:
                    d = pickle.load(fp)
                    if self.transform:
                        d= self.transform(d)
                    self.data.append(d)
            self.preloadData = True
    
    def changeTransform(self, newTransform):
        '''
        Change the transform for the existing dataset. The data will be reloaded from the memory with different transform

        Parameters
        ----------
        newTransform : transform
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.transform = newTransform
        if self.preloadData:
            self.data = []
            for i, trial in enumerate(self.labels):
                with open(os.path.join(self.dataPath,trial[1]), 'rb') as fp:
                    d = pickle.load(fp)
                    if self.transform:
                        d= self.transform(d)
                    self.data.append(d)
            
        

