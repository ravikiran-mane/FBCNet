#!/usr/bin/env python
# coding: utf-8
"""
Train model using the whole dataset and save the net weight.
@author: Yunji Zhang
"""
import numpy as np
import torch
import sys
import os
import time
import xlwt
import csv
import random
import math
import copy

masterPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(masterPath, 'centralRepo'))
from eegDataset import eegDataset
from baseModel import baseModel
import networks
import transforms
from saveData import fetchData

import warnings
warnings.filterwarnings("ignore")

# reporting settings
debug = False

def dictToCsv(filePath, dictToWrite):
    """
    Write a dictionary to a given csv file
    """
    with open(filePath, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dictToWrite.items():
            writer.writerow([key, value])

def config(datasetId = None, network = None, nGPU = None, subTorun=None):
    '''
    Define all the configurations in this function.
    -------
    params: datasetID (type: int): ID of dataset used to run, which is 0 or 1, default: 0.  
            network (type: str): Name of network used to run, default: "FBCNet".
            nGPU (type: int): Num of GPU used to run, default: 0, means use CPU.
            subTorun (type: int): ID of subject used to run, default: 0.
    -------
    return: config (type: dict): Config dictionary
            data (type: torch.Dataset): Dataset
            net (type: torch.nn): Initialized network model
    '''

    #%% Set the defaults use these to quickly run the network
    datasetId = datasetId or 0
    network = network or 'FBCNet'
    nGPU = nGPU or 0
    subTorun= subTorun or None
    selectiveSubs = False
    
    # decide which data to operate on:
    # datasetId ->  0:BCI-IV-2a data,    1: Korea data
    datasets = ['bci42a', 'korea']

    #%% Define all the model and training related options here.
    config = {}

    # Data load options:
    config['preloadData'] = False # whether to load the complete data in the memory

    # Random seed
    config['randSeed']  = 20190821
    
    # Network related details
    config['network'] = network
    config['batchSize'] = 16
    
    if datasetId == 1:
        config['modelArguments'] = {'nChan': 20, 'nTime': 1000, 'dropoutP': 0.5,
                                    'nBands':9, 'm' : 32, 'temporalLayer': 'LogVarLayer',
                                    'nClass': 2, 'doWeightNorm': True}
    elif datasetId == 0:
        config['modelArguments'] = {'nChan': 22, 'nTime': 1000, 'dropoutP': 0.5,
                                    'nBands':9, 'm' : 32, 'temporalLayer': 'LogVarLayer',
                                    'nClass': 4, 'doWeightNorm': True}
    
    # Training related details    
    config['modelTrainArguments'] = {'stopCondi':  {'c': {'Or': {'c1': {'MaxEpoch': {'maxEpochs': 1500, 'varName' : 'epoch'}},
                                                       'c2': {'NoDecrease': {'numEpochs' : 200, 'varName': 'valInacc'}} } }},
          'classes': [0,1], 'sampler' : 'RandomSampler', 'loadBestModel': True,
          'bestVarToCheck': 'valInacc', 'continueAfterEarlystop':True,'lr': 1e-3}
            
    if datasetId ==0:
        config['modelTrainArguments']['classes'] = [0,1,2,3] # 4 class data

    config['transformArguments'] = None

    # add some more run specific details.
    config['cv'] = 'trainTest'
    config['kFold'] = 1
    config['data'] = 'raw'
    config['subTorun'] = subTorun
    config['trainDataToUse'] = 1    # How much data to use for training
    config['validationSet'] = 0.2   # how much of the training data will be used a validation set

    # network initialization details:
    config['loadNetInitState'] = True
    config['pathNetInitState'] = config['network'] + '_'+ str(datasetId)

    #%% Define data path things here. Do it once and forget it!
    # Input data base folder:
    toolboxPath = os.path.dirname(masterPath)
    config['inDataPath'] = os.path.join(toolboxPath, 'data')
    
    # Input data datasetId folders
    if 'FBCNet' in config['network']:
        modeInFol = 'multiviewPython' # FBCNet uses multi-view data
    else:
        modeInFol = 'rawPython'

    # set final input location
    config['inDataPath'] = os.path.join(config['inDataPath'], datasets[datasetId], modeInFol)

    # Path to the input data labels file
    config['inLabelPath'] = os.path.join(config['inDataPath'], 'dataLabels.csv')

    # Output folder:
    # Lets store all the outputs of the given run in folder.
    config['outPath'] = os.path.join(toolboxPath, 'output')
    config['outPath'] = os.path.join(config['outPath'], datasets[datasetId]) # /FBCNet/output/bci42a

    # Network initialization:
    config['pathNetInitState'] = os.path.join(masterPath, 'netInitModels', config['pathNetInitState']+'.pth')
    # check if the file exists else raise a flag
    config['netInitStateExists'] = os.path.isfile(config['pathNetInitState'])

    # Path to save the trained model
    config['pathModel'] = os.path.join(masterPath, 'netModels')
    # check if the file exists else raise a flag
    config['netStateExists'] = os.path.isfile(config['pathModel'])

    #%% Some functions that should be defined hereve mode

    def setRandom(seed):
        '''
        Set all the random initializations with a given seed

        '''
        # Set np
        np.random.seed(seed)

        # Set torch
        torch.manual_seed(seed)

        # Set cudnn
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
                   
    #%% create output folder
    # based on current date and time -> always unique!
    randomFolder = str(time.strftime("%Y-%m-%d--%H-%M", time.localtime()))+ '-'+str(random.randint(1,1000))
    config['outPath'] = os.path.join(config['outPath'], randomFolder,'')
    config['resultsOutPath'] = os.path.join(config['outPath'], "Results")
    config['modelsOutPath'] = os.path.join(config['outPath'], config['network'])
    # create the path
    if not os.path.exists(config['outPath']):
        os.makedirs(config['outPath'])
    print('Outputs will be saved in folder : ' + config['outPath'])
    if not os.path.exists(config['resultsOutPath']):
        os.makedirs(config['resultsOutPath'])
    print('Results will be saved in folder : ' + config['resultsOutPath'])
    if not os.path.exists(config['modelsOutPath']):
        os.makedirs(config['modelsOutPath'])
    print('Models will be saved in folder : ' + config['modelsOutPath'])

    # Write the config dictionary
    dictToCsv(os.path.join(config['outPath'],'config.csv'), config)

    #%% Check and compose transforms
    if config['transformArguments'] is not None:
        if len(config['transformArguments']) >1 :
            transform = transforms.Compose([transforms.__dict__[key](**value) for key, value in config['transformArguments'].items()])
        else:
            transform = transforms.__dict__[list(config['transformArguments'].keys())[0]](**config['transformArguments'][list(config['transformArguments'].keys())[0]])
    else:
        transform = None

    #%% check and Load the data
    print('Data loading in progress')
    fetchData(os.path.dirname(config['inDataPath']), datasetId) # Make sure that all the required data is present!
    data = eegDataset(dataPath = config['inDataPath'], dataLabelsPath= config['inLabelPath'], preloadData = config['preloadData'], transform= transform)
    print('Data loading finished')

    #%% Check and load the model
    #import networks
    if config['network'] in networks.__dict__.keys():
        network = networks.__dict__[config['network']]
    else:
        raise AssertionError('No network named '+ config['network'] + ' is not defined in the networks.py file')

    # Load the net and print trainable parameters:
    net = network(**config['modelArguments'])
    print('Trainable Parameters in the network are: ' + str(count_parameters(net)))

    #%% check and load/save the the network initialization.
    if config['loadNetInitState']:
        if config['netInitStateExists']:
            netInitState = torch.load(config['pathNetInitState'])
        else:
            setRandom(config['randSeed'])
            net = network(**config['modelArguments'])
            netInitState = net.to('cpu').state_dict()
            torch.save(netInitState, config['pathNetInitState'])

    #%% Find all the subjects to run 
    subs = sorted(set([d[3] for d in data.labels]))
    nSub = len(subs)

    ## Set sub2run
    if selectiveSubs:
        config['subTorun'] = config['subTorun']
    else:
        if config['subTorun']:
            config['subTorun'] = list(range(config['subTorun'][0], config['subTorun'][1]))
        else:
            config['subTorun'] = list(range(nSub))


    # Call the network for training
    setRandom(config['randSeed'])
    net = network(**config['modelArguments'])
    net.load_state_dict(netInitState, strict=False)

    print("ALL CONFIG COMPLETED\n " + "*" * 30)
    return config, data, net

    

def train(config, data, initNet):
    '''
    Train model using the whole dataset and save the net weight.
    ------
    params: config (type: dict): Config dictionary
            data (type: torch.Dataset): Dataset
            initNet (type: torch.nn): Initialized network model
    '''


    #%% Let the training begin
    trainResults = []
    valResults = []
    
    start = time.time()
    
    trainData = copy.deepcopy(data)

    # print("")
    # print(testData[0]['data'].shape)
    
    # extract the desired amount of train data: 
    trainData.createPartialDataset(list(range(0, math.ceil(len(trainData)*config['trainDataToUse']))))

    # isolate the train and validation set
    valData = copy.deepcopy(trainData)
    valData.createPartialDataset(list( range( 
        math.ceil(len(trainData)*(1-config['validationSet'])) , len(trainData))))
    trainData.createPartialDataset(list(range(0, math.ceil(len(trainData)*(1-config['validationSet'])))))
    
    # D:\codes\BCI-VR\FBCNet\codes\netModels\2023-11-20--20-41-312\sub0\FBCNet_0
    model = baseModel(net=initNet, resultsSavePath=config['resultsOutPath'], modelSavePath=config['modelsOutPath'], batchSize= config['batchSize'], nGPU = nGPU)
    model.train(trainData, valData, **config['modelTrainArguments'])
    
    # extract the important results.
    trainResults.append([d['results']['trainBest'] for d in model.expDetails])
    valResults.append([d['results']['valBest'] for d in model.expDetails])
    
    # save the results
    results = {'train:' : trainResults[-1], 'val: ': valResults[-1]} 
    dictToCsv(os.path.join(config['resultsOutPath'],'results.csv'), results)
    
    # Time taken
    print("Time taken = "+ str(time.time()-start))

        
 


if __name__ == '__main__':

    arguments = sys.argv[1:]
    count = len(arguments)

    if count >0:
        datasetId = int(arguments[0])
    else:
        datasetId = None

    if count > 1:
        network = str(arguments[1])
    else:
        network = None

    if count >2:
        nGPU = int(arguments[2])
    else:
        nGPU = None

    if count >3:
        subTorun = [int(s) for s in str(arguments[3]).split(',')]

    else:
        subTorun = None
    
    config, data, net = config(datasetId, network, nGPU, subTorun)
    train(config, data, net)