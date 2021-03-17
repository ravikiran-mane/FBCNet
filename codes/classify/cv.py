#!/usr/bin/env python
# coding: utf-8
"""

10 fold cross-validation classification analysis of BCI Comp IV-2a and Korea datasets
@author: Ravikiran Mane

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
from statistics import mean

masterPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(masterPath, 'centralRepo'))
from eegDataset import eegDataset
from baseModel import baseModel
import networks
import transforms
from saveData import fetchData


# reporting settings
debug = False

def cv(datasetId = None, network = None, nGPU = None, subTorun=None):

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
    config['cv'] = 'subSpecific-Kfold'
    config['kFold'] = 10
    config['data'] = 'raw'
    config['subTorun'] = subTorun

    # CV fold details.
    # These files have been written to achieve consistent division of trials in 
    # fold across all methods.
    # For random division set config['loadCVFold'] to False
    config['loadCVFold'] = True
    config['pathCVFold'] = {0: 'CVIdx-subSpec-bci42a-seq.json',
                            1: 'CVIdx-subSpec-korea-seq.json'}

    # network initialization details:
    config['loadNetInitState'] = True;
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
    
    # give ful path to the pathCVFold
    for key, val in config['pathCVFold'].items():
        config['pathCVFold'][key] = os.path.join(masterPath,'cvFiles', val)
    config['pathCVFold']  = config['pathCVFold'][datasetId]
    if (not os.path.exists(config['pathCVFold'])) or config['kFold'] != 10:
        config['loadCVFold'] = False # cv fold divisions are only provided for 10-fold cv.
    
    # Output folder:
    # Lets store all the outputs of the given run in folder.
    config['outPath'] = os.path.join(toolboxPath, 'output')
    config['outPath'] = os.path.join(config['outPath'], datasets[datasetId], 'cv')

    # Network initialization:
    config['pathNetInitState'] = os.path.join(masterPath, 'netInitModels', config['pathNetInitState']+'.pth')
    # check if the file exists else raise a flag
    config['netInitStateExists'] = os.path.isfile(config['pathNetInitState'])
    
    #%% Some functions that should be defined here
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

    def excelAddData(worksheet, startCell, data, isNpData = False):
        '''
            Write the given max 2D data to a given given worksheet starting from the start-cell.
            List will be treated as a row.
            List of list will be treated in a matrix formate with inner list constituting a row.
            will return the modified worksheet which needs to be written to a file
            isNpData flag indicate whether the incoming data in the list is of np data-type
        '''
        #  Check the input type.
        if type(data) is not list:
            data = [[data]]
        elif type(data[0]) is not list:
            data = [data]
        else:
            data = data

        # write the data. starting from the given start cell.
        rowStart = startCell[0]
        colStart = startCell[1]

        for i, row in enumerate(data):
            for j, col in enumerate(row):
                if isNpData:
                    worksheet.write(rowStart+i, colStart+j, col.item())
                else:
                    worksheet.write(rowStart+i, colStart+j, col)

        return worksheet

    def dictToCsv(filePath, dictToWrite):
    	"""
    	Write a dictionary to a given csv file
    	"""
    	with open(filePath, 'w') as csv_file:
    		writer = csv.writer(csv_file)
    		for key, value in dictToWrite.items():
    			writer.writerow([key, value])

    def splitKfold(idx1, k, doShuffle = True):
        '''
        Split the index from given list in k random parts.
        Returns list with k sublists.
        '''
        idx = copy.deepcopy(idx1)
        lenFold = math.ceil(len(idx)/k)
        if doShuffle:
            np.random.shuffle(idx)
        return [idx[i*lenFold:i*lenFold+lenFold] for i in range(k)]

    def loadSplitFold(idx, path, subNo):
        '''
        Load the CV fold details saved in json formate.
        Returns list with k sublists corresponding to the k fold splitting.
        subNo is the number of the subject to load from. starts from 0
        '''
        import json
        with open(path) as json_file:
            data = json.load(json_file)
        data = data[subNo]
        # sort the values in sublists
        folds = []
        for i in list(set(data)):
            folds.append([idx[j] for (j, val) in enumerate(data) if val==i])

        return folds

    def generateBalancedFolds(idx, label, kFold = 5):
        '''
        Generate a class aware splitting of the data index in given number of folds.
        Returns list with k sublists corresponding to the k fold splitting.
        '''
        from sklearn.model_selection import StratifiedKFold
        folds = []
        skf = StratifiedKFold(n_splits=kFold)
        for train, test in skf.split(idx, label):
            folds.append([idx[i] for i in list(test)])
        return folds

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    #%%
    # based on current date and time -> always unique!
    randomFolder = str(time.strftime("%Y-%m-%d--%H-%M", time.localtime()))+ '-'+str(random.randint(1,1000))
    config['outPath'] = os.path.join(config['outPath'], randomFolder,'')
    # create the path
    if not os.path.exists(config['outPath']):
        os.makedirs(config['outPath'])
    print('Outputs will be saved in folder : ' + config['outPath'])

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
    
    # Select only the session 1 data because we will be doing cv on session 1 data.
    if len(data.labels[0])>4:
        idx = [i for i, x in enumerate(data.labels) if x[4] == '0']
        data.createPartialDataset(idx)
        
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


    #%% Let the training begin
    trainResults = []
    valResults = []
    testResults = []

    for i, sub in enumerate(subs):
        # for each subject
        if i not in config['subTorun']:
            continue

        start = time.time()

        # Run the cross-validation over all the folds
        subIdx = [i for i, x in enumerate(data.labels) if x[3] in sub]
        subY = [data.labels[i][2] for i in subIdx]

        if config['loadCVFold']:
            subIdxFold = loadSplitFold(subIdx, config['pathCVFold'], i)
        else:
            subIdxFold = generateBalancedFolds(subIdx, subY, config['kFold'] )

        trainResultsCV = []
        valResultsCV = []
        testResultsCV = []

        for j, folds in enumerate(subIdxFold):
            # for each fold:
            testIdx = folds
            rFolds = copy.deepcopy(subIdxFold)
            if j+1 < config['kFold']:
                valIdx = rFolds[j+1]
            else:
                valIdx = rFolds[0]
            rFolds.remove(testIdx)
            rFolds.remove(valIdx)
            trainIdx = [i for sl in rFolds for i in sl ]

            # separate the train, test and validation data
            testData = copy.deepcopy(data)
            testData.createPartialDataset(testIdx, loadNonLoadedData = True)
            trainData = copy.deepcopy(data)
            trainData.createPartialDataset(trainIdx, loadNonLoadedData = True)
            valData = copy.deepcopy(data)
            valData.createPartialDataset(valIdx, loadNonLoadedData = True)

            # Call the network
            net = network(**config['modelArguments'])
            net.load_state_dict(netInitState, strict=False)
            outPathSub = os.path.join(config['outPath'], 'sub'+ str(i), 'fold'+ str(j))
            model = baseModel(net=net, resultsSavePath=outPathSub, batchSize= config['batchSize'], nGPU = nGPU)
            model.train(trainData, valData, testData, **config['modelTrainArguments'])

            # extract the important results.
            trainResultsCV.append([d['results']['trainBest'] for d in model.expDetails])
            valResultsCV.append([d['results']['valBest'] for d in model.expDetails])
            testResultsCV.append([d['results']['test'] for d in model.expDetails])

        # Average the results. : This is only required for excel based reporting
        # You don't need this if you plan to just note the results from a terminal
        trainAccCV = [[r['acc'] for r in result] for result in trainResultsCV]
        trainAccCV = list(map(list, zip(*trainAccCV)))
        trainAccCV = [mean(data) for data in trainAccCV]
        valAccCV = [[r['acc'] for r in result] for result in valResultsCV]
        valAccCV = list(map(list, zip(*valAccCV)))
        valAccCV = [mean(data) for data in valAccCV]
        testAccCV = [[r['acc'] for r in result] for result in testResultsCV]
        testAccCV = list(map(list, zip(*testAccCV)))
        testAccCV = [mean(data) for data in testAccCV]

        # same for CM
        trainCmCV = [[r['cm'] for r in result] for result in trainResultsCV]
        trainCmCV = list(map(list, zip(*trainCmCV)))
        trainCmCV = [np.stack(tuple([cm for cm in cms]), axis = 2) for cms in trainCmCV]
        trainCmCV = [np.mean(data, axis =2) for data in trainCmCV]

        valCmCV = [[r['cm'] for r in result] for result in valResultsCV]
        valCmCV = list(map(list, zip(*valCmCV)))
        valCmCV = [np.stack(tuple([cm for cm in cms]), axis = 2) for cms in valCmCV]
        valCmCV = [np.mean(data, axis =2) for data in valCmCV]

        testCmCV = [[r['cm'] for r in result] for result in testResultsCV]
        testCmCV = list(map(list, zip(*testCmCV)))
        testCmCV = [np.stack(tuple([cm for cm in cms]), axis = 2) for cms in testCmCV]
        testCmCV = [np.mean(data, axis =2) for data in testCmCV]

        # Put everything back.
        temp1 = []
        temp2 = []
        temp3 = []
        for iTemp, trainAc in enumerate(trainAccCV):
            temp1.append({'acc': trainAc, 'cm': trainCmCV[iTemp]})
            temp2.append({'acc': valAccCV[iTemp], 'cm': valCmCV[iTemp]})
            temp3.append({'acc': testAccCV[iTemp], 'cm': testCmCV[iTemp]})

        # append to original results
        trainResults.append(temp1)
        valResults.append(temp2)
        testResults.append(temp3)

        # save the results
        results = {'train:' : trainResults[-1], 'val: ': valResults[-1], 'test': testResults[-1]}
        dictToCsv(os.path.join(outPathSub,'results.csv'), results)

        # Time taken
        print("Time taken = "+ str(time.time()-start))


    #%% Extract and write the results to excel file.
    # This is only required for excel based reporting
    # You don't need this if you plan to just note the results from a terminal

    # lets group the results for all the subjects using experiment.
    # the train, test and val accuracy and cm will be written
    trainAcc = [[r['acc'] for r in result] for result in trainResults]
    trainAcc = list(map(list, zip(*trainAcc)))
    valAcc = [[r['acc'] for r in result] for result in valResults]
    valAcc = list(map(list, zip(*valAcc)))
    testAcc = [[r['acc'] for r in result] for result in testResults]
    testAcc = list(map(list, zip(*testAcc)))

    print("Results sequence is train, val , test")
    print(trainAcc)
    print(valAcc)
    print(testAcc)

    # append the confusion matrix
    trainCm = [[r['cm'] for r in result] for result in trainResults]
    trainCm = list(map(list, zip(*trainCm)))
    trainCm = [np.concatenate(tuple([cm for cm in cms]), axis = 1) for cms in trainCm]

    valCm = [[r['cm'] for r in result] for result in valResults]
    valCm = list(map(list, zip(*valCm)))
    valCm = [np.concatenate(tuple([cm for cm in cms]), axis = 1) for cms in valCm]

    testCm = [[r['cm'] for r in result] for result in testResults]
    testCm = list(map(list, zip(*testCm)))
    testCm = [np.concatenate(tuple([cm for cm in cms]), axis = 1) for cms in testCm]

    print(trainCm)
    print(valCm)
    print(testCm)
    #%% Excel writing
    book = xlwt.Workbook(encoding="utf-8")
    for i, res in enumerate(trainAcc):
        sheet1 = book.add_sheet('exp-'+str(i+1), cell_overwrite_ok=True)
        sheet1 = excelAddData(sheet1, [0,0], ['SubId', 'trainAcc', 'valAcc', 'testAcc'])
        sheet1 = excelAddData(sheet1, [1,0], [[sub] for sub in subs])
        sheet1 = excelAddData(sheet1, [1,1], [[acc] for acc in trainAcc[i]], isNpData= True)
        sheet1 = excelAddData(sheet1, [1,2], [[acc] for acc in valAcc[i]], isNpData= True)
        sheet1 = excelAddData(sheet1, [1,3], [[acc] for acc in testAcc[i]], isNpData= True)

        # write the cm
        for isub, sub in enumerate(subs):
            sheet1 = excelAddData(sheet1, [len(trainAcc[0])+5,0+isub*len( config['modelTrainArguments']['classes'])], sub)
        sheet1 = excelAddData(sheet1, [len(trainAcc[0])+6,0], ['train CM:'])
        sheet1 = excelAddData(sheet1, [len(trainAcc[0])+7,0], trainCm[i].tolist(), isNpData= False)
        sheet1 = excelAddData(sheet1, [len(trainAcc[0])+11,0], ['val CM:'])
        sheet1 = excelAddData(sheet1, [len(trainAcc[0])+12,0], valCm[i].tolist(), isNpData= False)
        sheet1 = excelAddData(sheet1, [len(trainAcc[0])+17,0], ['test CM:'])
        sheet1 = excelAddData(sheet1, [len(trainAcc[0])+18,0], testCm[i].tolist(), isNpData= False)

    book.save(os.path.join(config['outPath'], 'results.xls'))

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
    
    cv(datasetId, network, nGPU, subTorun)
