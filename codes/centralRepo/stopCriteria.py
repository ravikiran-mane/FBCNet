#!/usr/bin/env python
# coding: utf-8
"""
A stop criteria class to compose any training termination condition. 
@author: Ravikiran Mane
"""

import sys

current_module = sys.modules[__name__]

class composeStopCriteria(object): 
    '''
    The parent class for all the stop criteria. 
    This takes a structure as an input and gives out the stopCriteria object. 
    
    Input: 
        criteria : criteria structure 
        variables: a structure containing all the runtime variables
    Output: true -> stop now
            False -> dont stop
    '''
    def __init__(self, c):
        self.c = current_module.__dict__[list(c.keys())[0]](**c[list(c.keys())[0]])
    
    def __call__(self, variables):
         return self.c(variables);

class And(object):
    '''
    performs the and operation on two stop criteria.
    
    Input:
        c1 : dictionary describing first criteria,
        c2 : dictionary describing second criteria.
    if you wish to do and on multiple cases then do like: And(And(A, B), C)...
    '''
    def __init__(self, c1, c2):
        self.c1 = current_module.__dict__[list(c1.keys())[0]](**c1[list(c1.keys())[0]])
        self.c2 = current_module.__dict__[list(c2.keys())[0]](**c2[list(c2.keys())[0]])
        
    def __call__(self, variables):
        return self.c1(variables) and self.c2(variables)

class Or(object):
    '''
    performs the or operation on two stop criteria.
    
    Input:
        c1 : dictionary describing first criteria,
        c2 : dictionary describing second criteria.
    if you wish to do or on multiple cases then do like: Or(Or(A, B), C)...
    '''
    def __init__(self, c1, c2):
        self.c1 = current_module.__dict__[list(c1.keys())[0]](**c1[list(c1.keys())[0]])
        self.c2 = current_module.__dict__[list(c2.keys())[0]](**c2[list(c2.keys())[0]])
        
    def __call__(self, variables):
        return self.c1(variables) or self.c2(variables)
    
class MaxEpoch(object):
    '''
        Stop on reaching max epoch. 
        init arguments: 
        maxEpochs = maximum epochs to watch. 
        varName = 'variable name to compare with in the variables dictionary'
    '''
    def __init__(self, maxEpochs, varName):
        self.maxEpochs = maxEpochs
        self.varName = varName
        
    def __call__(self, variables):
        return variables[self.varName] >= self.maxEpochs

class NoDecrease(object):
    '''
        Stop on no decrease of a particular variable. 
        init arguments: 
        numEpochs = number of epochs to wait while there is no decrease in the value.  
        varName = 'variable name to compare with in the variables dictionary'
        minChange = minimum relative decrease which resets the value. default: 1e-6
    '''
    def __init__(self, numEpochs, varName, minChange = 1e-6):
        self.numEpochs = numEpochs
        self.varName = varName
        self.minChange = minChange
        self.minValue = float("inf")
        self.currentEpoch = 0
        
    def __call__(self, variables):
                
        if variables[self.varName] <= (1 - self.minChange)* self.minValue:
            self.minValue= variables[self.varName]
            variables[self.varName+'Min'] = self.minValue
            self.currentEpoch = 0
        else:
            self.currentEpoch += 1
            
        return self.currentEpoch >= self.numEpochs

class LessThan(object):
    '''
        Stop when value of var name becomes less than given threshold. 
        init arguments: 
        minValue = minimum value to watch. 
        varName = 'variable name to compare with int the variables dictionary'
    '''
    def __init__(self, minValue, varName):
        self.minValue = minValue
        self.varName = varName
        
    def __call__(self, variables):
        return variables[self.varName] <= self.minValue