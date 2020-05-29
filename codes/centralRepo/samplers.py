#!/usr/bin/env python
# coding: utf-8
"""
Custom samplers for the EEG data.
These samplers can be used along with the dataloaders to load the data in a
particular fashion.
Will return the object of type torch.utils.data.sampler
Generally will take an input of dataset of type eegDataset.
@author: Ravikiran Mane
"""
from torch.utils.data import Sampler
import random
import copy
import torch
import numpy as np
from numpy.random import RandomState
from collections import Counter


class InterleaveWithReplacementSampler(Sampler):
    """This sampler will present the element in a interleaved fashion for
    all the classes.
    It means that if the dataset has samples from n classes then the samples
    returned will belong to 0, 1st, 2nd, ... , nth class in order. This ensures
    balanced distribution of samples from  all the classes in every batch.

    If the number of samples from each class are not same, then the samples from
    classes with less samples will be reused again.
    The sampler will stop when all the samples from the class with maximum samples are used.

    Arguments:
        eegDataset: the dataset that will be used by the dataloader.
    """

    def __init__(self, eegDataset):
        self.eegDataset = eegDataset
        labels = [l[2] for l in self.eegDataset.labels]
        classes = list(set(labels))
        classIdx = [[i for i,x in enumerate(labels) if x == clas] for clas in classes]
        trash = [random.shuffle(classId) for classId in classIdx]

        # Now that class ids are shuffled start picking one from each of them.
        classN = [len(l) for l in classIdx]
        classIter = copy.deepcopy(classN)
        maxLen = max(classN)

        idxList = []
        for i in range(maxLen):
            for j in range(len(classes)):
                classIter[j] -= 1
                if classIter[j] < 0:
                    classIter[j] = classN[j]-1
                idxList.append(classIdx[j][classIter[j]])
        self.idxList = idxList

    def __iter__(self):
        return iter(self.idxList)

    def __len__(self):
        return len(self.idxList)


class RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return self.num_samples


class ClassBalancedSampler(Sampler):
    r"""Provides a mini-batch that maintains the original class probabilities. This is without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, doShuffle = False, seed = 31426):
        self.data_source = data_source
        self.seed = seed
        self.rng = RandomState(self.seed)

        # Get classes
        labels = [l[2] for l in self.data_source.labels]
        classes = list(set(labels))

        # calculate class proportions
        classN = Counter(labels)
        classProb = [classN[i] for i in classes]
        classProb = np.array(classProb)/sum(classProb)

        # create a new labels list which is organized in most balanced fashion
        # here what we will do is for every place in the new list, we will try all the classes
        # then the among all the combinations, whatever that results in least deviation from the
        # original probability distribution
        # This approach is computationally expensive but works...
        availableClasses = copy.deepcopy(classes)
        labelSeq = []
        for i in range(len(labels)):
            labelSeq.append(None)
            loss = float("inf")
            best = None
            for j in availableClasses:
                labelSeq[i] = j
                lossNow = self.probDiff(self.calculateProb(labelSeq, classes), classProb)
                if lossNow < loss:
                    loss = lossNow
                    best = j
            labelSeq[i] = best
            classN[best] -= 1
            if classN[best] == 0:
                availableClasses.remove(best)

        # Now lets put the trials at each label sequence
        # find the index of all the available trials with given classes
        labels = np.array(labels)
        classIdx = [np.argwhere(labels == clas) for clas in classes]

        # push the index instead of the label sequence.
        labelSeq = np.array(labelSeq)
        for i, clas in enumerate(classes):
            # shuffle if asked:
            x = classIdx[i]
            if doShuffle:
                np.random.shuffle(x)

            labelSeq[np.argwhere(labelSeq==clas)] = x
        # convert back to a list of elements.
        self.idxList = list(labelSeq)

    def calculateProb(self, elements, classes = None):
        '''
        Parameters
        ----------
        elements : list of elements to check probability in.
        classes : list of classes for which the probability is required.
                if none then unique values in elements in the ascending order will be used as a class.
        Returns
        -------
        prob: np.array with same size as that of classes. returns the probability of each class.

        '''
        classes = classes or list(set(elements))

        prob = np.zeros(np.size(classes))
        classC = Counter(elements)
        className = list(classC.keys())
        classN = np.array(list(classC.values()))
        classN = classN/sum(classN)
        for i, key in enumerate(className):
            if key in classes:
                prob[classes.index(key)] = classN[i]

        return prob

    def probDiff(self, probA, probB):
        '''
        Calculates the absolute difference between given class probabilities

        Parameters
        ----------
        probA : np.array
            probability of list A.
        probB : np.array
            probability of list B.

        Returns
        -------
        x -> float difference in the probabilities

        '''
        return sum(abs(probB-probA))

    def __iter__(self):
        return iter(self.idxList)

    def __len__(self):
        return len(self.idxList)