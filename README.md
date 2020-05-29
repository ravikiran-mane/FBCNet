# FBCNet
## FBCNet: An Efficient Multi-view Convolutional Neural Network for Brain-Computer Interface

This is the PyTorch implementation of the FBCNet architecture for EEG-BCI classification. 

# FBCNet: Architecture

![The FBCNet architecture](/FBCNet-V2.png)

FBCNet is designed with the aim of effectively extracting the spectro-spatial discriminative information which is the signature of EEG-MI while avoiding the problem of overfitting in the presence of small datasets. In its core, FBCNet architecture is composed of the following four stages: 


1. Multi-view data representation: The multi-view representation of the EEG data is obtained by spectrally filtering the raw EEG with multiple narrow-band filters. 
1. Spatial transformation learning: The spatial discriminative patterns for every view are then learned using a Depthwise Convolution layer. 
1. Temporal feature extraction: Following spatial transformation, a novel Variance layer is used to effectively extract the temporal information.
1. Classification: A fully connected (FC) layer finally classifies features from Variance layer into given classes.

The multi-view EEG representation followed by the spatial filtering allows extraction of spectro-spatial discriminative features and variance layer provides a compact representation of the temporal information.

## FBCNet: Toolbox

This repository is designed as a toolbox that provides all the necessary tools for training and testing of BCI classification networks. All the core functionalities are defined in the codes directory. The package requirements to run all the codes are provided in file req.text. The complete instructions for utilising this toolbox are provided in instructions.txt. 

The cv.py and ho.py in /codes/classify/ are the entry points to use this toolbox.

## FBCNet: Results
The classification results for FBCNet and other competing architectures are as follows:
![The FBCNet results](/results.png)

## Cite:
If you find this architecture or toolbox useful then please cite this paper:

Ravikiran Mane, Effie Chew, Karen Chua, Kai Keng Ang, Neethu Robinson, A.P. Vinod, Seong-Whan Lee, and Cuntai Guan, **"FBCNet: An Efficient Multi-view Convolutional Neural Network for Brain-Computer Interface,"** IEEE Transactions on Neural Networks and Learning Systems.  (Under Review)

## Acknowledgment
We thank Ding Yi for the assistance in code preparation. 
