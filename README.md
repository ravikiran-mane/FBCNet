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
