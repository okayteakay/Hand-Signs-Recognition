# Numerical-Hand-Signs-Recognition

The goal of building this algorithm is to facilitate communications from a speech-impaired person to someone who doesn't understand sign language.

Training set: 1080 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (180 pictures per number).
Test set: 120 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (20 pictures per number).

Data preprocessing includes creating Numpy arrays, reshaping, normalization & flattening before being fed to neural networks. (However, CNNs do not require the last two operations)


The code file contains 6 neural network models -
1) Model 1 - A simple classification neural network with many hidden layers and neurons.
2) Model 2 - A relatively smaller neural network than model 1 due to lesser hidden layers and neurons.
3) Model 3 - We tune the hyperparamters using keras tuner & 9 different combinations.
4) Model 4 - Introduced Regularization 
5) Model 5 - Introduced Batch Normalization
6) Model 6 - Final Convolutional Neural Network Model with maximum accuracy, minimum loss & least number of epochs.

This project tries to compare the results obtained through several models in an image classification problem. 
The intuiton was to understand how CNNs work way better than other models(despite of hyperparameter tuning, regularization or Batch Normalization).

Finally, test images are given to the CNN model to classifiy.
