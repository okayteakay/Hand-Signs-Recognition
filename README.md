# Hand-Signs-Recognition

The goal of building this algorithm is to facilitate communications from a speech-impaired person to someone who doesn't understand sign language. We will be using two different datasets 

### 1) Numerical Hand Sign Dataset - To analyse the performance of different neural network models
### 2) Hand Gesture Dataset - To build a real time recogniton model using image processing techniques

### Descriptions

#### 1) Numerical Hand Sign Dataset (Fitting different models)

##### The dataset to this project is downloaded and read as h5 files, obtined from Andrew Ng's coursera course on Convolutional Neural Networks.
Training set: 1080 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (180 pictures per number).
Test set: 120 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (20 pictures per number).

##### Data preprocessing includes creating Numpy arrays, reshaping, normalization & flattening before being fed to neural networks. (However, CNNs do not require the last two operations)

##### The code file contains 6 neural network models -
###### a) Model 1 - A simple classification neural network with many hidden layers and neurons.
###### b) Model 2 - A relatively smaller neural network than model 1 due to lesser hidden layers and neurons.
###### c) Model 3 - We tune the hyperparamters using keras tuner & 9 different combinations.
###### d) Model 4 - Introduced Regularization 
###### e) Model 5 - Introduced Batch Normalization
###### f) Model 6 - Final Convolutional Neural Network Model with maximum accuracy, minimum loss & least number of epochs.

##### This project tries to compare the results obtained through several models in an image classification problem. 
The intuiton was to understand how CNNs work way better than other models(despite of hyperparameter tuning, regularization or Batch Normalization).

##### Finally, test images are given to the CNN model to classifiy.

#### 2) Real Time Hand Gesture Recogntion

##### Utilizes the Kaggle Dataset to build a simple Aritificial Neural Network Model which fits with over 99% accuracies on both training and test data. Real time gesture recognition using Morphological Transformations in OpenCV, Filering, Masking, Background Removal, Binary Thresholding, etc.
