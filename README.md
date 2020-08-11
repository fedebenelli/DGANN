# DGANN
> Transformer Oil Dissolved Gas Diagnostic (DGA) assisted by neural networks

This respository consists in some scripts, a simple neural network module and a database of DGA provided by Duval

## `extract_data.py`
Extract the data from DGA done on faulted transformers, classifing them by type of fault, then saves it to a .csv file

## `NeuralNetwork.py`
Module that contains the Neural Network (NN), it consists in a simple backprop NN with a couple added modules using a RELU function.

## `train_nn.py`
Script used to train the Neural Network, after it ends training it shows two pictures "convergence.svg" and "precision.svg". Before it ends it saves the NN object to a file called `nn.obj`

The first one shows the loss value for each iteration

[](./convergence.svg)

The second one shows the correlation between the real data (y axis) and predicted data on the NN (x axis)

[](./precision.svg)

## `predict_nn.py`
Main program script that loads the `nn.obj` object and then it asks the user for data to load and returns a dictionary with the estimated probability for each fault.
