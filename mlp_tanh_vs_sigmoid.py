"""
Author:  Leigh Stauffer
File Name:  mlp_tanh_vs_sigmoid.py
Assignment 4

This code is borrowed from deeplearning.net/tutorial/code/logistic_sgd.py
Uses Theano to conduct deep learning with back propagation and gradient
descent.  This program compares the success rates of an mlp using the
tanh function and an mlp using the sigmoid function.
"""
from __future__ import print_function
__docformat__ = 'restructedtext en'
from mlp import MLP
from mlp import HiddenLayer
import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T


def load_data(dataset):
    """Loads the pickeled data."""
    
    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    return train_set, valid_set, test_set

def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    classifierMLP = pickle.load(open('best_mlp_model.pkl', 'rb'))
    classifierTanH = pickle.load(open('best_mlp_sigmoid.pkl', 'rb'))

    # compile a predictor function
    predict_model_MLP = theano.function(
        inputs=[classifierMLP.input],
        outputs=classifierMLP.logRegressionLayer.y_pred)

    predict_model_TanH = theano.function(
        inputs=[classifierTanH.input],
        outputs=classifierTanH.logRegressionLayer.y_pred)

    # We can test it on some examples from test test
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]  #Test set y is the target, test set x is 28X28 actual data
    
    predicted_values_MLP = predict_model_MLP(test_set_x)
    predicted_values_TanH = predict_model_TanH(test_set_x)

    hitMLP = 0
    hitTanH = 0
    total = 10000
    #confusion = np.zeros((10,10),"int")
    for k in range(10000):
        if test_set_y[k] == predicted_values_MLP[k]:
            hitMLP += 1
        if test_set_y[k] == predicted_values_TanH[k]:
            hitTanH += 1
            
    #print("Predicted values for the first 10 examples in test set:")
    #print(predicted_values)
    print("MLP with Sigmoid Success Rate:  " + str((hitTanH / total) * 100) + "%")
    print("MLP with TanH Success Rate:  " + str((hitMLP / total) * 100) + "%")


if __name__ == '__main__':
    #sgd_optimization_mnist()
    predict()
