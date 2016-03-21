"""
Author:  Leigh Stauffer
File Name:  mlp_confusion.py
Assignment 4

This code is borrowed from deeplearning.net/tutorial/code/logistic_sgd.py
Uses Theano to conduct deep learning with back propagation and gradient
descent.  This file constructs a confusion matrix for easier visualization
of results (at a glance).
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
    classifier = pickle.load(open('best_mlp_model.pkl', 'rb'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.logRegressionLayer.y_pred)

    # We can test it on some examples from test test
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]  #Test set y is the target, test set x is 28X28 actual data
    
    predicted_values = predict_model(test_set_x)
    
    confusion = np.zeros((10,10),"int")
    for k in range(10000):
        row = test_set_y[k]
        col = predicted_values[k]
        confusion[row,col] += 1
    
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)
    print(confusion)


if __name__ == '__main__':
    #sgd_optimization_mnist()
    predict()
