"""
Author: Leigh Stauffer
Deep Learning - Assignment 3
File Name: backprop.py

This module represents a perceptron, a relatively simple learning machine.
This class contains __init__ and __str__ mehtods, used primarily for
testing.  This preceptron has hidden units and utilizes back propagation
when training.

Progress: Part 1
"""
import numpy as np
import pickle


class Perceptron():
    """A perceptron implementation."""
    

    #Constructor
    def __init__(self, inputN, outputM, h, weightScale = 1, test = False):
        """Initializes a perceptron with specified number of
        inputs (N) and outputs (M)."""
        self._N = inputN
        self._M = outputM
        self._hiddenUnits = h
        self._W_IH = np.random.randn(self._N + 1, self._hiddenUnits) * weightScale
        self._W_HO = np.random.randn(self._hiddenUnits + 1, self._M) * weightScale
            
        
    #String representation
    def __str__(self):
        """Returns a string representation of a perceptron."""
        return "A perceptron with "+str(self._N)+" input(s) and "+str(self._M)+" output(s)."
    

    def test(self, inputVector):
        """This method takes an input vector (numpy array) and appends a 1 to it
        for the bias term.  Then, it performs the dot product operation before
        comparing the resulting vector to 0.  Returns True of False."""
        Hnet = np.dot(np.append(inputVector, 1), self._W_IH)
        H = squash(Hnet)
        Onet = np.dot(np.append(H,1), self._W_HO)
        O = squash(Onet)
        return O
    

    def train(self, inputs, targets, niter = 10000, eta = 1):
        """This method is the training method.  It will begin adjusting the weights of
        a specific perceptron depending on the input patterns and the target patterns.
        Both the 'patterns' and 'targets' parameters should be matrices.
        NOTE: 'targets' should have as many rows as 'patterns'!!!"""
        for i in range(niter):
            weight_IH = np.zeros(((self._N + 1), self._hiddenUnits))
            weight_HO = np.zeros(((self._hiddenUnits + 1),self._M))
            for j in range(len(inputs)):
                Hnet = (np.dot(np.append(inputs[j], 1), self._W_IH))
                H = squash(Hnet)
                Onet = (np.dot(np.append(H,1), self._W_HO))
                O = squash(Onet)
                lil_dO = (targets[j] - O) * (O * (1 - O))
                lil_dH = (np.dot(lil_dO, self._W_HO.T)[:-1]) * (H * (1 - H))
                weight_IH += np.outer(np.append(inputs[j],1),lil_dH)
                weight_HO += np.outer(np.append(H,1), lil_dO)
            self._W_IH += (eta * weight_IH) / len(inputs)
            self._W_HO += (eta * weight_HO) / len(inputs)
        return self._W_IH, self._W_HO

    def save(self,filename):
        """This method saves the state of particular perceptron (most likely those that
        have already been trained)."""
        fileObj = open(filename, 'wb')
        pickle.dump(self, fileObj)
        fileObj.close()
        
    def load(self,filename):
        """This method loads a new set of weights to be trained or used."""
        fileObj = open(filename, 'rb')
        self = pickle.load(fileObj)
        return self

def squash(variable):
    """This function is used to smooth out the step function, making it differentiable."""
    return (1 / (1 + np.exp(-variable)))

        
