"""
Author: Leigh Stauffer
Deep Learning - Assignment 3
File Name: part3.py

This is the driver function indicating results for Part 3.
See this module and backprop.py for more details on partial
credit if results are strange.

**NOTE TO SELF:  Once you finish testing, make sure you change
the number of training iterations to the appropriate number!!!

**Current Progress:  Check that output is acceptable and clean
up any output necessary.
"""
import numpy as np
from backprop import Perceptron
from backprop import squash
from pattern import loadPatterns
from pattern import printPattern

def buildTarget():
    """This will build the 2500 X 10 matrix for training."""
    T = list()
    for p in range():
        pass
        

def main():
    """This is the driver program for showing results from part 1.
    Uncomment the code below if you need to pickle a new trained
    perceptron."""

    digitsMatrices, digitsTargets = loadPatterns("digits_train.txt")
    digitsTests, digitsTestTargets = loadPatterns("digits_test.txt")
    target = np.array(())
    for p in range(2500):
        arrayOfTen = np.array(())
        beginning = np.append(np.zeros(digitsTargets[p]),np.ones(1))
        arrayOfTen = np.append(beginning,np.zeros(9-digitsTargets[p]))
        target = np.append(target,arrayOfTen)
    target = np.reshape(target,(2500,10))
    """trainEta = 1   #Change only these when tweaking for better results
    numberOfPatterns = 10000 #Change only these when tweaking for better results
    print("Initialized Perceptron: \n")
    digitsPerceptron = Perceptron(196,10,3,.1)
    digitsPerceptron.train(digitsMatrices,target,numberOfPatterns,trainEta)
    digitsPerceptron.save("part3.dat")"""
    newPerceptron = Perceptron(196,10,3,.1)
    unPickled = newPerceptron.load("part3.dat")
    
    confusion = np.zeros((10,10),"int")
    for k in range(2500):
        row = k // 250
        col = np.argmax(unPickled.test(digitsTests[k]))
        confusion[row,col] += 1
    print(confusion)
    
    """threshold = .23 #Change only these when tweaking for better results
    #print("Expect greater than " + str(threshold) +": " + str(unPickled.test(twoTests[521])))
    for i in range(len(twoTests)):
        test = unPickled.test(twoTests[i])
        if test >= threshold and target[i] == 0:
            falsePositives += 1
        if test < threshold and target[i] == 1:
            misses += 1
    print("Misses: " + str(misses) + "/250")
    print("False Positives: " + str(falsePositives) + "/2250")"""


if __name__ == "__main__": main()

