"""
Author: Leigh Stauffer
Deep Learning - Assignment 3
File Name: part2.py

This is the driver function indicating results for Part 2.
See backprop.py for more details on partial credit if results
are strange.

**NOTE TO SELF:  Once you finish testing, make sure you change
the number of training iterations to the appropriate number!!!
"""
import numpy as np
from backprop import Perceptron
from backprop import squash
from pattern import loadPatterns
from pattern import printPattern

def main():
    """This is the driver program for showing results from part 1.
    Uncomment the code below if you need to pickle a new trained
    perceptron."""

    twoMatrices, twoTargets = loadPatterns("digits_train.txt")
    twoTests, twoTestsTargets = loadPatterns("digits_test.txt")
    target = np.zeros(2500)
    target[500:750] = 1
    falsePositives = 0
    misses = 0
    """
    trainEta = 1   #Change only these when tweaking for better results
    numberOfPatterns = 2000 #Chang only these when tweaking for better results
    print("Initialized Perceptron: \n")
    twoPerceptron = Perceptron(196,1,3,.1)
    twoPerceptron.train(twoMatrices,target,numberOfPatterns,trainEta)
    twoPerceptron.save()"""
    newPerceptron = Perceptron(196,1,3,.1)
    unPickled = newPerceptron.load('weights.dat')
    threshold = .23 #Change only these when tweaking for better results
    #print("Expect greater than " + str(threshold) +": " + str(unPickled.test(twoTests[521])))
    for i in range(len(twoTests)):
        test = unPickled.test(twoTests[i])
        if test >= threshold and target[i] == 0:
            falsePositives += 1
        if test < threshold and target[i] == 1:
            misses += 1
    print("Misses: " + str(misses) + "/250")
    print("False Positives: " + str(falsePositives) + "/2250")


if __name__ == "__main__": main()
