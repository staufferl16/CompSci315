"""
Author: Leigh Stauffer
Deep Learning - Assignment 2
File Name: classifier.py

This module uses both perceptron.py and pattern.py to discriminate
a hand-written 2 from not-a-handwritten-2.  Simply press F5 to see
results for training, testing, and stats for false-positives and
false-negatives (or "misses").
"""
import numpy as np
from perceptron import Perceptron
from pattern import loadPatterns
from pattern import printPattern


def main():
    """"This is the main method that will run all the tests and
    display the results."""
    
    target = np.zeros(2500)
    target[500:750] = 1
    #target = np.array(np.append(np.zeros(500),np.ones(250)))
    #target = np.array(np.append(target, np.zeros(1750)))
    #target = np.empty(2500)
    #for k in range(2500):
        #target[k] = ((k//250) == 2)
    #print(len(target))
    falsePositives = 0
    misses = 0
    print("Initialized Perceptron: \n")
    twoPerceptron = Perceptron(196,1,True)
    #print(len(target))
    twoMatrices, twoTargets = loadPatterns("digits_train.txt")
    twoTests, twoTestsTargets = loadPatterns("digits_test.txt")
    twoPerceptron.train(twoMatrices,target,1000)
    print("Expect True: " + str(twoPerceptron.test(twoTests[521])))
    for i in range(len(twoTests)):
        test = twoPerceptron.test(twoTests[i])
        if test == True and target[i] == 0:
            falsePositives += 1
        if test == False and target[i] == 1:
            misses += 1
    print("Misses: " + str(misses) + "/250")
    print("False Positives: " + str(falsePositives) + "/2250")
    
        
    

if __name__ == "__main__": main()
