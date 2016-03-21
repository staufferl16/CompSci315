"""
Author: Leigh Stauffer
Deep Learning - Assignment 2
File Name: perceptron.py

This module represents a perceptron, a relatively simple learning machine.
This class contains __init__ and __str__ mehtods, used primarily for
testing.  Other functionality will be defined in the pattern.py module.
"""
import numpy as np


class Perceptron():
    """A perceptron implementation."""
    

    #Constructor
    def __init__(self, inputN, outputM, test = False):
        """Initializes a perceptron with specified number of
        inputs (N) and outputs (M)."""
        self._N = inputN
        self._M = outputM
        #self._weights = (2 * np.random.random((self._N + 1, self._M)) - 1) / 20
        self._weights = 0.001 * np.random.randn(self._N + 1, self._M)
        if test == True:
            print(self._weights)
        self._trainingIter = 0
            
        
    #String representation
    def __str__(self):
        """Returns a string representation of a perceptron."""
        return "A perceptron with "+str(self._N)+" input(s) and "+str(self._M)+" output(s)."
    

    def test(self, inputVector):
        """This method takes an input vector (numpy array) and appends a 1 to it
        for the bias term.  Then, it performs the dot product operation before
        comparing the resulting vector to 0.  Returns True of False."""
        return np.dot(np.append(inputVector, 1),self._weights) > 0
    

    def train(self, inputs, targets, niter = 1000):
        """This method is the training method.  It will begin adjusting the weights of
        a specific perceptron depending on the input patterns and the target patterns.
        Both the 'patterns' and 'targets' parameters should be matrices.
        NOTE: 'targets' should have as many rows as 'patterns'!!!"""
        self._trainingIter = 0
        augmentedIn = np.column_stack((inputs,np.ones((len(inputs))))) #Input augmented with 1's (for bias)
        #print(augmentedIn)
        for i in range(1,niter):
            weightChanges = np.zeros((self._N + 1, self._M))
            self._trainingIter +=1
            print("Training Iteration: " + str(self._trainingIter) + "/" + str(niter))
            for j in range(len(augmentedIn)):
                Oj = (np.dot(augmentedIn[j], self._weights)>0)
                #print(Oj)
                Dj = targets[j] - Oj
                weightChanges += np.outer(augmentedIn[j], Dj)
            self._weights += (weightChanges/niter)
                #print(self._weights)

def main():
    """This main function simply prints out the testing for the AND and OR functions
    in order to display proper perceptron trianing and functionality."""
    inputArray = np.array([[0,0],[0,1],[1,0],[1,1]])
    targetAND = np.array([0,0,0,1])
    targetOR = np.array([0,1,1,1])
    print("Initializing 1X3 Perceptron....")
    testP = Perceptron(2,1,True)
    print("\n"+"Testing learning AND: ")
    testP.train(inputArray, targetAND)
    print("Weights after training: \n" + str(testP._weights))
    print("Expect False: " + str(testP.test(np.array([0,0]))))
    print("Expect False: " + str(testP.test(np.array([0,1]))))
    print("Expect False: " + str(testP.test(np.array([1,0]))))
    print("Expect True: " + str(testP.test(np.array([1,1]))) + "\n")
    print("Testing learning OR: ")
    testP.train(inputArray, targetOR)
    print("Weights after training: \n" + str(testP._weights))
    print("Expect False: " + str(testP.test(np.array([0,0]))))
    print("Expect True: " + str(testP.test(np.array([0,1]))))
    print("Expect True: " + str(testP.test(np.array([1,0]))))
    print("Expect True: " + str(testP.test(np.array([1,1]))) + "\n")

if __name__ == "__main__": main()
        
