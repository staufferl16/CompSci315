"""
Author: Leigh Stauffer
Deep Learning - Assignment 3
File Name: part1.py

This is the driver function indicating results for Part 1.
See backprop.py for more details on partial credit if results
are strange.
"""
import numpy as np
from backprop import Perceptron
from backprop import squash

def main():
    """This is the driver program for showing results from part 1."""
    
    inputArray = np.array([[0,0],[0,1],[1,0],[1,1]])
    targetXOR = np.array([[0],[1],[1],[0]])
    print("Initializing 1X3 Perceptron....")
    testP = Perceptron(2,1,3)
    print(str(testP) + "\n")
    print("\n"+"Testing learning XOR: \n")
    testP.train(inputArray, targetXOR)
    print("Training results:\n")
    print("Expect Close to 0: " + str(testP.test(np.array([0,0]))))
    print("Expect Close to 1: " + str(testP.test(np.array([0,1]))))
    print("Expect Close to 1: " + str(testP.test(np.array([1,0]))))
    print("Expect Close to 0: " + str(testP.test(np.array([1,1]))) + "\n")


if __name__ == "__main__": main()
