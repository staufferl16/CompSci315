"""
Author: Leigh Stauffer
Deep Learning - Assignment 2
File Name: pattern.py

This module creates useful numpy structures representing data from
digits_train.txt and digits_test.txt.
"""
import numpy as np


#class Pattern():
"""Creates useful numpy data structures.  Has only two methods."""

def getPattern(fileName,start,end):
    """Creates numpy structure representing data from
    digits_train.txt or digits_test.txt.  Returns two numpy
    structures (an array holding numpy matrices of each
    block). Start is index of the matrix to begin building
    the structure with, and end is the index on which to
    terminate teh build."""
    targetArray = []
    patternArray = []
    data =open(fileName, 'r').read().split('\nt')
    #print(data[1])
    for block in data[start:end]:    #When testing, use [0:4] accessor
        blockArray = []
        patterns = block.split('\n')
        for i in range(len(patterns)):
            patterns[i].strip()
        for line in patterns:
            if "-" in line:
                #print(line)
                pass
            else:
                values = line.split(' ')
                del values[-1]
                #print(values)
                if values == '\n':
                    pass
                #print(values)
                else:
                    for item in range(len(values)):
                        values[item].strip()
                    blockArray.append(values)
                """else:
                    del patterns[-1]
                    values = line.split(' ')
                    del values[-1]
                    #print(values)
                    if values == '\n':
                        pass
                    #print(values)
                    else:
                        blockArray.append(values)"""
                    
            #blockArray.append(patternArray)
        blockMatrix = np.array(blockArray[:-1]).astype('float')
        targetVector = np.array(blockArray[-1]).astype('int')
        patternArray.append(blockMatrix)
        targetArray.append(targetVector)
    return patternArray, targetArray
    #print(patternArray)
    #print(targetArray)


def visualize(dataMatrix):
    """Returns a visualization of a given number matrix."""
    display = ""
    for pattern in dataMatrix:
        for item in pattern[:-1]:
            #print(item)
            for element in item:
                if element.any() == None:
                    pass
                elif element.any() > 0.0:
                    char = "*"
                    display += char
                else:
                    char = " "
                    display += char
            display += "\n"
    return display

def loadPatterns(filename):
    """Use this method when loading all matrices, not just a few.
    This code given by Professor Levy for dealing with issues
    concerning the EOF character."""
    data = open(filename).read().split('\nt')
    #print(len(data))
    #print(data[0])
    inputs = np.zeros((2500, 196))
    targets = np.zeros((2500, 1))
    
    for block in range(len(data)):
        blocks = data[block].split('\n')[1:]
        lines = blocks[:15]
        pattern = np.fromstring(" ".join(lines[:-1]), sep = " ")
        inputs[block] = pattern
        targets[block] = block // 250
        #inputArray = np.append(inputArray, inputs)
        #print(inputs.shape)
        #print(inputs)
        #print(targets)
    return inputs, targets

def printPattern(inputs, targets):
    """Visualizes the digit pattern returned by loadPatterns()
    method. This code was also given by Professor Levy due to
    issues from EOF characters and formatting issues."""
    s = ' '
    #print(inputs)
    for k in range(196):
        if k%14 == 0:
            s += '\n'
        if inputs[k] > 0:
            s += '*'
        else:
            s += ' '
    return s

        

def test():
    """This is the main function for testing."""
    #getPattern("digits_train.txt")
    this, that = getPattern("digits_test.txt",2498,2499)
    print(this)
    print(that)
    print(visualize(this))
    exampleMatrix, exampleTarget = getPattern("digits_test.txt",893,894)
    print("\nExpect visualization of a three: ")
    print(visualize(exampleMatrix))
    #print("Number of Patterns: " + str(len(pattern)))
    #print("Number of Targets: " + str(len(target)))
    testMatrix, testTarget = loadPatterns("digits_train.txt")
    #print(testMatrix[123])
    print("Expect a 3:  \n")
    print(printPattern(testMatrix[897],testTarget[898]))

if __name__ == "__main__" : test()
