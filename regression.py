"""
Author: Leigh Stauffer
Assignment 1

This program reads in data from a particular file and finds the least squares
solution to y = mx + b for both inputs.

"""
import math
import numpy as np
from io import StringIO


def findLLSx1(myArray):
    """
    This method finds the linear least square solution for x1 column
    from data.txt. Returns the m and b values for the best fitting
    line as strings. (y = mx + b)
    """
    meanX1 = sum(myArray["x1"]) / len(myArray["x1"])
    meanY = sum(myArray["y"]) / len(myArray["y"])
    top = sum((myArray["x1"] - meanX1) * (myArray["y"] - meanY))
    bottom = sum((myArray["x1"] - meanX1)* (myArray["x1"] - meanX1))
    slope = top/bottom
    yCoord = float(meanY - (slope*meanX1))
    answerX1 = "m = " + str(slope) + "\tb = " + str(yCoord)
    return answerX1

def findLLSx2(myArray):
    """
    This method finds the linear least square solution for x2 column
    from data.txt. Returns the m and b values for the best fitting
    line as strings. (y = mx + b)
    """
    meanX2 = sum(myArray["x2"]) / len(myArray["x2"])
    meanY = sum(myArray["y"]) / len(myArray["y"])
    top = sum((myArray["x2"] - meanX2) * (myArray["y"] - meanY))
    bottom = sum((myArray["x2"] - meanX2)* (myArray["x2"] - meanX2))
    slope = top/bottom
    yCoord = float(meanY - (slope*meanX2))
    answerX2 = "m = " + str(slope) + "\tb = " + str(yCoord)
    return answerX2

def solveFullLinearRegression(myArray):
    """
    This method finds the full linear regression y = w1x1 + w2x2 + b
    using np.linalg.lstsq.  Returns the w1, w2, and b values for the
    best fitting line as strings.
    """
    allX = np.vstack([myArray["x1"],myArray["x2"], np.ones(len(myArray["x1"]))]).T
    allY = myArray["y"]
    w1, w2, b = np.linalg.lstsq(allX, allY)[0]
    answer = "w1 = " + str(w1) + "\tw2 = " + str(w2) + "\tb = " + str(b)
    return answer

def classifyData(myArray):
    """
    Turns the data set from a regression problem into a classification problem by
    running each pair of points (x1 and x2) through the regression equation.  If
    the regression is greater than 0, this method returns a 0.  If the regression
    is less than or equal to 0, this method returns a 1.
    """
    allX = np.vstack([myArray["x1"],myArray["x2"], np.ones(len(myArray["x1"]))]).T
    allY = myArray["y"]
    w1, w2, b = np.linalg.lstsq(allX, allY)[0]
    classArray =((w1*myArray["x1"])+(w2*myArray["x2"])+b) > 0
    zArray = myArray["z"]
    zArray = zArray > 0
    answer = sum((zArray == classArray).astype('float')) / len(classArray)
    return "Success Rate = " + str(answer*100) + "%"

def trainFromData(myArray, sizeOfTraining):
    """Learning is about to happen..."""
    trainArray = myArray[0:sizeOfTraining]
    learnArray = myArray[sizeOfTraining:len(myArray)]
    trainX = np.vstack([trainArray["x1"],trainArray["x2"], np.ones(len(trainArray["x1"]))]).T
    trainY = trainArray["y"]
    w1, w2, b = np.linalg.lstsq(trainX, trainY)[0]
    classArray =((w1*learnArray["x1"])+(w2*learnArray["x2"])+b) > 0
    zArray = learnArray["z"]
    zArray = zArray > 0
    successRate = sum((zArray == classArray).astype('float')) / len(learnArray)
    return "Success Rate = " + str(successRate * 100) + "%"

def baselineTest(myArray):
    """Baseline Test that shows final report on percentage correct when
    w1 = w2 = b = 0."""
    w1 = w2 = b = 0
    classArray =((w1*myArray["x1"])+(w2*myArray["x2"])+b) > 0
    zArray = myArray["z"]
    zArray = zArray > 0
    successRate = sum((zArray == classArray).astype('float')) / len(myArray)
    return "Success Rate = " + str(successRate * 100) + "%"
    

def main():
    dataArray = np.genfromtxt("data.txt", names=True)
    print("Part 1, X1:  " + findLLSx1(dataArray) + "\n")
    print("Part 1, X2:  " + findLLSx2(dataArray) + "\n")
    print("Part 2:  " + solveFullLinearRegression(dataArray) + "\n")
    print("Part 3:  " + classifyData(dataArray) + "\n")
    print("Part 4, Train 25, Learn 75:  " + trainFromData(dataArray,25) + "\n")
    print("Part 4, Train 50, Learn 50:  " + trainFromData(dataArray,50) + "\n")
    print("Part 4, Train 75, Learn 25:  " + trainFromData(dataArray,75) + "\n")
    print("Part 4, Baseline Test:  " + baselineTest(dataArray) + "\n")

if __name__ == "__main__": main()


    
    



