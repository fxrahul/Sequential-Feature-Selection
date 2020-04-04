# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:11:57 2019

@author: Rahul
"""
#----------------------------------------------------Importing Packages---------------------------------------------
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from collections import OrderedDict
from operator import itemgetter
#------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------Importing CSV---------------------------------------------------------------
data = pd.read_csv('SFS.csv')
train = pd.DataFrame(data)
train = train.iloc[np.random.permutation(len(train))].reset_index(drop=True)
train_x = train.drop(train.columns[-1] , axis = 1)
train_y = train[train.columns[-1]]
#------------------------------------------------------------------------------------------------------------------
def checkStratification(UniqueValueOfClass,perCentageOfClasses):
    totalCountForTrain = 0
    totalCountForTest = 0
    for z in range(len(UniqueValueOfClass)):
        totalCountForTrain += perCentageOfClasses['train'][UniqueValueOfClass[z]]
        totalCountForTest += perCentageOfClasses['test'][UniqueValueOfClass[z]]
    
    for y in range(len(UniqueValueOfClass)):
        perCentageOfClasses['train'][UniqueValueOfClass[y]] /= totalCountForTrain
        perCentageOfClasses['test'][UniqueValueOfClass[y]] /= totalCountForTest

    flag = 0
    for z in range(len(UniqueValueOfClass)):
        testRangeForCheckingPlus = (perCentageOfClasses['test'][UniqueValueOfClass[z]]) + 0.1
        testRangeForCheckingMinus = (perCentageOfClasses['test'][UniqueValueOfClass[z]]) - 0.1
        if perCentageOfClasses['train'][UniqueValueOfClass[z]] == perCentageOfClasses['test'][UniqueValueOfClass[z]] or  perCentageOfClasses['train'][UniqueValueOfClass[z]] <= testRangeForCheckingPlus or perCentageOfClasses['train'][UniqueValueOfClass[y]] >= testRangeForCheckingMinus :
            flag = 1
        else:
            flag = 0
    return flag

def calculateAccuracy(actual,predicted):
    matchingValue = 0
    for i in range(len(actual)):
        if actual.iloc[i] == predicted[i]:
            matchingValue += 1
    return (matchingValue/len(actual))
    
def stratifiedKFold(data,noOfInstancesInEachFold,UniqueValueOfClass):
    clf = DecisionTreeClassifier(random_state=0)
    perCentageOfClasses = {}
    trainingFold = data[0:(len(data)-noOfInstancesInEachFold)]
    testingFold = data[(len(data)-noOfInstancesInEachFold):len(data)]
    perCentageOfClasses['train'] = {}
    perCentageOfClasses['test'] = {}
    for n in range(len(UniqueValueOfClass)):
        classValue = UniqueValueOfClass[n]
        perCentageOfClasses['train'][classValue] = {}
        perCentageOfClasses['test'][classValue] = {}
        Class = trainingFold.columns[-1]
        countTrain = trainingFold[trainingFold.Class == classValue].shape[0]
        countTest = testingFold[testingFold.Class == classValue].shape[0]
        perCentageOfClasses['train'][classValue] = countTrain
        perCentageOfClasses['test'][classValue] = countTest
    
    flag = checkStratification(UniqueValueOfClass,perCentageOfClasses)
        
    if flag == 1 :
        clf.fit(trainingFold.drop(labels=trainingFold.columns[-1],axis = 1),trainingFold[trainingFold.columns[-1]])
        testing_x = testingFold.drop(labels=testingFold.columns[-1],axis = 1)
        predictedLabel = clf.predict(testing_x)
        
        accuracy = calculateAccuracy(testingFold[testingFold.columns[-1]],predictedLabel)
        
    else:
        data = data.iloc[np.random.permutation(len(data))].reset_index(drop=True)
        stratifiedKFold(data,noOfInstancesInEachFold,UniqueValueOfClass)
        
    
    return perCentageOfClasses,trainingFold,testingFold,accuracy
    
#-----------------------------------------------------SFS Start----------------------------------------------------
def SFS(k):

    featureNames = train_x.columns
    noOfFeatures = len(featureNames)
    individualFeaturesAccuracy = dict() 
    UniqueValueOfClass = train[train.columns[-1]].unique()
    for i in range(noOfFeatures):
        featureToCheck = train[train.columns[i]]
        featureName = featureNames[i]
        datasetForCrossValidation = pd.concat([featureToCheck,train_y],axis = 1)
        
        noOfInstancesInEachFold = (int) ( len(datasetForCrossValidation)/k )
        
        folds = []
        start = 0
        for j in range(k):
            end = start + noOfInstancesInEachFold
            testFold = datasetForCrossValidation[start:end]
            badIndex = []
            for p in range(len(testFold)):
                badIndex.append(start)
                start += 1
            trainFold = datasetForCrossValidation[~(datasetForCrossValidation.index.isin(badIndex))]
            folds.append( pd.concat([trainFold, testFold], axis=0).reset_index(drop = True) )
            start = end
    
        
        #------------------------------Stratified K Cross Validation----------------------------------
        accuracies = []
        
        for m in range(len(folds)):
            data = folds[m]        
            perCentageOfClasses,trainingFold,testingFold,accuracy = stratifiedKFold(data,noOfInstancesInEachFold,UniqueValueOfClass)
            accuracies.append(accuracy)
        averageAccuracies = np.mean(accuracies)
        individualFeaturesAccuracy[featureName] = averageAccuracies
    
    individualFeaturesAccuracy = OrderedDict(sorted(individualFeaturesAccuracy.items(), key = itemgetter(1), reverse = True))
    
    #-------------------------------------------Selecting Best Features-------------------------------------------
    featureSelected = []
   
    features = list(individualFeaturesAccuracy.keys())
    
    individualFeaturesAccuracy = list(individualFeaturesAccuracy.items())

    currentAccuracy = 0
    previousAccuracy = individualFeaturesAccuracy[0][1]
    featureSelected = []
    featureSelected.append(features[0])

    features.remove(features[0])
 

    while len(features) != None:
        featureSubsetAccuraciesDictionary = {}
        
        for i in range(len(features)): 
            
            featureSubsetCopy = []
            for s in range(len(featureSelected)):
                featureSubsetCopy.append( featureSelected[s] )
            featureSubsetCopy.append(features[i])
            featureForTraining = pd.DataFrame({'columnToRemove':[0]})
            
            for j in range(len(featureSubsetCopy)):
                oneFeatureTrainData = train[train.columns[train.columns.get_loc(featureSubsetCopy[j])]]
                featureForTraining = pd.concat([featureForTraining,oneFeatureTrainData],axis = 1).reset_index(drop=True)
            featureSubsetDataset = pd.concat([featureForTraining,train_y],axis = 1)
            featureSubsetDataset = featureSubsetDataset.drop(featureSubsetDataset.columns[0],axis = 1).reset_index(drop=True)
            noOfInstancesInEachFold = (int) ( len(featureSubsetDataset)/k )
        
            folds = []
            start = 0
            for l in range(k):
                end = start + noOfInstancesInEachFold
                testFold = featureSubsetDataset[start:end]
                badIndex = []
                for p in range(len(testFold)):
                    badIndex.append(start)
                    start += 1
                trainFold = featureSubsetDataset[~(featureSubsetDataset.index.isin(badIndex))]
                folds.append( pd.concat([trainFold, testFold], axis=0).reset_index(drop = True) )
                start = end
        
            
            #------------------------------Stratified K Cross Validation----------------------------------
            accuracies = []
            
            for m in range(len(folds)):
                data = folds[m]
                perCentageOfClasses,trainingFold,testingFold,accuracy = stratifiedKFold(data,noOfInstancesInEachFold,UniqueValueOfClass)
                accuracies.append(accuracy)
            featureAccuracy = np.mean(accuracies)
            featureSubsetAccuraciesDictionary[features[i]] = featureAccuracy

        maximum = max(featureSubsetAccuraciesDictionary, key=featureSubsetAccuraciesDictionary.get) 
        currentAccuracy = featureSubsetAccuraciesDictionary[maximum]
        if currentAccuracy > previousAccuracy:
            previousAccuracy = currentAccuracy
            features.remove(maximum)
            featureSelected.append(maximum)       
        else: 
            break
    
            
    print("No of Features Selected: ",len(featureSelected))
    print("Selected Features: ",featureSelected)
    #---------------------------------------------------------------------------------------------------------
       
#----------------------------------------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------------------------

#-----------------------------------------------------SFS End------------------------------------------------------

#------------------------------------------------------Start-------------------------------------------------------
if __name__ == "__main__":
    SFS(5) #5 is value of k in Cross-Validation
#------------------------------------------------------End---------------------------------------------------------