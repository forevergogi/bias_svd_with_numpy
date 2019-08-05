from model.BiasSVD import BiasSVD
import pickle
import operator
import time
from threading import Thread
import numpy as np
from numpy import matmul,argsort,transpose,repeat
import pandas as pd

def train(trainData,epochs,lr,K,Lambda,saveModel,valData=None,ckptStep=10):
    svd = BiasSVD(epochs,lr,K,Lambda,saveModel,ckptStep)
    svd.fit(trainData,valData,)

def calc_metric(testData,modelFilePath,topN):
    '''
    The metric includes the Recall and Coverage metrics.
    Recall = sum(The previous picked items in Top N items of a user) / (userNum*N)
    Coverage = total recommended items / total items
    :param test_data:
    :param modelFilePath:
    :param topN:
    :return:
    '''
    with open(modelFilePath, 'rb') as fi:
        model = pickle.load(fi)
        usersDict = model[0]
        itemsDict = model[1]
        meanV = model[2]
        Bu = model[3]
        Bi = model[4]
        Pu = model[5]
        Qi = model[6]
        # M = matmul(Pu, transpose(Qi)) + meanV + Bi + Bu.reshape(-1,1)
        # topNItems= argsort(M,axis=1)[:,topN]
        # hit = 0
        # testItemSet = set()
        # for data in test_data:
        #     userId,itemId,rating = data[0],data[1],data[2]
        #     if itemId in itemsDict:
        #         testItemSet.add(itemId)
        #         if userId in usersDict:
        #             userIndex, itemIndex = usersDict[userId], itemsDict[itemId]
        #             if itemIndex in topNItems[userIndex] and rating == 1:
        #                 hit += 1
        # n_recall = len(testItemSet)
        # recall = hit * 1.0 / n_recall
        #
        # itemsNum = len(itemsDict) # total items number in the training set
        # flattenTopNMatrix = topNItems.flatten()
        # topNItemsSet = set(flattenTopNMatrix)
        # coverage = len(topNItemsSet) * 1.0 / itemsNum
        topNItemsDict ={}
        topNItemSet = set()

        for userId in usersDict:
            startTime = time.time()
            userIndex = usersDict[userId]
            itemsRatingDict = {}
            for itemId in itemsDict:
                itemIndex = itemsDict[itemId]
                Rui = meanV + Bu[userIndex] + Bi[itemIndex] + Pu[userIndex].dot(Qi[itemIndex])
                itemsRatingDict[itemId] = Rui
            itemsRatingDictSorted = sorted(itemsRatingDict.items(),key=operator.itemgetter(1),reverse=True)
            topNItemsIdList = []
            for i in range(topN):
                itemId = itemsRatingDictSorted[i][0]
                topNItemsIdList.append(itemId)
                topNItemSet.add(itemId)
            topNItemsDict[userId] = topNItemsIdList
            endTime = time.time()
            print("userid: %d, cost time: %.4f" %(userId,(endTime-startTime)))
        hit = 0
        testItemSet = set()
        for data in testData:
            userId,itemId,rating = data[0],data[1],data[2]
            if itemId in itemsDict:
                testItemSet.add(itemId)
                if userId in usersDict:
                    userIndex, itemIndex = usersDict[userId], itemsDict[itemId]
                    if itemIndex in topNItemsDict[userIndex] and rating == 1:
                        hit += 1
        n_recall = len(testItemSet)
        recall = hit * 1.0 / n_recall

        itemsNum = len(itemsDict) # total items number in the training set
        coverage = len(topNItemSet) * 1.0 / itemsNum

        return recall,coverage


class MetricCalculator(object):
    def __init__(self,testData,modelFilePath,topN):
        self.testData = testData
        self.topN = topN
        with open(modelFilePath, 'rb') as fi:
            model = pickle.load(fi)
            self.usersDict = model[0]
            self.itemsDict = model[1]
            self.meanV = model[2]
            self.Bu = model[3]
            self.Bi = model[4]
            self.Pu = model[5]
            self.Qi = model[6]
        self.topNItemsDict = {}
        self.topNItemSet = set()






