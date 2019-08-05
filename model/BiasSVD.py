#!/usr/bin/python
#coding=utf-8
'''
    Module:The Bias SVD Algorithm implementation using numpy
    Author: Timmy Qiao
    Date: Jun 11 2019
    The main procedures include:

'''
import numpy as np
import time
import pandas as pd
import pickle

class BiasSVD(object):
    '''
    Implementation of the BiasSVD approach
    '''
    def __init__(self):
        pass

    def __init__(self,epochs,lr,K,Lambda,saveModel=False,ckptStep=10):
        '''
        :param epochs: the max training epoches
        :param lr: learning rate
        :param K: the latent vector dimension
        :param Lambda: the regularized coefficient
        '''
        super(BiasSVD, self).__init__()
        self.epochs = epochs
        self.lr = lr
        self.K = K
        self.Lambda = Lambda
        self.usersDict = {}
        self.itemsDict = {}
        self.saveModel = saveModel
        self.ckptStep = ckptStep

    def fit(self,trainData,valData=None):
        rateNums = trainData.shape[0]
        train_df = pd.DataFrame(trainData) # convert the narray object to DataFrame object
        userIds = np.array(train_df[0].value_counts().keys())
        itemIds = np.array(train_df[1].value_counts().keys())
        userIds = np.sort(userIds)
        itemIds = np.sort(itemIds)
        for index,id in enumerate(userIds):
            self.usersDict[id] = index
        for index,id in enumerate(itemIds):
            self.itemsDict[id] = index

        userNum = len(userIds)
        itemNum = len(itemIds)
        self.meanV = 1.0 * np.sum(trainData[:,2]) / rateNums
        initV = np.sqrt(self.meanV / self.K)
        self.Pu = initV + np.random.uniform(-0.01,0.01,(userNum,self.K))
        self.Qi = initV + np.random.uniform(-0.01,0.01,(itemNum,self.K))
        self.Bu = np.random.rand(userNum) / (self.K ** 0.5)
        self.Bi = np.random.rand(itemNum) / (self.K ** 0.5)

        for i in range(self.epochs):
            epochTime = time.time()
            sumRmse = 0.0
            # Using the SGD approach to optimize the loss function
            # We need to permute the training orders of the same training data
            train_permute_indices = np.random.permutation(trainData.shape[0])
            curStep = 0
            for index in train_permute_indices:
                userId = trainData[index][0]
                itemId = trainData[index][1]
                userIndex = self.usersDict[userId]
                itemIndex = self.itemsDict[itemId]
                rating = trainData[index][2] * 1.0
                # The Estimation of R(u,i) can be represented as:
                # R(u,i) = meanV + Bu[u] + Bi[i] + Pu[u]^T * Qi[i]
                Rui = self.meanV + self.Bu[userIndex] + self.Bi[itemIndex] + \
                self.Pu[userIndex].dot(self.Qi[itemIndex])
                error = rating - Rui
                sumRmse += error ** 2
                p,q = self.Pu[userIndex], self.Qi[itemIndex]
                # Update the parameters using SGD approach
                self.Bu[userIndex] += self.lr * (error - self.Lambda * self.Bu[userIndex])
                self.Bi[itemIndex] += self.lr * (error - self.Lambda * self.Bi[itemIndex])
                self.Pu[userIndex] += self.lr * (error * q - self.Lambda * p)
                self.Qi[itemIndex] += self.lr * (error * p - self.Lambda * q)
                curStep += 1
                stepTime = time.time()
                if curStep % 100000 == 0:
                    print("Epoch %d Step %d cost time %.4f ms, train avg RMSE: %.4f"  \
                          %(i+1,curStep,1000*(time.time() - stepTime),np.sqrt(sumRmse * 1.0 / curStep)))

            epochRmse = np.sqrt(sumRmse * 1.0 / rateNums)

            if valData.any():
                _,valRmse = self.evaluate(valData)
                print("Epoch %d cost time %.4fs, train RMSE: %.4f, validation RMSE: %.4f"  \
                      %(i+1,(time.time()-epochTime),epochRmse,valRmse))

            if self.saveModel and (i + 1) % self.ckptStep == 0:
                model = (self.usersDict, self.itemsDict, self.meanV, self.Bu, self.Bi, self.Pu, self.Qi)
                model_name = 'ckpt_' + str(i + 1) + '.pkl'
                model_path = 'save_model/' + model_name
                with open(model_path, 'wb') as fi:
                    pickle.dump(model, fi)

        return (self.usersDict, self.itemsDict, self.meanV, self.Bu, self.Bi, self.Pu, self.Qi)

    def evaluate(self,val):
        print('Validating the validation dataset now...')
        loss = 0.0
        preds = []
        for i in range(val.shape[0]):
            sample = val[i]
            userId = sample[0]
            itemId = sample[1]
            if userId in self.usersDict \
                    and itemId in self.itemsDict:
                userIndex = self.usersDict[userId]
                itemIndex = self.itemsDict[itemId]
                pred = self.meanV + self.Bu[userIndex] + self.Bi[itemIndex] + \
                       self.Pu[userIndex].dot(self.Qi[itemIndex])
                if pred > 1.0:
                    pred = 1.0
                elif pred < 0.0:
                    pred = 0.0
                preds.append(pred)

                if val.shape[1] == 3:
                    truth = sample[2] * 1.0
                    loss += (pred - truth) ** 2

            if (i+1) % 100000 == 0:
                print('%d data have been validated...'%(i+1))
        print('Validating has been finised...')

        if val.shape[1] == 3:
            rmse = np.sqrt(loss / val.shape[0])
            return pred,rmse

        return pred

    def predict(self,testData):
        return self.evaluate(self,testData)

    def loadModel(self,file_path):
        with open(file_path,'rb') as fi:
            model = pickle.load(fi)
            self.usersDict = model[0]
            self.itemsDict = model[1]
            self.meanV = model[2]
            self.Bu = model[3]
            self.Bi = model[4]
            self.Pu = model[5]
            self.Qi = model[6]






















