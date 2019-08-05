from utils.train import train,calc_metric
from utils.prepare_training import get_data
import os

if __name__ == '__main__':
    trainData,testData,valData = get_data()
    # K = 30
    # epochs = 50
    # lr = 0.05
    # Lambda = 0.5
    # saveModel = True
    # ckptStep = 10
    # model = train(trainData,epochs,lr,K,Lambda,saveModel,valData,ckptStep)
    iter = 10
    topN = 10
    modelFilePath = 'save_model_exp2'
    modelFileName = 'ckpt_' + str(iter) + '.pkl'
    modelFullPath = os.path.join(modelFilePath,modelFileName)
    recall,coverage = calc_metric(testData,modelFullPath,topN)
    print(recall,coverage)