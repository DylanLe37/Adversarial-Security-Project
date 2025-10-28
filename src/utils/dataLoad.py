import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def loadEmber(dataPath="/home/dylan/Documents/Adversarial-Security-Project/Data/Ember",data='Train'):
    print('Loading Ember...')

    trainDat = pd.read_parquet(f'{dataPath}/train_ember_2018_v2_features.parquet')
    testDat = pd.read_parquet(f'{dataPath}/test_ember_2018_v2_features.parquet')

    labelCol = trainDat.columns[-1]

    yTrain = trainDat[labelCol].astype('int32')
    yTest = testDat[labelCol].values.astype('int32')
    xTrain = trainDat.drop(columns=[labelCol]).values
    xTest = testDat.drop(columns=[labelCol]).values

    goodTrain = yTrain!=-1
    yTrain = yTrain[goodTrain]
    xTrain = xTrain[goodTrain]

    scaler = StandardScaler()
    scaler.fit_transform(xTrain)