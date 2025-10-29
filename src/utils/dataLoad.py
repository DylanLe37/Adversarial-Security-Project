import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pyarrow.parquet import ParquetFile
import pyarrow as pa

def loadEmber(dataPath="/home/dylan/Documents/Adversarial-Security-Project/Data/Ember",sampleCount=250000): #no ram need subsamp, data has no struct so can take whatever
    print('Loading Ember...')
    print('Currently datapath is hardcoded, remember to change that')

    yTrain = np.squeeze(pd.read_parquet(f'{dataPath}/train_ember_2018_v2_features.parquet',columns=['Label'])
                        .values.astype('int32'))

    pf = ParquetFile(f'{dataPath}/train_ember_2018_v2_features.parquet')
    subSet = next(pf.iter_batches(batch_size=sampleCount))
    xTrain = pa.Table.from_batches([subSet]).to_pandas()
    xTrain = xTrain.drop(columns=['Label']).values
    yTrain = yTrain[:sampleCount]

    goodTrain = yTrain!=-1

    yTrain = yTrain[goodTrain]
    xTrain = xTrain[goodTrain]

    yTest = np.squeeze(pd.read_parquet(f'{dataPath}/test_ember_2018_v2_features.parquet',columns=['Label'])
                        .values.astype('int32'))
    pf = ParquetFile(f'{dataPath}/test_ember_2018_v2_features.parquet')
    subSet = next(pf.iter_batches(batch_size=int(sampleCount/4)))
    xTest = pa.Table.from_batches([subSet]).to_pandas()
    xTest = xTest.drop(columns=['Label']).values
    yTest = yTest[:int(sampleCount/4)]

    scaler = StandardScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.transform(xTest)

    return xTrain, yTrain, xTest, yTest