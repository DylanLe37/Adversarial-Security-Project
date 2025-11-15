import sys
sys.path.insert(0, 'src')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import os

from model.detectionModel import malwareDetector
from utils.dataLoad import loadEmber
from utils.modelEval import evaluateModel
from attacks.fgsm import FGSM
from attacks.pgd import PGD

def trainWithAdvEx(model,xTrain,yTrain,xTest,yTest,attackType='pgd',eps=0.1,epochs=15,batchSize=128,learningRate=0.001,device='cpu'):

    xTrainTens = torch.FloatTensor(xTrain)
    yTrainTens = torch.LongTensor(yTrain)
    xTestTens = torch.FloatTensor(xTest)
    yTestTens = torch.LongTensor(yTest)

    trainData = TensorDataset(xTrainTens,yTrainTens)
    trainLoader = DataLoader(trainData,batch_size=batchSize,shuffle=True)
    testData = TensorDataset(xTestTens,yTestTens)
    testLoader = DataLoader(testData,batch_size=batchSize,shuffle=True)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),lr=learningRate)

    bestRobustAcc = 0

    for epoch in range(epochs):
        model.train()
        totalLoss = 0

        for batchX,batchY in tqdm(trainLoader,desc="Training"):
            batchX = batchX.to(device)
            batchY = batchY.to(device).float().unsqueeze(1)

            batchXNP = batchX.cpu().numpy()
            batchYNP = batchY.cpu().numpy().flatten().astype(int)

            if attackType == 'fgsm':
                batchXAdv,_ = FGSM(model,batchXNP,batchYNP,eps=eps,verbose=False)
            else:
                batchXAdv,_ = PGD(model,batchXNP,batchYNP,eps=eps,eps_step=eps/10,maxIter=10,verbose=False)

            # batchXAdv = np.array(batchXAdv)
            batchXAdv = torch.FloatTensor(batchXAdv).to(device)

            combinedX = torch.cat([batchX,batchXAdv],dim=0)
            combinedY = torch.cat([batchY,batchY],dim=0)

            outputs = model(combinedX)
            loss = criterion(outputs,combinedY)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totalLoss+=loss.item()

        averageLoss = totalLoss/len(trainLoader)

        cleanMetrics,_,_, = evaluateModel(model,xTest,yTest,device)
        print(f'Clean Test Accuracy: {cleanMetrics['accuracy']:.4f}')

        if attackType == 'fgsm':
            xTestAdv,_ = FGSM(model,xTest[:1000],yTest[:1000],eps=eps,verbose=False)
        else:
            xTestAdv,_ = PGD(model,xTest[:1000],yTest[:1000],eps=eps,eps_step=eps/10,maxIter=40,verbose=False)

        robustMetrics,_,_ = evaluateModel(model,xTestAdv,yTest[:1000],device)
        print(f'Robust Test Accuracy: {robustMetrics['accuracy']:.4f}')

        if robustMetrics['accuracy'] > bestRobustAcc:
            bestRobustAcc = robustMetrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'cleanAccuracy': cleanMetrics['accuracy'],
                'robustAccuracy': robustMetrics['accuracy'],
                'attackType': attackType,
                'eps': eps
            },f'Results/models/malwareDetector_AdvTrain_{attackType}_eps{eps}.pth')

    print(f'Best robust accuracy: {bestRobustAcc:.4f}')
    return model

if __name__ == '__main__':
    xTrain,yTrain,xTest,yTest = loadEmber()

    model = malwareDetector(inputDim = xTrain.shape[1])
    device = torch.device('cpu')
    model = model.to(device)

    model = trainWithAdvEx(
        model,xTrain,yTrain,xTest,yTest,
        attackType ='pgd',
        eps=0.1,
        epochs=15,
        device=device)