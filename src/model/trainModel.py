import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from tqdm import tqdm
import sys
import os
from pathlib import Path
sys.path.insert(0,'src')
from model.detectionModel import malwareDetector
from utils.dataLoad import loadEmber

def trainEpoch(model,dataLoader,criterion,optimizer,device):
    model.train()
    totalLoss = 0

    for batchX,batchY in tqdm(dataLoader,desc="Training"):
        batchX=batchX.to(device)
        batchY=batchY.to(device).float().unsqueeze(1)

        outputs = model(batchX)
        loss = criterion(outputs, batchY)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totalLoss += loss.item()

    return totalLoss/len(dataLoader)

def evaluate(model,dataLoader,device):
    model.eval()
    allPreds = []
    allLabels = []
    allProbs = []

    with torch.no_grad():
        for batchX,batchY in tqdm(dataLoader,desc="Evaluating"):
            batchX=batchX.to(device)

            outputs = model(batchX)
            probs = outputs.cpu()
            preds = (probs > 0.5).int()

            allProbs.extend(probs.flatten())
            allPreds.extend(preds.flatten())
            allLabels.extend(batchY.flatten())

    allPreds = np.array(allPreds)
    allLabels = np.array(allLabels)
    allProbs = np.array(allProbs)

    metrics = {
        'accuracy': accuracy_score(allLabels, allPreds),
        'precision': precision_score(allLabels, allPreds),
        'recall': recall_score(allLabels, allPreds),
        'f1': f1_score(allLabels, allPreds),
        'roc_auc': roc_auc_score(allLabels, allPreds),
    }

    return metrics

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device used:{device}")

    xTrain,yTrain,xTest,yTest = loadEmber()

    xTrainTens = torch.FloatTensor(xTrain)
    yTrainTens = torch.LongTensor(yTrain)
    xTestTens = torch.FloatTensor(xTest)
    yTestTens = torch.LongTensor(yTest)

    batchSize = 128
    trainData = TensorDataset(xTrainTens, yTrainTens)
    testData = TensorDataset(xTestTens, yTestTens)

    trainLoader = DataLoader(trainData,batch_size=batchSize,shuffle=True)
    testLoader = DataLoader(testData,batch_size=batchSize,shuffle=False)

    inputDim = xTrain.shape[1]
    model = malwareDetector(input_dim = inputDim).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    numEpochs = 15
    bestF1 = 0

    print('Training...')
    for epoch in range(numEpochs):
        trainLoss = trainEpoch(model,trainLoader,criterion,optimizer,device)
        print(f'Training Loss:{trainLoss:.4f}')

        metrics = evaluate(model,testLoader,device)
        print(f'Test Acc: {metrics['accuracy']:.4f}')
        print(f'Test Precision: {metrics["precision"]:.4f}')
        print(f'Test Recall: {metrics["recall"]:.4f}')
        print(f'Test F1: {metrics["f1"]:.4f}')
        print(f'Test ROC-AUC: {metrics["roc_auc"]:.4f}')

        if metrics['f1'] > bestF1:
            bestF1 = metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            },'results/models/malwareDetectorBest.pth')

    print(f'\n Training done, best F1:{bestF1:.4f}')

if __name__ == '__main__':
    os.makedirs('Results/models',exist_ok=True)
    main()