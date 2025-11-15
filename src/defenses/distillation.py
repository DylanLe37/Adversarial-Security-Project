import sys
sys.path.insert(0, 'src')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import copy

from model.detectionModel import malwareDetector
from utils.dataLoad import loadEmber
from utils.modelEval import evaluateModel
from attacks.fgsm import FGSM
from attacks.pgd import PGD

class DistilledModel(nn.Module):
    def __init__(self,baseModel,temp=1):
        super().__init__()
        self.baseModel = baseModel
        self.temp = temp

    def forward(self,x,useTemp=False):
        logits = self.baseModel.network[:-1](x)
        if useTemp:
            return torch.sigmoid(logits/self.temp)
        else:
            return torch.sigmoid(logits)

def trainTeacher(model,xTrain,yTrain,xTest,yTest,temp=20,epochs=15,batchSize=128,learningRate=0.001,device='cpu'):
    print(f'Training teacher model')

    teacher = DistilledModel(model,temp=temp)

    xTrainTens = torch.FloatTensor(xTrain).to(device)
    yTrainTens = torch.FloatTensor(yTrain).unsqueeze(1).to(device)
    xTestTens = torch.FloatTensor(xTest).to(device)
    yTestTens = torch.LongTensor(yTest)

    trainData = TensorDataset(xTrainTens, yTrainTens)
    trainLoader = DataLoader(trainData,batch_size=batchSize,shuffle=True)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(teacher.parameters(),lr=learningRate)

    for epoch in range(epochs):
        teacher.train()
        totalLoss = 0

        for batchX,batchY in tqdm(trainLoader,desc=f'Epoch {epoch+1}/{epochs}'):
            outputs = teacher(batchX,useTemp=True)
            loss = criterion(outputs,batchY)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totalLoss += loss.item()

        averageLoss = totalLoss/len(trainLoader)
        print(f'Training Loss: {averageLoss:.4f}')

        teacher.eval()
        with torch.no_grad():
            outputs = teacher(xTestTens,useTemp=False)
            preds = (outputs.cpu().numpy().flatten()>0.5).astype(int)
            accuracy = np.mean(preds==yTestTens.numpy())
            print(f'Test Accuracy: {accuracy:.4f}')

    return teacher

def trainStudent(teacher,xTrain,yTrain,xTest,yTest,temp=20,epochs=15,batchSize=128,learningRate=0.001,device='cpu'):
    print(f'Training student model')

    studentBase = malwareDetector(inputDim=xTrain.shape[1])
    student = DistilledModel(studentBase,temp=temp)

    xTrainTens = torch.FloatTensor(xTrain).to(device)
    yTrainTens = torch.FloatTensor(yTrain).unsqueeze(1).to(device)
    xTestTens = torch.FloatTensor(xTest).to(device)
    yTestTens = torch.LongTensor(yTest)

    trainData = TensorDataset(xTrainTens, yTrainTens)
    trainLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(student.parameters(), lr=learningRate)

    teacher.eval()
    bestAcc = 0

    for epoch in range(epochs):
        student.train()
        totalLoss = 0
        for batchX,batchY in tqdm(trainLoader,desc=f'Epoch {epoch+1}/{epochs}'):
            with torch.no_grad():
                softLabels = teacher(batchX,useTemp=True)
            studentOutputs = student(batchX,useTemp=True)

            loss = criterion(studentOutputs,softLabels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totalLoss += loss.item()

        averageLoss = totalLoss/len(trainLoader)
        print(f'Training Loss: {averageLoss:.4f}')

        student.eval()
        with torch.no_grad():
            outputs = student(xTestTens,useTemp=False)
            preds = (outputs.cpu().numpy().flatten()>0.5).astype(int)
            accuracy = np.mean(preds==yTestTens.numpy())
            print(f'Test Accuracy: {accuracy:.4f}')

        if accuracy > bestAcc:
            bestAcc = accuracy
            torch.save({
                'epoch':epoch,
                'model_state_dict':student.baseModel.state_dict(),
                'accuracy':accuracy,
                'temperateure':temp
            },'Results/models/malwareDetector_distilled.pth')
            print(f'Saved best model with accuracy {bestAcc:.4f}')

    return student

def defensiveDistillation(xTrain,yTrain,xTest,yTest,temp=20,device='cpu'):
    checkpoint = torch.load('Results/models/malwareDetectorBest.pth')
    teacherBase = malwareDetector(inputDim=xTrain.shape[1])
    teacherBase.load_state_dict(checkpoint['model_state_dict'])

    teacher = DistilledModel(teacherBase,temp=temp).to(device)
    teacher.eval()

    with torch.no_grad():
        xTestTens = torch.FloatTensor(xTest).to(device)
        outputs = teacher(xTestTens,useTemp=False)
        preds = (outputs.cpu().numpy().flatten()>0.5).astype(int)
        teacherAcc = np.mean(preds==yTest)
        print(f'Teacher Accuracy: {teacherAcc:.4f}')

    student = trainStudent(teacher,xTrain,yTrain,xTest,yTest,temp=temp,epochs=15,device=device)

    return student

if __name__ == '__main__':
    xTrain,yTrain,xTest,yTest = loadEmber()
    device = torch.device('cpu')

    studentModel = defensiveDistillation(xTrain,yTrain,xTest,yTest,temp=20,device=device)

    studentModel.eval()
    xTestTens = torch.FloatTensor(xTest).to(device)

    with torch.no_grad():
        outputs = studentModel(xTestTens,useTemp=False)
        preds = (outputs.cpu().numpy().flatten()>0.5).astype(int)
        finalAcc = np.mean(preds==yTest)
    print(f'Distilled Model Accuracy: {finalAcc:.4f}')

    xSamp = xTest[:1000]
    ySamp = yTest[:1000]

    baseModel = studentModel.baseModel
    baseModel.eval()

    xAdv,metrics = PGD(baseModel,xSamp,ySamp,eps=0.1)

    print(f'Clean Accuracy:{metrics['cleanAccuracy']:.4f}')
    print(f'Robust Accuracy:{metrics['advAccuracy']:.4f}')