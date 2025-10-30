import torch
import sys
import os

sys.path.insert(0,'src')

from model.detectionModel import malwareDetector
from utils.dataLoad import loadEmber
from utils.modelEval import evaluateModel,plotConfMat,plotROC

print('Loading test data...')

_,_,xTest,yTest = loadEmber()

checkpoint = torch.load('Results/models/malwareDetectorBest.pth')

model = malwareDetector(input_dim = xTest.shape[1])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

metrics,preds,probs = evaluateModel(model,xTest,yTest)

print('Test Set Performance:')
print(f'\n Acc: {metrics['accuracy']:.4f}')
print(f'\n Precision: {metrics['precision']:.4f}')
print(f'\n Recall: {metrics['recall']:.4f}')
print(f'\n F1-Score: {metrics['f1']:.4f}')
print(f'\n ROC-AUC: {metrics['roc_auc']:.4f}')

print(f'\n Making figures...')
os.makedirs('Results/Figures',exist_ok=True)

plotConfMat(yTest,preds,'Results/Figures/ConfusionMatrixBaseline.png')
plotROC(yTest,probs,'Results/Figures/ROCCurveBaseline.png')