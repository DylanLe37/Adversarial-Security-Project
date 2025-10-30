import torch
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def evaluateModel(model,X,y,device='cpu'):
    model.eval()

    Xtens = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        outputs = model(Xtens)
        probs = outputs.cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y, preds),
        'precision': precision_score(y, preds),
        'recall': recall_score(y, preds),
        'f1': f1_score(y, preds),
        'roc_auc': roc_auc_score(y, probs),
    }

    return metrics,preds,probs

def plotConfMat(yTrue,yPreds,savePath=None):
    confMat = confusion_matrix(yTrue,yPreds)

    plt.figure(figsize = (10,10))
    sns.heatmap(confMat,annot=True,fmt='d',cmap='Blues')
    plt.xlabel('True label')
    plt.ylabel('Predicted label')

    if savePath:
        plt.savefig(savePath,format='png',bbox_inches='tight')
    plt.show()

def plotROC(yTrue,yProbs,savePath=None):
    fpR,tpR,_ = roc_curve(yTrue,yProbs)
    auc = roc_auc_score(y_true=yTrue,y_score=yProbs)

    plt.figure(figsize = (10,10))
    plt.plot(fpR,tpR,label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0,1],[0,1],'k--',label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)

    if savePath:
        plt.savefig(savePath,format='png',bbox_inches='tight')
    plt.show()