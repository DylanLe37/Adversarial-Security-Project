import torch
import torch.nn as nn
import numpy as np
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

def FGSM(model,X,y,eps=0.1,device='cpu',verbose=True):

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(X.shape[1],),
        nb_classes=2,
        clip_values=(X.min(), X.max())
    )

    attack = FastGradientMethod(
        estimator=classifier,
        eps=eps,
        eps_step=eps,
        targeted=False,
        num_random_init=0,
        batch_size=128
    )

    xAdv = attack.generate(x=X)

    model.eval()
    with torch.no_grad():
        xTens = torch.FloatTensor(X).to(device)
        cleanOutputs = model(xTens).cpu().numpy().flatten()
        cleanPreds = (cleanOutputs > 0.5).astype(int)
    with torch.no_grad():
        xAdvTens = torch.FloatTensor(xAdv).to(device)
        advOutputs = model(xAdvTens).cpu().numpy().flatten()
        advPreds = (advOutputs > 0.5).astype(int)

    attackMetrics={}

    attackMetrics['successRate'] = np.mean(cleanPreds !=advPreds)

    attackMetrics['cleanAccuracy'] = np.mean(cleanPreds==y)
    attackMetrics['advAccuracy'] = np.mean(advPreds==y)
    attackMetrics['accuracyDrop'] = attackMetrics['cleanAccuracy'] - attackMetrics['advAccuracy']

    malware = (y==1)
    notMalware = (y==0)

    correctMalware = (cleanPreds[malware]==1)

    if np.sum(correctMalware)>0:
        attackMetrics['malwareEvasionRate'] = np.mean(advPreds[malware][correctMalware]==0)
    else:
        attackMetrics['malwareEvasionRate'] = 0

    correctNotMalware = (cleanPreds[notMalware]==0)

    if np.sum(correctNotMalware)>0:
        attackMetrics['falsePositiveRate'] = np.mean(advPreds[notMalware][correctNotMalware]==1)
    else:
        attackMetrics['falsePositiveRate'] = 0

    perturbation = xAdv-X
    attackMetrics['averagePerturbation'] = np.mean(np.abs(perturbation))
    attackMetrics['maxPerturbation'] = np.max(np.abs(perturbation))

    if verbose==True:
        print(f'Clean Acc:{attackMetrics['cleanAccuracy']:.2%}')
        print(f'Adversarial Acc:{attackMetrics['advAccuracy']:.2%}')
        print(f'Acc drop:{attackMetrics['accuracyDrop']:.2%}')
        print(f'Attack Success Rate:{attackMetrics['successRate']:.2%}')
        print(f'Malware Evasion Rate: {attackMetrics['malwareEvasionRate']:.2%}')
        print(f'False Positive Rate: {attackMetrics['falsePositiveRate']:.2%}')
        print(f'Average Perturbation:{attackMetrics['averagePerturbation']:.4f}')
        print(f'Max Perturbation:{attackMetrics['maxPerturbation']:.4f}')

    return xAdv,attackMetrics

if __name__ == '__main__':
    import sys
    sys.path.insert(0,'src')
    from model.detectionModel import malwareDetector
    from utils.dataLoad import loadEmber

    _,_,xTest,yTest = loadEmber()

    checkpoint = torch.load('Results/models/malwareDetectorBest.pth')
    model=malwareDetector(input_dim=xTest.shape[1])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    xSamp = xTest[:1000]
    ySamp = yTest[:1000]

    xAdv,metrics = FGSM(model,xSamp,ySamp)

    print(f'\n Larger steps')
    xAdv, metrics = FGSM(model, xSamp, ySamp, eps=0.2)

    print(f'\n Smaller steps')
    xAdv, metrics = FGSM(model, xSamp, ySamp, eps=0.05)