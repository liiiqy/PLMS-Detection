import torch
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
from scipy.special import expit

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def f1(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return f1_score(target.data.cpu().numpy(), pred.cpu().numpy(), average='binary')



def u_f1(output, target):
    # output=torch.sigmoid(output)
    y = []
    pred = []
    with torch.no_grad():
        y.append(target.cpu().numpy())
        pred.append(output.cpu().numpy())
        y = np.array(y).reshape(-1,1)
        pred = np.array(pred).reshape(-1,1)
        yypred= pred>0.5
        yypred = yypred.astype(int)
    return f1_score(y,yypred,average='binary')
    
    
# Dice similarity function
def u_dice(output, target, k = 1):
    pred = []
    true = []
    with torch.no_grad():
        true.append(target.cpu().numpy())
        pred.append(output.cpu().numpy())
        
        true = np.array(true).reshape(-1,1)
        pred = np.array(pred).reshape(-1,1)
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice