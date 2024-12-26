import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import os
import numpy as np
from glob import glob
import math
from sklearn.model_selection import LeaveOneOut
from statsmodels.tsa.ar_model import AutoReg
from adacost import AdaCostClassifier
from EMGfeature import feature

def load_foldsdata(np_data_path, n_folds,random=123):
    files = sorted(glob(os.path.join(np_data_path, "*.npz"))) 

    folds_data={}  
    num_patients=n_folds
    id=0
    
    loo=LeaveOneOut()
    for train_index,test_index in loo.split(np.arange(0,num_patients)): 
        index=test_index[0]
        # for step 2
        train_files=[]
        test_files=[]
        train_files_1_num=0
        train_files_0_num=0
        # for step 1
        window_size=50
        move=50
        X_train=np.zeros(shape=(0,9*30))
        y_train=np.zeros(shape=(0))

        for i in train_index: 
            string= '%02d' % i
            tmp_train_files=[s for s in files if 'A'+string+'_' in s]          
            
            # train      
            for j in range(len(tmp_train_files)):
                a=np.load(tmp_train_files[j])
                x=a['EMGdata'].reshape(-1)
                x=x[62:1564]
                features = feature(x, window_size, move)
                X_train=np.concatenate((X_train,features),axis=0)
                
                if a['LM']==1:
                    y_train=np.append(y_train,1)
                    train_files.append(tmp_train_files[j])
                    train_files_1_num+=1

                elif a['LM']==0 and (train_files_1_num/2)>train_files_0_num:
                    y_train=np.append(y_train,-1)
                    train_files.append(tmp_train_files[j])
                    train_files_0_num+=1 
                else: 
                    y_train=np.append(y_train,-1)
                    train_files_0_num+=1
        
        model = AdaCostClassifier(early_termination=True,random_state=random,learning_rate=1)    
        model.fit(X_train, y_train) 
        
        X_test=np.zeros(shape=(0,9*30))    

        # for step 2
        test_files_1_num=0
        test_files_0_num=0
        lst=[]
        
        for i in test_index: 
            string= '%02d' % i
            tmp_test_files=[s for s in files if 'A'+string+'_' in s]
            
            for j in range(len(tmp_test_files)):
                a=np.load(tmp_test_files[j])
                x=a['EMGdata'].reshape(-1)
                x=x[62:1564]

                features=feature(x,window_size,move)
                X_test=np.concatenate((X_test,features),axis=0)       
                    
            predictions = model.predict(X_test)
            
            for j in range(len(tmp_test_files)):   
                if predictions[j]==1:
                    test_files.append(tmp_test_files[j])
                    test_files_1_num+=1 
                    lst.append(j)  
                else:
                    test_files_0_num+=1 
                    
        
        np.save('lst_'+str(id)+'.npy', lst)  

        folds_data[id]=[train_files,test_files]   
        id+=1
    return folds_data




def calc_class_weight(labels_count):
    total = np.sum(labels_count)
    class_weight = dict()
    num_classes = len(labels_count)

    factor = 1 / num_classes
    mu = [factor * 1.5, factor * 2, factor * 1.5, factor, factor * 1.5]

    for key in range(num_classes):
        score = math.log(mu[key] * total / float(labels_count[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        class_weight[key] = round(class_weight[key] * factor, 2)

    class_weight = [class_weight[i] for i in range(num_classes)]

    return class_weight


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
