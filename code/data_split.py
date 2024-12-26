# -*- coding: utf-8 -*-
import json
import argparse
import torch
import torch.nn as nn

from utils.util import *

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def save(dict):
    if isinstance(dict, str):
        dict = eval(dict)
    with open('data_split.txt', 'w', encoding='utf-8') as f:
        str_ = json.dumps(dict, ensure_ascii=False) 
        f.write(str_)

def load():
    with open('data_split.txt', 'r', encoding='utf-8') as f:
        data = f.readline().strip()
        # print(type(data), data)
        dict = json.loads(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data_split')
    parser.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    parser.add_argument('-f', '--num_folds', type=int,default=1,
                      help='num_folds')
    parser.add_argument('-da', '--np_data_dir', type=str,default='.\data\load_data',
                      help='Directory containing numpy files')
    
    args = parser.parse_args()
    folds_data = load_foldsdata(args.np_data_dir, args.num_folds,random=SEED)
    save(folds_data)
