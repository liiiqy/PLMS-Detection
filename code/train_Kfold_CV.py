import argparse
import collections
import numpy as np

from data_loader.data_loaders import *
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils.util import *

import torch
import torch.nn as nn

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def weights_init_normal(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.Conv1d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def load():
    with open('data_split.txt', 'r', encoding='utf-8') as f:
        data = f.readline().strip()
        dict = json.loads(data)
        return dict

def main(config, folds_data, fold_id):
    batch_size = config["data_loader"]["args"]["batch_size"]

    logger = config.get_logger('train')

    # build model architecture, initialize weights, then print to console
    model = config.init_obj('arch', module_arch)
    model.apply(weights_init_normal)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    # criterion = getattr(nn.SmoothL1Loss(), config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    fold_id=str(fold_id)
    data_loader, valid_data_loader, data_count = data_generator_np(folds_data[fold_id][0],
                                    folds_data[fold_id][1], batch_size)
    # weights_for_each_class = calc_class_weight(data_count)
    weights_for_each_class = 1
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      fold_id=fold_id,
                      valid_data_loader=valid_data_loader,
                      class_weights=weights_for_each_class)

    trainer.train()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    parser.add_argument('-f', '--fold_id', type=str,default="2",
                      help='fold_id')
    parser.add_argument('-da', '--np_data_dir', type=str,default='.\data\load_data',
                      help='Directory containing numpy files')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []
    args = parser.parse_args()

    
    fold_id=int(args.fold_id)
    config = ConfigParser.from_args(parser, fold_id, options)
    num_folds=config["data_loader"]["args"]["num_folds"]
    folds_data = load()

    main(config, folds_data, fold_id)