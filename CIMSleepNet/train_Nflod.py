import argparse
import collections
import numpy as np

from data_loader.data_loaders import *
from model.Polyloss import PolyLoss
from model.MISS_SC_loss import Miss_SupConLoss
from model.loss import CrossEntropyLoss, MSELoss, MSE_KLD_Loss
import model.metric as module_metric
# import model.model_CNN_VIT1 as module_arch  # model_EEG_CNNvit_Orthogv2
import model.model_miss_Trans_GRU_edf as module_arch  # model_EEG_CNNvit_Orthogv2
from parse_config import ConfigParser
from trainer.trainer import Trainer
from utils.util import *
import random
import torch
import torch.nn as nn
import os

# fix random seeds for reproducibility
SEED = 111
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


def main(config, fold_id):
    batch_size = config["data_loader"]["args"]["batch_size"]

    logger = config.get_logger('train')

    # build model architecture, initialize weights, then print to console
    model = config.init_obj('arch', module_arch)
    # model.apply(weights_init_normal)
    logger.info(model)

    # get function handles of loss and metrics
    # criterion = getattr(module_loss, config['loss'])
    data_loader, valid_data_loader, test_data_loader, data_count, train_data_count = data_generator_np_miss_edf(folds_data[fold_id][0],
                                                                                     folds_data[fold_id][1],folds_data[fold_id][2], batch_size)

    weights_for_each_class = calc_class_weight_edf(data_count)

    criterion = PolyLoss(ce_weight=weights_for_each_class)
    #criterion = CrossEntropyLoss()
    criterion_Miss_SC = Miss_SupConLoss(temperature=0.07)
    criterion_Miss_MSE_x = MSE_KLD_Loss()
    criterion_Miss_MSE_h = MSE_KLD_Loss()

    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    Miss_SC_trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    Miss_MSEx_trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    Miss_MSEh_trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    Miss_SC_optimizer = config.init_obj('Miss_SC_optimizer', torch.optim, Miss_SC_trainable_params)
    Miss_MSEx_optimizer = config.init_obj('Miss_MSEx_optimizer', torch.optim, Miss_MSEx_trainable_params)
    Miss_MSEh_optimizer = config.init_obj('Miss_MSEh_optimizer', torch.optim, Miss_MSEh_trainable_params)
    trainer = Trainer(model, criterion, criterion_Miss_SC, criterion_Miss_MSE_x, criterion_Miss_MSE_h, metrics,
                      optimizer, Miss_SC_optimizer, Miss_MSEx_optimizer, Miss_MSEh_optimizer,
                      config=config,
                      data_loader=data_loader,
                      fold_id=fold_id,
                      valid_data_loader=valid_data_loader,
                      test_data_loader=test_data_loader,
                      class_weights=weights_for_each_class)


    trainer.train()


if __name__ == '__main__':



    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-f', '--fold_id', type=str, default="0",
                      help='fold_id')
    args.add_argument('-da', '--np_data_dir', type=str, default="data/output_data/.../",
                      help='Directory containing numpy files')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []

    args2 = args.parse_args()
    fold_id = int(args2.fold_id)

    config = ConfigParser.from_args(args, fold_id, options)

    folds_data = load_folds_data(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])

    main(config, fold_id)






