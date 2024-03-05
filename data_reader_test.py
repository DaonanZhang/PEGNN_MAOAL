import pickle

import torch
from matplotlib import pyplot as plt

import Dataloader_PEGNN as dl
import myconfig
import solver
import support_functions

import torch
import os
import random
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


def main():
    settings = {
        'agent_id': '00000',
        'agent_dir': './logs',
        'origin_path': 'Dataset_res250/',

        # debug mode=>data_set
        'debug': True,
        'bp': False,

        'batch': 32,
        'full_batch': 128,
        'accumulation_steps': 128 // 32,

        'nn_lr': 1e-5,
        'emb_dim': 16,
        'k': 20,
        'conv_dim': 256,

        'emb_hidden_dim': 256,
        'epoch': 1000,
        'test_batch': 32,
        'es_mindelta': 0.5,
        'es_endure': 10,

        'num_features_in': 2,

        'num_features_out': 1,

        'seed': 1,
        'model': 'PEGNN',
        'fold': 0,
        'holdout': [0, 1],
        'lowest_rank': 1,

        'hp_marker': 'tuned',
        'nn_length': 3,
        'nn_hidden_dim': 32,
        'dropout_rate': 0.1,

        # for transformer
        'd_model': 19,
        'nhead': 19,
        'dim_feedforward': 1024,
        'transformer_dropout': 0.1,
        'num_encoder_layers': 3,

        'aux_task_num': 3,

        # if task_head_ use mffn
        # 'task_head_nn_length': 3,
        # 'task_head_nn_hidden_dim': 32,
        # 'task_head_dropout_rate': 0.1,

        'aux_op_dic': {'mcpm1': 0, 'mcpm2p5': 1, 'mcpm4': 2},

        'env_op_dic': {'ta': 0, 'hur': 1, 'plev': 2, 'precip': 3, 'wsx': 4, 'wsy': 5, 'globalrad': 6, 'ncpm1': 7,
                       'ncpm2p5': 8},

        'hyper': {'lr': 0.001, 'decay': 0.0, 'pre': 0, 'interval': 10, 'aux_loss_weight': 0.1},
    }

    coffer_slot = myconfig.coffer_path + str("123456") + '/'

    settings['coffer_slot'] = coffer_slot

    support_functions.seed_everything(settings['seed'])

    fold = settings['fold']
    holdout = settings['holdout']
    lowest_rank = settings['lowest_rank']

    coffer_slot = settings['coffer_slot'] + f'{fold}/'

    support_functions.make_dir(coffer_slot)

    if not torch.cuda.is_available():
        device = torch.device("cpu")
        ngpu = 0
        print(f'Working on CPU')
    else:
        device = torch.device("cuda")
        ngpu = torch.cuda.device_count()
        if ngpu > 1:
            device_list = [i for i in range(ngpu)]
            print(f'Working on multi-GPU {device_list}')
        else:
            print(f'Working on single-GPU')

        # get standarization restore info
    with open(settings['origin_path'] + f'Folds_Info/norm_{fold}.info', 'rb') as f:
        dic_op_minmax, dic_op_meanstd = pickle.load(f)

    from model import PEGCN

    # model = PEGCN(num_features_in=settings['num_features_in'], num_features_out=settings['num_features_out'],
    #               emb_hidden_dim=settings['emb_hidden_dim'], emb_dim=settings['emb_dim'], k=settings['k'],
    #               conv_dim=settings['conv_dim'], aux_task_num=settings['aux_task_num'], settings=settings).to(device)
    # model = model.float()

    # dataset_train = dl.IntpDataset(settings=settings, mask_distance=-1, call_name='train')
    #
    # dataloader_tr = torch.utils.data.DataLoader(dataset_train, batch_size=settings['batch'], shuffle=True,
    #                                             collate_fn=dl.collate_fn, num_workers=1, prefetch_factor=32,
    #                                             drop_last=False)


    aux_loader_train = dl.IntpDataset(settings=settings, mask_distance=-1, call_name='aux')
    aux_loader_tr = torch.utils.data.DataLoader(aux_loader_train, batch_size=settings['batch'], shuffle=True,
                                                collate_fn=dl.collate_fn, num_workers=16, prefetch_factor=32,
                                                drop_last=True)

    #
    aux_iter = iter(aux_loader_tr)
    # data_iter = iter(dataloader_tr)
    #
    for i in range(3):
    #     # try:
    #     #     batch = next(data_iter)
    #     # except StopIteration:
    #     #     data_iter = iter(dataloader_tr)
    #     #     batch = next(data_iter)
    #
        aux_batch = next(aux_iter)
    #
        x_b, c_b, y_b, aux_y_b, input_lengths, rest_features = aux_batch

        x_b, c_b, y_b, input_lengths, rest_features = x_b.to(device), c_b.to(device), \
                                                      y_b.to(device), \
                                                      input_lengths.to(device), \
                                                      rest_features.to(device)
    #
        aux_y_b = [item.to(device) for item in aux_y_b]
    #
    #     # print(aux_y_b[0].shape)
    #     # print(y_b.shape)
    #     # torch.Size([32, 76, 1])
    #     # torch.Size([32, 76, 1])
    #
        print(f'x_b.shape: {x_b.shape}')
        print(f'c_b.shape: {c_b.shape}')
        print(f'y_b.shape: {y_b.shape}')
        print(f'input_lengths.shape: {input_lengths.shape}')
        print(f'rest_features.shape: {rest_features.shape}')
        print(f'aux_y_b[0].shape: {aux_y_b[0].shape}')
    #
    #     # x_b.shape: torch.Size([32, 76, 2])
    #     # c_b.shape: torch.Size([32, 76, 2])
    #     # y_b.shape: torch.Size([32, 76, 1])
    #     # input_lengths.shape: torch.Size([32])
    #     # rest_features.shape: torch.Size([32, 196, 13])
    #     # aux_y_b[0].shape: torch.Size([32, 76, 1])
    #
    #     # outputs_b, targets_b, aux_outputs_b, aux_targets_b = model(inputs = x_b, targets = y_b, coords = c_b, input_lengths = input_lengths,
    #     #                              rest_features = rest_features, head_only= True, aux_answers = aux_y_b)


if __name__ == "__main__":
    main()
