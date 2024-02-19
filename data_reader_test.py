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
        'origin_path': './Dataset_res250/',

        # debug mode=>data_set
        'debug': True,
        'bp': False,

        # full_batch->batch->accumulation_steps double
        'batch': 32,
        'accumulation_steps': 1,
        'epoch': 1000,
        'test_batch': 64,
        'nn_lr': 1e-5,
        'es_mindelta': 0.5,
        'es_endure': 10,

        # 'num_features_in': 14,
        'num_features_in': 2,

        'num_features_out': 1,
        'emb_hidden_dim': 256,
        'emb_dim': 32,
        'k': 20,
        'conv_dim': 256,

        'seed': 1,
        'model': 'PEGNN',
        'fold': 0,
        'holdout': [0,1],
        'lowest_rank': 1,

        'hp_marker': 'tuned',

        'aux_task_num': 3,

        # if task_head_ use mffn
        # 'task_head_nn_length': 3,
        # 'task_head_nn_hidden_dim': 32,
        # 'task_head_dropout_rate': 0.1,

        'aux_op_dic': {'mcpm1': 0, 'mcpm2p5': 1, 'mcpm4': 2},

        'env_op_dic': {'ta': 0, 'hur': 1, 'plev': 2, 'precip': 3, 'wsx': 4, 'wsy': 5, 'globalrad': 6, 'ncpm1': 7,
                       'ncpm2p5': 8}

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


    dataset_train = dl.IntpDataset(settings=settings, mask_distance=-1, call_name='train')

    dataloader_tr = torch.utils.data.DataLoader(dataset_train, batch_size=settings['batch'], shuffle=True, collate_fn=dl.collate_fn, num_workers=1, prefetch_factor=32, drop_last=False)



    # dataset_train2 = dl.IntpDataset(settings=settings, mask_distance=-1, call_name='train')
    #
    # dataloader_tr2 = torch.utils.data.DataLoader(dataset_train2, batch_size=settings['batch'], shuffle=True,
    #                                             collate_fn=dl.collate_fn, num_workers=1, prefetch_factor=32,
    #                                             drop_last=False)


    data_iter = iter(dataloader_tr)
    # data_iter2 = iter(dataloader_tr2)

    for i in range(1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader_tr)
            batch = next(data_iter)


        x_b, c_b, y_b, aux_y_b, input_lenths, rest_feature = batch

        # print(f'x_b: {x_b.shape}')
        # print(f'input_lengths: {input_lengths.shape}')
        # print(f'rest_features: {rest_features.shape}')

        # q_series_tr, input_lenths_tr, input_series_tr, answers_tr = batch
        # print(f'q_series_tr: {q_series_tr.shape}')
        # print(f'input_lenths_tr: {input_lenths_tr.shape}')
        # print(f'input_series_tr: {input_series_tr.shape}')
        # print(f'answers_tr: {answers_tr.shape}')


    # modify in model
    # feature_size = 12  # feature dim
    # nhead = 4  # multi-head num
    # num_layers = 2
    # encoder_layers = TransformerEncoderLayer(d_model=feature_size, nhead=nhead)
    # transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
    #
    # feature_embedding = transformer_encoder(rest_features)
    #
    # print(feature_embedding.shape)

if __name__ == "__main__":
    main()

