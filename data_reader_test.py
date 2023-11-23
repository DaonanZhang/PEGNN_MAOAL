import pickle

import torch
from matplotlib import pyplot as plt

import Dataloader_PEGNN as dl
import myconfig
import support_functions

import torch
import os
import random
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset



settings = {
    'agent_id': '00000',
    'agent_dir': './logs',
    'origin_path': '../Dataset_res250/',

    # debug mode=>data_set
    'debug': False,
    'bp': False,

    # full_batch->batch->accumulation_steps double
    'batch': 32,
    'accumulation_steps': 512 // 32,
    'epoch': 1000,
    'test_batch': 25,
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
    'holdout': 0,
    'lowest_rank': 1,

    'hp_marker': 'tuned',

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


dataloader_tr = torch.utils.data.DataLoader(dataset_train, batch_size=settings['batch'], shuffle=False, collate_fn=dl.collate_fn, num_workers=16, prefetch_factor=32, drop_last=False)

#
# print(dataset_train[1][0])
# print(dataset_train.op_dic)
# # 只有三个torch
# print(dataloader_tr)


# data_iter = iter(dataloader_tr)
# batch_data = next(data_iter)
# # 假设 batch_data 包含输入图像和对应的标签
# images, labels = batch_data
#
# # 可视化图像
#
# plt.figure(figsize=(10, 10))
# for i in range(len(images)):
#     plt.subplot(1, len(images), i + 1)
#     plt.imshow(images[i].permute(1, 2, 0))  # 如果图像是通道在第一维的形式，需要转换通道顺序
#     plt.title(f"Label: {labels[i]}")
#     plt.axis("off")
#
# plt.show()



