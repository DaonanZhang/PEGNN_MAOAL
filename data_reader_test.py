import pickle

import torch
from matplotlib import pyplot as plt
from torch import nn

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
import torch.nn.functional as F
from model import MultiLayerFeedForwardNN




class FeatureEncoder(nn.Module):
    """
    Given a list of environmental features, encode them using the environmental feature encoding
    """

    def __init__(self, env_embed_dim, env_dim=11, settings=None, ffn=None):
        """
        Args:
            env_embed_dim: the output environmental embedding dimension
            env_dim: the dimension of features input
        """
        super(FeatureEncoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env_embed_dim = env_embed_dim
        self.feature_dim = env_dim
        self.ffn = ffn
        # input_dim:13
        self.input_embed_dim = env_dim
        self.nn_length = settings['nn_length']
        self.nn_hidden_dim = settings['nn_hidden_dim']
        if self.ffn is not None:
            # by creating the ffn, the weights are initialized use kaiming_init
            self.ffn = MultiLayerFeedForwardNN(self.input_embed_dim, env_embed_dim,
                                               num_hidden_layers=settings['nn_length'],
                                               hidden_dim=settings['nn_hidden_dim'],
                                               dropout_rate=settings['dropout_rate'])


    def forward(self, env_features):
        """
        Given a list of env_features after one_hot coding, give environmental embedding
        Args:
            env_features: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            env_emb: Tensor shape (batch_size, num_context_pt, env_embed_dim)
        """
        env_emb = env_features
        # Feed Forward Network
        if self.ffn is not None:
            return self.ffn(env_emb)
        else:
            return env_emb

class mini_feature_model(nn.Module):
    def __init__(self, emb_hidden_dim, emb_dim, settings):

        super(mini_feature_model, self).__init__()

        self.envenc = FeatureEncoder(
            env_embed_dim=emb_hidden_dim, ffn=True, settings=settings
        )

        self.envdec = nn.Sequential(
            nn.Linear(emb_hidden_dim, emb_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(emb_hidden_dim // 2, emb_hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(emb_hidden_dim // 4, emb_dim)
        )



    def forward(self, rest_features):
        env_features = self.envenc(rest_features)
        # decrease the dimension of the embedding to the emb_dim
        env_features = self.envdec(env_features)
        return env_features


def padded_seq_to_vectors(padded_seq, logger):
    # Get the actual lengths of each sequence in the batch
    actual_lengths = logger.int()
    # Step 1: Form the first tensor containing all actual elements from the batch
    mask = torch.arange(padded_seq.size(1), device=padded_seq.device) < actual_lengths.view(-1, 1)
    tensor1 = torch.masked_select(padded_seq, mask.unsqueeze(-1)).view(-1, padded_seq.size(-1))

    # Step 2: Form the second tensor to record which row each element comes from
    tensor2 = torch.repeat_interleave(torch.arange(padded_seq.size(0), device=padded_seq.device), actual_lengths)
    return tensor1, tensor2
def main():
    settings = {
        'agent_id': '00000',
        'agent_dir': './logs',
        'origin_path': 'Dataset_res250/',

        # debug mode=>data_set
        'debug': True,
        'bp': False,

        'batch': 16,
        'full_batch': 128,
        'accumulation_steps': 128 // 16,

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
        'd_model': 16,
        'nhead': 4,
        'dim_feedforward': 256,
        'transformer_dropout': 0.1,
        'num_encoder_layers': 3,

        'aux_task_num': 1,

        # if task_head_ use mffn
        'task_head_nn_length': 3,
        'task_head_nn_hidden_dim': 32,
        'task_head_dropout_rate': 0.1,

        'aux_op_dic': {'mcpm1': 0},

        'env_op_dic': {'ta': 0, 'hur': 1, 'plev': 2, 'precip': 3, 'wsx': 4, 'wsy': 5, 'globalrad': 6},

        'hyper': {'lr': 0.001, 'decay': 0.0, 'pre': 0, 'interval': 10, 'aux_loss_weight': 0.1},

        'num_workers': 8,
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


    # aux_loader_train = dl.IntpDataset(settings=settings, mask_distance=-1, call_name='aux')
    # aux_loader_tr = torch.utils.data.DataLoader(aux_loader_train, batch_size=settings['batch'], shuffle=True,
    #                                             collate_fn=dl.collate_fn, num_workers=16, prefetch_factor=32,
    #                                             drop_last=True)

    #
    # aux_iter = iter(aux_loader_tr)
    # data_iter = iter(dataloader_tr)
    #

    model = mini_feature_model(emb_hidden_dim=settings['emb_hidden_dim'], emb_dim=settings['emb_dim'], settings=settings).to(device)


    test_dataloaders = []
    # for mask_distance in [0, 20, 50]:
    for mask_distance in [0]:
        test_dataloaders.append(
            torch.utils.data.DataLoader(
                dl.IntpDataset(settings=settings, mask_distance=mask_distance, call_name='test'),
                batch_size=settings['batch'], shuffle=False, collate_fn=dl.collate_fn, num_workers = settings['num_workers'], prefetch_factor=32, drop_last=True
            )
        )

    for dataloader_ex in test_dataloaders:
        print(len(dataloader_ex))

        for index, test_batch in enumerate(dataloader_ex):

            x_b, c_b, y_b, aux_y_b, input_lengths, rest_features = test_batch

            x_b, c_b, y_b, input_lengths, rest_features = x_b.to(device), c_b.to(device), y_b.to(
                device), input_lengths.to(device), rest_features.to(device)

            aux_y_b = [item.to(device) for item in aux_y_b]





            # print(f'rest_features.shape: {rest_features.shape}')
            #
            # batch_size, num_fields, feature_dim = rest_features.size()
            #
            # rest_feature_Q_reshaped = rest_features.view(batch_size, -1)
            #
            # print(f'rest_feature_Q_reshaped.shape: {rest_feature_Q_reshaped.shape}')
            #
            # batch_size, num_features = rest_feature_Q_reshaped.size()
            #
            # # 定义目标长度
            # target_length = 2600
            #
            # pad_length = target_length - num_features
            #
            # if pad_length > 0:
            #     padded_rest_feature_Q = F.pad(rest_feature_Q_reshaped, (0, pad_length), value=0)
            # else:
            #     padded_rest_feature_Q = rest_feature_Q_reshaped[:, :target_length]
            #
            # print(f'padded_rest_feature_Q.shape: {padded_rest_feature_Q.shape}')


            # outputs_b, targets_b, _, _ = model(inputs=x_b, targets=y_b, coords=c_b,
                #                                    input_lengths=input_lengths,
                #                                    rest_features=rest_features, head_only=True,
                #                                    aux_answers=aux_y_b)
                # output_list.append(outputs_b)
                # target_list.append(targets_b)

    # for i in range(3):
    #     # try:
    #     #     batch = next(data_iter)
    #     # except StopIteration:
    #     #     data_iter = iter(dataloader_tr)
    #     #     batch = next(data_iter)
    #
        # aux_batch = next(aux_iter)

    #     x_b, c_b, y_b, aux_y_b, input_lengths, rest_features = aux_batch
    #
    #     x_b, c_b, y_b, input_lengths, rest_features = x_b.to(device), c_b.to(device), \
    #                                                   y_b.to(device), \
    #                                                   input_lengths.to(device), \
    #                                                   rest_features.to(device)
    # #
    #     aux_y_b = [item.to(device) for item in aux_y_b]
    # #
    #     # print(aux_y_b[0].shape)
    #     # print(y_b.shape)
    #     # torch.Size([32, 76, 1])
    #     # torch.Size([32, 76, 1])
    #
        # print(f'x_b.shape: {x_b.shape}')
        # print(f'c_b.shape: {c_b.shape}')
        # print(f'y_b.shape: {y_b.shape}')
        # print(f'input_lengths.shape: {input_lengths.shape}')
        # print(f'rest_features.shape: {rest_features.shape}')
        # print(f'aux_y_b[0].shape: {aux_y_b[0].shape}')
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
