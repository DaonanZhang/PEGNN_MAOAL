import os, glob, inspect, time, math, torch, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# in the slurm file
sys.path.append('/pfs/data5/home/kit/tm/lm6999/GPR_INTP_SAQN/Datasets')


import Dataloader_PEGNN as dl
from model import *
import torch.optim as optim
from torch.nn import functional as F
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from matplotlib.lines import Line2D
import myconfig
from datetime import datetime
import json
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LambdaLR
import random

import sys

# in the slurm file
sys.path.append('/pfs/data5/home/kit/tm/lm6999/GPR_INTP_SAQN/util')

import support_functions



# where is the auxilary task and the loss from there?
def bmc_loss(pred, target, noise_var, device):
    """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).to(device))     # contrastive-like loss
    # loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 
    return loss

# lack of the file: support_functions.py !!
    
def training(settings, job_id):
    support_functions.seed_everything(settings['seed'])
    
    fold = settings['fold']
    holdout = settings['holdout']
    lowest_rank = settings['lowest_rank']
    
    coffer_slot = settings['coffer_slot'] + f'{fold}/'
    support_functions.make_dir(coffer_slot)
    
    # print sweep settings
    print(json.dumps(settings, indent=2, default=str))
    
    # Get device setting
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

    # no such folder?
    # get standarization restore info
    with open(settings['origin_path'] + f'Folds_Info/norm_{fold}.info', 'rb') as f:
        dic_op_minmax, dic_op_meanstd = pickle.load(f)

    # build dataloader
    dataset_train = dl.IntpDataset(settings=settings, mask_distance=-1, call_name='train')
    dataloader_tr = torch.utils.data.DataLoader(dataset_train, batch_size=settings['batch'], shuffle=True, collate_fn=dl.collate_fn, num_workers=16, prefetch_factor=32, drop_last=True)
    
    # self_dataloaders = []
    # self_dataloaders.append(
    #     torch.utils.data.DataLoader(
    #         dl.IntpDataset(settings=settings, mask_distance=0, call_name='train_self'), 
    #         batch_size=settings['batch'], shuffle=False, collate_fn=dl.collate_fn, num_workers=16, prefetch_factor=32, drop_last=True
    #     )
    # )
    
    test_dataloaders = []
    for mask_distance in [0, 20, 50]:
        test_dataloaders.append(
            torch.utils.data.DataLoader(
                dl.IntpDataset(settings=settings, mask_distance=mask_distance, call_name='test'), 
                batch_size=settings['batch'], shuffle=False, collate_fn=dl.collate_fn, num_workers=16, prefetch_factor=32, drop_last=True
            )
        )
    
    # build model
    model = PEGCN(num_features_in=settings['num_features_in'], num_features_out=settings['num_features_out'], emb_hidden_dim=settings['emb_hidden_dim'], emb_dim=settings['emb_dim'], k=settings['k'], conv_dim=settings['conv_dim']).to(device)
    model = model.float()
    loss_func = torch.nn.MSELoss()
    if ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=device_list)
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['nn_lr'])
    
    # set training loop
    epochs = settings['epoch']
    batch_size = settings['batch']
    print("\nTraining to %d epochs (%d of test batch size)" %(epochs, batch_size))
    
    # fire training loop
    start_time = time.time()
    list_total = []
    list_err = []
    best_err = float('inf')
    es_counter = 0
    
    iter_counter = 0
    inter_loss = 0
    mini_loss = 0
    data_iter = iter(dataloader_tr)
    
    t_train_iter_start = time.time()


    while True:
        # train 1 iteration
        model.train()
        real_iter = iter_counter//settings['accumulation_steps'] + 1

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader_tr)
            batch = next(data_iter)
        x_b, c_b, y_b, input_lenths = batch
        
        x_b, c_b, y_b, input_lenths = x_b.to(device), c_b.to(device), y_b.to(device), input_lenths.to(device)
        outputs_b, targets_b = model(x_b, y_b, c_b, input_lenths, True)
        
        batch_loss = loss_func(outputs_b, targets_b)
        batch_loss /= settings['accumulation_steps']

        inter_loss += batch_loss.item()
        mini_loss += batch_loss.item()
        # backward propagation
        batch_loss.backward()

        if (iter_counter+1) % settings['accumulation_steps'] == 0:
            optimizer.step()
            optimizer.zero_grad()
            t_train_iter_end = time.time()
            print(f'\tIter {real_iter} - Loss: {mini_loss} - real_iter_time: {t_train_iter_end - t_train_iter_start}', end="\r", flush=True)
            mini_loss = 0
            t_train_iter_start = t_train_iter_end
        iter_counter += 1
        

        # Test batch
        if (iter_counter+1) % (settings['test_batch'] * settings['accumulation_steps']) == 0:
            model.eval()
            output_list = []
            target_list = []
            test_loss = 0
            for dataloader_ex in test_dataloaders:
                for x_b, c_b, y_b, input_lenths in dataloader_ex:
                    x_b, c_b, y_b, input_lenths = x_b.to(device), c_b.to(device), y_b.to(device), input_lenths.to(device)
                    outputs_b, targets_b = model(x_b, y_b, c_b, input_lenths, True)
                    
                    output_list.append(outputs_b)
                    target_list.append(targets_b)

            # print(f'output_list: {len(output_list)}')
            # print(f'output_list_item: {output_list[0].size()}')
            output = torch.cat(output_list)
            target = torch.cat(target_list)
            # print(f'output: {output.size()}')
            # print(f'target: {target.size()}')
            test_loss = torch.nn.MSELoss(reduction='sum')(output, target).item()
                    
            output = output.squeeze().detach().cpu()
            target = target.squeeze().detach().cpu()

            test_means_origin = output * dic_op_meanstd['mcpm10'][1] + dic_op_meanstd['mcpm10'][0]
            test_y_origin = target * dic_op_meanstd['mcpm10'][1] + dic_op_meanstd['mcpm10'][0]

            mae = mean_squared_error(test_y_origin, test_means_origin, squared=False)
            r_squared = stats.pearsonr(test_y_origin, test_means_origin)
            print(f'\t\t--------\n\t\tIter: {str(real_iter)}, inter_train_loss: {inter_loss}\n\t\t--------\n')
            print(f'\t\t--------\n\t\ttest_loss: {str(test_loss)}, last best test_loss: {str(best_err)}\n\t\t--------\n')
            print(f'\t\t--------\n\t\tr_squared: {str(r_squared[0])}, MSE: {str(mae)}\n\t\t--------\n')

            list_err.append(float(test_loss))
            list_total.append(float(inter_loss))
            inter_loss = 0

            title = f'Fold{fold}_holdout{holdout}_Md_all: MSE {round(mae, 2)} R2 {round(r_squared[0], 2)}'

            support_functions.save_square_img(
                contents=[test_y_origin.numpy(), test_means_origin.numpy()], 
                xlabel='targets_ex', ylabel='output_ex', 
                savename=os.path.join(coffer_slot, f'test_{real_iter}'),
                title=title
            )
            if best_err - test_loss > settings['es_mindelta']:
                best_err = test_loss
                torch.save(model.state_dict(), coffer_slot + "best_params")
                es_counter = 0
            else:
                es_counter += 1
                print(f"INFO: Early stopping counter {es_counter} of {settings['es_endure']}")
                if es_counter >= settings['es_endure']:
                    print('INFO: Early stopping')
                    es_flag = 1
                    break
                    
#             output_list = []
#             target_list = []
#             test_loss = 0
#             for dataloader_ex in self_dataloaders:
#                 for x_b, c_b, y_b, input_lenths in dataloader_ex:
#                     x_b, c_b, y_b, input_lenths = x_b.to(device), c_b.to(device), y_b.to(device), input_lenths.to(device)
#                     outputs_b, targets_b = model(x_b, y_b, c_b, input_lenths, True)
                    
#                     batch_loss = loss_func(outputs_b, targets_b)
                    
#                     test_loss += batch_loss.item()
#                     output_list.append(outputs_b.detach().cpu())
#                     target_list.append(targets_b.detach().cpu())

#             output = torch.cat(output_list).squeeze()
#             target = torch.cat(target_list).squeeze()

#             test_means_origin = output * dic_op_meanstd['mcpm10'][1] + dic_op_meanstd['mcpm10'][0]
#             test_y_origin = target * dic_op_meanstd['mcpm10'][1] + dic_op_meanstd['mcpm10'][0]

#             mae = mean_squared_error(test_y_origin, test_means_origin, squared=False)
#             r_squared = stats.pearsonr(test_y_origin, test_means_origin)
#             print(f'\t\t--------\n\t\tr_squared: {str(r_squared[0])}, MSE: {str(mae)}\n\t\t--------\n')

#             list_err.append(float(test_loss))
#             list_total.append(float(inter_loss))
#             inter_loss = 0

#             title = f'Fold{fold}_holdout{holdout}_Md_all: MSE {round(mae, 2)} R2 {round(r_squared[0], 2)}'
#             support_functions.save_square_img(
#                 contents=[test_y_origin.numpy(), test_means_origin.numpy()], 
#                 xlabel='targets_ex', ylabel='output_ex', 
#                 savename=os.path.join(coffer_slot, f'self_{real_iter}'),
#                 title=title
#             )
                    
        if real_iter > settings['epoch']:
            break
    
    elapsed_time = time.time() - start_time
    print("Elapsed: "+str(elapsed_time))

    return list_total, list_err


# to evaluate the result of training
def evaluate(settings, job_id):
    support_functions.seed_everything(settings['seed'])
    
    # scan the correct coffer

    # Fold（折叠）：
    #
    # "Fold" 通常指的是在交叉验证（Cross-Validation）中的一个子集数据。交叉验证是一种评估机器学习模型性能的方法，它将数据集分成若干个互不重叠的折叠（folds），然后依次使用这些折叠来训练和验证模型。
    # 例如，5折交叉验证将数据集分成5个折叠，每次使用其中4个折叠来训练模型，然后使用剩下的1个折叠来验证模型。这个过程循环5次，每个折叠都曾被用作验证集。

    # "Fold" 可以表示交叉验证中的一个数据子集，也可以表示折叠的数量。
    # Holdout（保留集）：
    # "Holdout" 是指从数据集中保留一部分数据，不用于训练模型，而是用于评估模型的性能。通常，将数据集划分为训练集（用于训练模型）和测试集（用于评估模型）。
    # 在一些情况下，还可以进一步划分为训练集、验证集和测试集，其中验证集用于调整模型的超参数。
    # "Holdout" 数据集的目的是模拟模型在未见过的数据上的性能，以便评估模型的泛化能力。

    fold = settings['fold']
    holdout = settings['holdout']
    lowest_rank = settings['lowest_rank']
    
    coffer_dir = ''
    dirs = os.listdir(myconfig.coffer_path)
    dirs.sort()
    for dir in dirs:
        if job_id in dir:
            coffer_dir = myconfig.coffer_path + dir + f'/{fold}/'
            break
    
    # Get device setting
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
        
    # build model
    model = PEGCN(num_features_in=settings['num_features_in'], num_features_out=settings['num_features_out'], emb_hidden_dim=settings['emb_hidden_dim'], emb_dim=settings['emb_dim'], k=settings['k'], conv_dim=settings['conv_dim']).to(device)
    model = model.float()
    
    if ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=device_list)
        
    model.load_state_dict(torch.load(coffer_dir + "best_params"))

    # build dataloader
    rtn_mae_list = []
    rtn_rsq_list = []
    for mask_distance in [0, 20, 50]:
        dataset_eval = dl.IntpDataset(settings=settings, mask_distance=mask_distance, call_name='eval')
        dataloader_ev = torch.utils.data.DataLoader(dataset_eval, batch_size=settings['batch'], shuffle=False, collate_fn=dl.collate_fn, num_workers=16, prefetch_factor=32, drop_last=True)

        # Eval batch
        model.eval()
        output_list = []
        target_list = []
        for x_b, c_b, y_b, input_lenths in dataloader_ev:
            x_b, c_b, y_b, input_lenths = x_b.to(device), c_b.to(device), y_b.to(device), input_lenths.to(device)

            outputs_b, targets_b = model(x_b, y_b, c_b, input_lenths, True)
            output_list.append(outputs_b.detach().cpu())
            target_list.append(targets_b.detach().cpu())
                    
        output = torch.cat(output_list).squeeze()
        target = torch.cat(target_list).squeeze()

        test_means_origin = output * dic_op_meanstd['mcpm10'][1] + dic_op_meanstd['mcpm10'][0]
        test_y_origin = target * dic_op_meanstd['mcpm10'][1] + dic_op_meanstd['mcpm10'][0]


        # Mean Absolute Error
        mae = mean_squared_error(test_y_origin, test_means_origin, squared=False)
        r_squared = stats.pearsonr(test_y_origin, test_means_origin)

        rtn_mae_list.append(float(mae))
        rtn_rsq_list.append(float(r_squared[0]))

        print(f'\t\t--------\n\t\tr_squared: {str(r_squared[0])}, MSE: {str(mae)}\n\t\t--------\n')
        print(f'\t\t--------\n\t\tDiffer: {test_means_origin.max() - test_means_origin.min()}, count: {test_y_origin.size(0)}\n\t\t--------\n')

        title = f'Fold{fold}_holdout{holdout}_Md{mask_distance}: MSE {round(mae, 2)} R2 {round(r_squared[0], 2)}'
        support_functions.save_square_img(
            contents=[test_y_origin.numpy(), test_means_origin.numpy()], 
            xlabel='targets_ex', ylabel='output_ex', 
            savename=os.path.join(coffer_dir, f'result_{mask_distance}'),
            title=title
        )
        targets_ex = test_y_origin.unsqueeze(1)
        output_ex = test_means_origin.unsqueeze(1)
        diff_ex = targets_ex - output_ex
        pd_out = pd.DataFrame(
            torch.cat(
                (targets_ex, output_ex, diff_ex), 1
            ).numpy()
        )
        pd_out.columns = ['Target', 'Output', 'Diff']
        pd_out.to_csv(os.path.join(coffer_dir, f'result_{mask_distance}.csv'), index=False)

    return rtn_mae_list, rtn_rsq_list
