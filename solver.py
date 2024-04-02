import os, glob, inspect, time, math, torch, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from spatial_utils import MaskedMSELoss
import psutil

# in the slurm file
# sys.path.append('/pfs/data5/home/kit/tm/lm6999/GPR_INTP_SAQN/Datasets')


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
import tqdm

import sys

# in the slurm file
sys.path.append('/pfs/data5/home/kit/tm/lm6999/GPR_INTP_SAQN/util')

import support_functions



def map_param_to_block(shared_params):
    param_to_block = {}

    for i, (name, param) in enumerate(shared_params):
        if 'spenc' or 'dec' in name:
            param_to_block[i] = 0
        elif 'transformer' in name:
            param_to_block[i] = 1
        elif 'conv1' in name:
            param_to_block[i] = 2
        elif 'conv2' in name:
            param_to_block[i] = 3
    module_num = 4
    return param_to_block, module_num

class hypermodel(nn.Module):
    def __init__(self, task_num, module_num, param_to_block):

        super(hypermodel, self).__init__()
        self.task_num = task_num
        self.module_num = module_num
        self.param_to_block = param_to_block
        self.modularized_lr = nn.Parameter(torch.ones(task_num, module_num))
        self.nonlinear = nn.ReLU()
        self.scale_factor = 1.0

    def forward(self, loss_vector, shared_params, whether_single=1, train_lr=1.0):
        if whether_single == 1:
            grads = torch.autograd.grad(loss_vector[0], shared_params, create_graph=True)
            if self.nonlinear is not None:
                grads = tuple(self.nonlinear(self.modularized_lr[0][self.param_to_block[m]]) * g * train_lr for m, g in
                              enumerate(grads))
            else:
                grads = tuple(
                    self.modularized_lr[0][self.param_to_block[m]] * g * train_lr for m, g in enumerate(grads))
            return grads
        else:
            # main target loss and grad

            grads = torch.autograd.grad(loss_vector[0], shared_params, create_graph=True)
            loss_num = len(loss_vector)
            for task_id in range(1, loss_num):
                try:
                    loss_value = loss_vector[task_id].item()
                    zero_grads = [torch.zeros_like(param) for param in shared_params]
                    if loss_value == 0.0:
                        aux_grads = zero_grads
                    else:
                        aux_grads = torch.autograd.grad(loss_vector[task_id], shared_params, create_graph=True)
                    # aux_grads = torch.autograd.grad(loss_vector[task_id], shared_params, create_graph=True)
                    if self.nonlinear is not None:
                        # tupel with len: len(grads, aux_grads)
                        # g:grads, g_aux:aux_grads, m:index in zip()--len(grads)
                        grads = tuple((g + self.scale_factor * self.nonlinear(
                            self.modularized_lr[task_id - 1][self.param_to_block[m]]) * g_aux) * train_lr for m, (g, g_aux)
                                      in enumerate(zip(grads, aux_grads)))
                    else:
                        grads = tuple((g + self.scale_factor * self.modularized_lr[task_id - 1][
                            self.param_to_block[m]] * g_aux) * train_lr for m, (g, g_aux) in
                                      enumerate(zip(grads, aux_grads)))
                except Exception as e:
                    print(f'Error: {e}')
                    print(f'{[loss_vector[id] for id in range(1, loss_num)]}')
                    sys.exit()
            return grads

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



from gauxlearn.optim import MetaOptimizer

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

    # get standarization restore info
    with open(settings['origin_path'] + f'Folds_Info/norm_{fold}.info', 'rb') as f:
        dic_op_minmax, dic_op_meanstd = pickle.load(f)

    # build dataloader
    dataset_train = dl.IntpDataset(settings=settings, mask_distance=-1, call_name='train')
    dataloader_tr = torch.utils.data.DataLoader(dataset_train, batch_size=settings['batch'], shuffle=True, collate_fn=dl.collate_fn, num_workers = settings['num_workers'], prefetch_factor=32, drop_last=True)

    dataset_train2 = dl.IntpDataset(settings=settings, mask_distance=-1, call_name='train')
    dataloader_tr2 = torch.utils.data.DataLoader(dataset_train2, batch_size=settings['batch'], shuffle=True,
                                                collate_fn=dl.collate_fn, num_workers = settings['num_workers'], prefetch_factor=32,
                                                drop_last=True)

    aux_loader_train = dl.IntpDataset(settings=settings, mask_distance=-1, call_name='aux')
    aux_loader_tr = torch.utils.data.DataLoader(aux_loader_train, batch_size=settings['batch'], shuffle=True,
                                                collate_fn=dl.collate_fn, num_workers = settings['num_workers'], prefetch_factor=32,
                                                drop_last=True)

    test_dataloaders = []
    for mask_distance in [0, 20, 50]:
        test_dataloaders.append(
            torch.utils.data.DataLoader(
                dl.IntpDataset(settings=settings, mask_distance=mask_distance, call_name='test'),
                batch_size=settings['batch'], shuffle=False, collate_fn=dl.collate_fn, num_workers = settings['num_workers'], prefetch_factor=32, drop_last=True
            )
        )

    # build model
    model = PEGCN(num_features_in=settings['num_features_in'], num_features_out=settings['num_features_out'],
                  emb_hidden_dim=settings['emb_hidden_dim'],emb_dim=settings['emb_dim'], k=settings['k'],
                  conv_dim=settings['conv_dim'],aux_task_num=settings['aux_task_num'], settings=settings).to(device)
    model = model.float()

    # Added part MAOAL
    # build hypermodel
    shared_parameters = [param for name, param in model.named_parameters() if 'task' not in name]

    shared_parameters1 = [(name,param) for name, param in model.named_parameters() if 'task' not in name]

    for name, param in shared_parameters1:
        print(f'name: {name}, param: {param.shape}')


    param_to_block, module_num = map_param_to_block(shared_parameters1)

    modular = hypermodel(settings['aux_task_num'], module_num, param_to_block).to(device)


    # loss_func = torch.nn.MSELoss()

    loss_func = MaskedMSELoss()

    if ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=device_list)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=settings['nn_lr'])

    m_optimizer = optim.SGD( modular.parameters(), lr = settings['hyper']['lr'], momentum = 0.9, weight_decay = settings['hyper']['decay'] )
    meta_optimizer = MetaOptimizer(meta_optimizer= m_optimizer, hpo_lr = 1.0, truncate_iter = 3, max_grad_norm = 10)
    modular = modular.to(device)

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
    # -----------------init loss-----------------
    # inter_loss = 0
    # mini_loss = 0
    inter_loss = torch.tensor(0., device=device)
    mini_loss = torch.tensor(0., device=device)

    # aux_iter_loss = [0] * settings['aux_task_num']
    # aux_mini_loss = [0] * settings['aux_task_num']

    aux_iter_loss = [torch.tensor(0., device=device) for _ in range(settings['aux_task_num'])]
    aux_mini_loss = [torch.tensor(0., device=device) for _ in range(settings['aux_task_num'])]

    data_iter = iter(dataloader_tr)
    data_iter2 = iter(dataloader_tr2)
    aux_iter = iter(aux_loader_tr)

    t_train_iter_start = time.time()


    print("working on training loop")

    for i in range(epochs):
        # train 1 iteration
        model.train()
        real_iter = iter_counter//settings['accumulation_steps'] + 1

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader_tr)
            batch = next(data_iter)

        x_b, c_b, y_b, aux_y_b, input_lengths, rest_features = batch

        x_b, c_b, y_b, input_lengths, rest_features = x_b.to(device), c_b.to(device), \
                                                               y_b.to(device), \
                                                               input_lengths.to(device), \
                                                               rest_features.to(device)

        aux_y_b = [item.to(device) for item in aux_y_b]

        # forward propagation
        outputs_b, targets_b, aux_outputs_b, aux_targets_b = model(inputs = x_b, targets = y_b, coords = c_b, input_lengths = input_lengths,
                                     rest_features = rest_features, head_only= True, aux_answers = aux_y_b)

        # print(f'outputs_b:{outputs_b.shape}')
        # print(f'targets_b:{targets_b.shape}')
        # outputs_b: torch.Size([32, 1])
        # targets_b: torch.Size([32, 1])

        # for i in range(3):
        #     print(f'aux_outputs_b:{outputs_b[i].shape}')
        #     print(f'aux_targets_b:{targets_b[i].shape}')

        # 0: aux_outputs_b:torch.Size([1])
        # 0: aux_targets_b:torch.Size([1])
        primary_loss = loss_func(outputs_b, targets_b)
        primary_loss /= settings['accumulation_steps']

        inter_loss += primary_loss
        mini_loss += primary_loss

        # # -----------------aux_loss-----------------
        aux_loss_list = []
        for aux_output, aux_target in zip(aux_outputs_b, aux_targets_b):
            aux_loss = loss_func(aux_output, aux_target)
            aux_loss_list.append((aux_loss * settings['hyper']['aux_loss_weight'] / settings['accumulation_steps']))

        for i in range(0 ,settings['aux_task_num']):
            # aux_iter_loss[i] += aux_loss_list[i].item()
            # aux_mini_loss[i] += aux_loss_list[i].item()
            aux_iter_loss[i] += aux_loss_list[i]
            aux_mini_loss[i] += aux_loss_list[i]

        # -----------------a full_batch, opt.step optimize the shared parameters only-----------------
        if (iter_counter+1) % settings['accumulation_steps'] == 0:

            # -----------------MAOAL-----------------
            loss_list = [mini_loss] + aux_mini_loss
            common_grads = modular(loss_list, shared_parameters, whether_single=0, train_lr=1.0)
            loss_vec = torch.stack(loss_list)
            total_loss = torch.sum(loss_vec)
            optimizer.zero_grad()
            total_loss.backward()

            for p, g in zip(shared_parameters, common_grads):
                p.grad = g


            # TODO plot grad flow
            
            optimizer.step()
            del common_grads

            # -----------------log-----------------
            t_train_iter_end = time.time()
            print(f'\tIter {real_iter} - Loss: {mini_loss.item()} Aux_loss:{[loss.item() for loss in aux_mini_loss]} - real_iter_time: {t_train_iter_end - t_train_iter_start}', end="\r", flush=True)

            # -----------------reset-----------------
            mini_loss = torch.tensor(0., device=device)
            aux_mini_loss = [torch.tensor(0., device=device) for _ in range(settings['aux_task_num'])]
            t_train_iter_start = t_train_iter_end

            # -----------------optimize the Hyper Parameters-----------------
        if (real_iter) % settings['hyper']['interval'] == 0 and real_iter > settings['hyper']['pre']:

            try:
                aux_batch = next(aux_iter)
            except StopIteration:
                aux_iter = iter(aux_loader_tr)
                aux_batch = next(aux_iter)

            try:
                train_batch2 = next(data_iter2)
            except StopIteration:
                data_iter2 = iter(dataloader_tr2)
                train_batch2 = next(data_iter2)

            # ----------------------------meta_loss----------------------------
            meta_x_b, meta_c_b, meta_y_b, meta_aux_y_b, meta_input_lengths, meta_rest_features = aux_batch
            meta_x_b, meta_c_b, meta_y_b, meta_input_lengths, meta_rest_features = meta_x_b.to(device), meta_c_b.to(
                device), meta_y_b.to(device), \
                                                                                   meta_input_lengths.to(
                                                                                       device), meta_rest_features.to(
                device)

            meta_aux_y_b = [item.to(device) for item in meta_aux_y_b]

            meta_outputs_b, meta_targets_b, _, _ = model(inputs=meta_x_b, targets=meta_y_b, coords=meta_c_b,
                                                         input_lengths=meta_input_lengths,
                                                         rest_features=meta_rest_features, head_only=True,
                                                         aux_answers=meta_aux_y_b)

            meta_primary_loss = loss_func(meta_outputs_b, meta_targets_b)
            meta_primary_loss /= settings['accumulation_steps']
            meta_total_loss = meta_primary_loss

            # ----------------------------train_loss2----------------------------

            train_x_b, train_c_b, train_y_b, train_aux_y_b, train_input_lengths, train_rest_features = train_batch2

            train_x_b, train_c_b, train_y_b, train_input_lengths, train_rest_features = train_x_b.to(
                device), train_c_b.to(device), \
                                                                                        train_y_b.to(device), \
                                                                                        train_input_lengths.to(device), \
                                                                                        train_rest_features.to(device)

            train_aux_y_b = [item.to(device) for item in train_aux_y_b]

            outputs_b, targets_b, aux_outputs_b, aux_targets_b = model(inputs=train_x_b, targets=train_y_b,
                                                                       coords=train_c_b,
                                                                       input_lengths=train_input_lengths,
                                                                       rest_features=train_rest_features,
                                                                       head_only=True,
                                                                       aux_answers=train_aux_y_b)

            primary_loss = loss_func(outputs_b, targets_b)
            primary_loss /= settings['accumulation_steps']

            # # -----------------aux_loss-----------------
            aux_loss_list = []
            for aux_output, aux_target in zip(aux_outputs_b, aux_targets_b):
                aux_loss = loss_func(aux_output, aux_target)
                aux_loss_list.append(aux_loss * settings['hyper']['aux_loss_weight'] / settings['accumulation_steps'])

            loss_list = [primary_loss] + aux_loss_list
            train_common_grads = modular(loss_list, shared_parameters, whether_single=0, train_lr=1.0)

            meta_optimizer.step(val_loss=meta_total_loss, train_grads=train_common_grads,
                                aux_params=list(modular.parameters()), shared_parameters=shared_parameters)

            # -----------------log-----------------
            print(f'\tIter {real_iter} - Meta_Loss: {meta_total_loss.item()} - Main_loss: {primary_loss.item()}',
                  end="\r", flush=True)

        iter_counter += 1


        # -----------------test batch-----------------
        print((iter_counter+1) % (settings['test_batch'] * settings['accumulation_steps']))
        # only log for after finish a full_batch
        if (iter_counter+1) % (settings['test_batch'] * settings['accumulation_steps']) == 0:
            model.eval()
            output_list = []
            target_list = []
            test_loss = torch.tensor(0., device=device)

            for dataloader_ex in test_dataloaders:

                for test_batch in dataloader_ex:

                    num_open_files_in_test = psutil.Process().num_fds()
                    print(f"current open files {num_open_files_in_test} in test")
                    sleep_for_open_too_many_files()

                    x_b, c_b, y_b, aux_y_b, input_lengths, rest_features = test_batch

                    x_b, c_b, y_b, input_lengths, rest_features = x_b.to(device), c_b.to(device), y_b.to(
                        device), input_lengths.to(device), rest_features.to(device)

                    aux_y_b = [item.to(device) for item in aux_y_b]

                    outputs_b, targets_b, _, _ = model(inputs=x_b, targets=y_b, coords=c_b,
                                                                       input_lengths=input_lengths,
                                                                       rest_features=rest_features, head_only=True,
                                                                       aux_answers=aux_y_b)
                    output_list.append(outputs_b)
                    target_list.append(targets_b)

            output = torch.cat(output_list)
            target = torch.cat(target_list)
            # ___________________reset_______________________
            test_loss = torch.nn.MSELoss(reduction='sum')(output, target)

            output = output.squeeze().detach().cpu()
            target = target.squeeze().detach().cpu()

            # -----------------restore result-----------------
            min_val = dic_op_minmax['mcpm10'][0]
            max_val = dic_op_minmax['mcpm10'][1]
            test_means_origin = output * (max_val - min_val) + min_val
            test_y_origin = target * (max_val - min_val) + min_val

            mae = mean_squared_error(test_y_origin, test_means_origin, squared=False)
            r_squared = stats.pearsonr(test_y_origin, test_means_origin)
            print(f'\t\t--------\n\t\tIter: {str(real_iter)}, inter_train_loss: {inter_loss}\n\t\t--------\n')
            print(f'\t\t--------\n\t\ttest_loss: {str(test_loss.item())}, last best test_loss: {str(best_err)}\n\t\t--------\n')
            print(f'\t\t--------\n\t\tr_squared: {str(r_squared[0])}, MSE: {str(mae)}\n\t\t--------\n')

            list_err.append(float(test_loss.item()))
            list_total.append(float(inter_loss.item()))
            inter_loss = torch.tensor(0., device=device)

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

        print("one epoch finished")
        if real_iter > settings['epoch']:
            break

    elapsed_time = time.time() - start_time
    print("Elapsed: "+str(elapsed_time))

    return list_total, list_err

def sleep_for_open_too_many_files():
    while True:
            num_open_files = psutil.Process().num_fds()
            if num_open_files < 300:
                break
            time.sleep(10)


# to evaluate the result of training
def evaluate(settings, job_id):
    support_functions.seed_everything(settings['seed'])
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
    model = PEGCN(num_features_in=settings['num_features_in'], num_features_out=settings['num_features_out'],
                  emb_hidden_dim=settings['emb_hidden_dim'], emb_dim=settings['emb_dim'], k=settings['k'],
                  conv_dim=settings['conv_dim'], aux_task_num= 0, settings=settings).to(device)

    model = model.float()
    
    if ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=device_list)

    model.load_state_dict(torch.load(coffer_dir + "best_params"))


    # build dataloader
    rtn_mae_list = []
    rtn_rsq_list = []

    for mask_distance in [0, 20, 50]:
        dataset_eval = dl.IntpDataset(settings=settings, mask_distance=mask_distance, call_name='eval')

        dataloader_ev = torch.utils.data.DataLoader(dataset_eval, batch_size=settings['batch'], shuffle=False, collate_fn=dl.collate_fn, num_workers = settings['num_workers'], prefetch_factor=32, drop_last=True)

        # Eval batch, where the print comes from
        model.eval()
        output_list = []
        target_list = []
        for x_b, c_b, y_b, aux_y_b, input_lengths, rest_features in dataloader_ev:
            x_b, c_b, y_b, input_lengths, rest_features = x_b.to(device), c_b.to(device), y_b.to(
                device), input_lengths.to(device), rest_features.to(device)

            aux_y_b = [item.to(device) for item in aux_y_b]

            # forward propagation
            outputs_b, targets_b, _, _ = model(inputs=x_b, targets=y_b, coords=c_b,
                                                                       input_lengths=input_lengths,
                                                                       rest_features=rest_features, head_only=True,
                                                                       aux_answers=aux_y_b)

            output_list.append(outputs_b.detach().cpu())
            target_list.append(targets_b.detach().cpu())
                    
        output = torch.cat(output_list).squeeze()
        target = torch.cat(target_list).squeeze()

        # -----------------restore result-----------------
        min_val = dic_op_minmax['mcpm10'][0]
        max_val = dic_op_minmax['mcpm10'][1]
        test_means_origin = output * (max_val - min_val) + min_val
        test_y_origin = target * (max_val - min_val) + min_val

        # -----------------Mean Absolute Error-----------------
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
