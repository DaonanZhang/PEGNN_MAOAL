import os
import sys

import support_functions

sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "PEGNN"))
import json
import time
import myconfig
import solver as solver
from datetime import datetime


def make_dir(path):
    try:
        os.mkdir(path)
    except:
        pass


# rebuild the folder missed?
def build_folder_and_clean(path):
    check = os.path.exists(path)
    if check:
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    else:
        os.makedirs(path)


def train(job_id, settings):
    result_sheet = []

    print("Start training...")
    list_total, list_err = solver.training(settings=settings, job_id=job_id)
    print("Start evaluation...")
    best_err, r_squared = solver.evaluate(settings=settings, job_id=job_id)

    result_sheet.append([list_total, list_err, best_err, r_squared])

    # collect wandb result into file
    rtn = {
        "best_err": sum(result_sheet[0][2]) / len(result_sheet[0][2]),
        "r_squared": sum(result_sheet[0][3]) / len(result_sheet[0][3]),
        "list_total_0": result_sheet[0][0],
        "list_err_0": result_sheet[0][1],
    }

    json_dump = json.dumps(rtn)
    with open(settings['agent_dir'] + f'/{job_id}.rtn', 'w') as fresult:
        fresult.write(json_dump)



# RuntimeError: mat1 and mat2 shapes cannot be multiplied (2974x42 and 46x256)
# problem for number of the dataset_size since i change the size into minimal size
# but if this problem occurs in ssh server then means all right
if __name__ == '__main__':
    job_id = '123456'

    settings = {
        'agent_id': '00000',
        'agent_dir': './logs',
        'origin_path': 'Dataset_res250/',

        # debug mode=>data_set
        'debug': True,
        'bp': False,


        'batch': 8,
        'full_batch': 32,
        'accumulation_steps': 32 // 8,

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
        'd_model': 256,
        'nhead': 8,
        'dim_feedforward': 1024,
        'transformer_dropout': 0.1,
        'num_encoder_layers': 3,

        'aux_task_num': 3,

        'aux_op_dic': {'mcpm1': 0, 'mcpm2p5': 1, 'mcpm4': 2},

        'env_op_dic': {'ta': 0, 'hur': 1, 'plev': 2, 'precip': 3, 'wsx': 4, 'wsy': 5, 'globalrad': 6},

        'hyper': {'lr': 0.001, 'decay': 0.0, 'pre': 50, 'interval': 10,'aux_loss_weight': 0.001},

        'heads': {'nn_length': 3, 'nn_hidden_dim': 64, 'dropout_rate': 0.1},
    }

    # build working folder
    dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    coffer_slot = myconfig.coffer_path + str(job_id) + '/'

    # missed
    make_dir(coffer_slot)

    build_folder_and_clean(coffer_slot)

    settings['coffer_slot'] = coffer_slot


    if 'Dataset_res250' in settings['origin_path'] or 'LUFT_res250' in settings['origin_path']:
        settings['tgt_op'] = 'mcpm10'
    elif 'ERA_res250' in settings['origin_path']:
        settings['tgt_op'] = 'soil_temperature_level_1'

    train(job_id, settings)
