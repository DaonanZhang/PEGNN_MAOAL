import os
import sys
import json
import time

import wandb

import myconfig
import solver as solver
from datetime import datetime

import sys
sys.path.append('/pfs/data5/home/kit/tm/lm6999/GPR_INTP_SAQN/util')
import support_functions


# just the entry point for the training process, more details see in solver
def train(job_id, settings):
    result_sheet = []
    list_total, list_err = solver.training(settings=settings, job_id=job_id)
    best_err, r_squared = solver.evaluate(settings=settings, job_id=job_id)
    result_sheet.append([list_total, list_err, best_err, r_squared])
    
    # collect wandb result into file
    rtn = {
        "best_err": sum(result_sheet[0][2])/len(result_sheet[0][2]), 
        "r_squared": sum(result_sheet[0][3])/len(result_sheet[0][3]), 
        "list_total_0": result_sheet[0][0],
        "list_err_0": result_sheet[0][1],
    }
    json_dump = json.dumps(rtn)
    with open(settings['agent_dir'] + f'/{job_id}.rtn', 'w') as fresult:
        fresult.write(json_dump)


if __name__ == '__main__':

    # 主函数入口


    # @@@@
    # job_id = sys.argv[1]
    # config = json.loads(sys.argv[2])
    # agent_id = sys.argv[3]
    # agent_dir = sys.argv[4]

    job_id = 'only_for_test'
    with wandb.init(config=None):
        agent_id = wandb.run.id
        agent_dir = wandb.run.dir
        config = dict(wandb.config)

        # §§ DB folder
        # config['origin_path'] = '../Datasets/LUFT_res250/'
        config['origin_path'] = '../Dataset_res250/'

        config['debug'] = False
        config['bp'] = False

        config['batch'] = 64
        # §§
        # can't load config['full_batch'] lr and so on, nor if i run the new start first, what should i do then
        # §§
        config['accumulation_steps'] = 128 // 64
        config['epoch'] = 10000
        config['test_batch'] = 50
        # §§
        config['nn_lr'] = 0.01
        config['es_mindelta'] = 0.5
        config['es_endure'] = 30

        config['num_features_in'] = 10
        config['num_features_out'] = 1
        # §§
        config['emb_hidden_dim'] = 64 * 4

        config['seed'] = 1
        config['model'] = 'PEGNN'
        config['fold'] = 0
        config['holdout'] = [0, 1]
        config['lowest_rank'] = 1

        # can't find config['full_batch'] where should it comes from
        print("--------------" + config['full_batch'] + "------------------")

    settings = {
        'agent_id': agent_id,
        'agent_dir': agent_dir,
    }
    
    settings.update(config)

    # build working folder
    dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    coffer_slot = myconfig.coffer_path + str(job_id) + '-' + dt_string + '/'
    support_functions.make_dir(coffer_slot)
    settings['coffer_slot'] = coffer_slot
    
    train(job_id, settings)
    