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

    job_id = sys.argv[1]
    config = json.loads(sys.argv[2])
    agent_id = sys.argv[3]
    agent_dir = sys.argv[4]

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
    