import wandb
import subprocess
import os
import json
import time
import myconfig
from datetime import datetime


# check echo from 'sacct' to tell the job status
def check_status(status):
    rtn = 'RUNNING'
    
    lines = status.split('\n')
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        if 'FAILED' in line:
            rtn = 'FAILED'
            break
        elif 'COMPLETED' not in line:
            rtn = 'PENDING'
            break
    else:
        rtn = 'COMPLETED'
        
    return rtn


def wrap_task(config=None):
    # recieve config for this run from Sweep Controller

    with wandb.init(config=config):
        agent_id = wandb.run.id
        agent_dir = wandb.run.dir
        config = dict(wandb.config)

        # §§ DB folder
        config['origin_path'] = './Dataset_res250/'

        config['debug'] = False
        config['bp'] = False
        
        config['batch'] = 64

        config['accumulation_steps'] = config['full_batch'] // config['batch']
        config['epoch'] = 10000
        config['test_batch'] = 50

        config['es_mindelta'] = 0.5
        config['es_endure'] = 30
        config['num_features_in'] = 2
        config['nn_lr'] = config['lr']

        # -------------Best HP after first sweep----------------
        config['full_batch'] = 128
        config['k'] = 20
        config['lr'] = 1e-5
        config['emb_dim'] = 16
        config['conv_dim'] = 256


        config['num_features_out'] = 1
        config['emb_hidden_dim'] = config['emb_dim'] * 4
        
        config['seed'] = 1
        config['model'] = 'PEGNN'

        config['fold'] = 0
        config['holdout'] = [0, 1]
        config['lowest_rank'] = 1

        # for hyperparameter tuning
        config['hp_marker'] = 'tuned'
        config['nn_length'] = 3
        config['nn_hidden_dim'] = 32
        config['dropout_rate'] = 0.1
        config['aux_task_num'] = 3

        # -------------HP for Transformer----------------
        config['d_model'] = 256
        config['nhead'] = 4
        config['dim_feedforward'] = 1024
        config['transformer_dropout'] = 0.1

            # -------------Need HP optimization----------------
        config['num_encoder_layers'] = 3
        config['num_MPL'] = 3

        # -------------nn_length and nn_hidden_dim Need HP optimization----------------
        config['heads'] =  {'nn_length': 3, 'nn_hidden_dim': 64, 'dropout_rate': 0.1},



        config['aux_op_dic'] =  {'mcpm1': 0, 'mcpm2p5': 1, 'mcpm4': 2}
        config['env_op_dic'] = {'ta': 0, 'hur': 1, 'plev': 2, 'precip': 3, 'wsx': 4, 'wsy': 5, 'globalrad': 6}


        # -------------Keep as original paper-------------
        config['hyper'] = {'lr': 0.001, 'decay': 0.0, 'pre': 0, 'interval': 10,'aux_loss_weight': 0.01}




        # §§ slurm command: squeue
        while True:
            cmd = f"squeue -n {myconfig.project_name}"
            status = subprocess.check_output(cmd, shell=True).decode()
            lines = status.split('\n')[1:-1]
            if len(lines) <= myconfig.pool_size:
                break
            else:
                time.sleep(60)

        # partition gpu_4 => dev_gpu_4
        # time = 24:00:00 => 00:30:00
        # then build up the slurm script

        job_script = \
f"""#!/bin/bash
#SBATCH --job-name={myconfig.project_name}
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:4
#SBATCH --error={myconfig.log_path}%x.%j.err
#SBATCH --output={myconfig.log_path}%x.%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={myconfig.e_mail}
#SBATCH --export=ALL
#SBATCH --time=24:00:00

eval \"$(conda shell.bash hook)\"
conda activate {myconfig.conda_env}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{myconfig.conda_env}/lib

job_id=$SLURM_JOB_ID
python {myconfig.train_script_name} $job_id '{json.dumps(config)}' {agent_id} {agent_dir}
"""


        # wandb config agent id agent dir
        
        # Write job submission script to a file
        # change to windows cmd
        with open(myconfig.slurm_scripts_path + f"{wandb.run.id}.sbatch", "w") as f:
            f.write(job_script)

        # current_direcorty =os.getcwd()
        # slurm_scripts_diretocry = os.path.join(current_direcorty, myconfig.slurm_scripts_path)
        # with open( slurm_scripts_diretocry + f"{wandb.run.id}.cmd", "w") as f:
        #     f.write(job_script)
        
        # Submit job to Slurm system and get job ID

        # change script to windows cmd
        cmd = "sbatch " + myconfig.slurm_scripts_path + f"{wandb.run.id}.sbatch"
        # cmd = slurm_scripts_diretocry + f"{wandb.run.id}.cmd"

        # change to windows cmd
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        # subprocess.run([cmd] , shell=True)

        # close for now
        job_id = output.split()[-1]


        wandb.log({
            "job_id" : job_id,
        })
        return job_id
        
           
if __name__ == '__main__':
    rtn = wrap_task()
    print(f'******************************************************* Process Finished with code {rtn}')
    wandb.finish()
