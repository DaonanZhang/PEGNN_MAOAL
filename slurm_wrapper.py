import wandb
import subprocess
import os
import json
import time
import myconfig
from datetime import datetime


# Slurm（Simple Linux Utility for Resource Management）是一个用于高性能计算（HPC）和超级计算机集群管理的开源作业调度系统。
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
        
        config['origin_path'] = '../Datasets/LUFT_res250/'
        config['debug'] = False
        config['bp'] = False
        
        config['batch'] = 64
        config['accumulation_steps'] = config['full_batch'] // 64
        config['epoch'] = 10000
        config['test_batch'] = 50
        config['nn_lr'] = config['lr']
        config['es_mindelta'] = 0.5
        config['es_endure'] = 30

        config['num_features_in'] = 10
        config['num_features_out'] = 1
        config['emb_hidden_dim'] = config['emb_dim'] * 4
        
        config['seed'] = 1
        config['model'] = 'PEGNN'
        config['fold'] = 0
        config['holdout'] = [0, 1]
        config['lowest_rank'] = 1
        
        # wait until available pipe slot
        while True:
            cmd = f"squeue -n {myconfig.project_name}"
            status = subprocess.check_output(cmd, shell=True).decode()
            lines = status.split('\n')[1:-1]
            if len(lines) <= myconfig.pool_size:
                break
            else:
                time.sleep(60)
        
        # then build up the slurm script
        job_script = \
f"""#!/bin/bash
#SBATCH --job-name={myconfig.project_name}
#SBATCH --partition=sdil
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
        
        # Write job submission script to a file
        with open(myconfig.slurm_scripts_path + f"{wandb.run.id}.sbatch", "w") as f:
            f.write(job_script)
        
        # Submit job to Slurm system and get job ID
        cmd = "sbatch " + myconfig.slurm_scripts_path + f"{wandb.run.id}.sbatch"
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        job_id = output.split()[-1]
        wandb.log({
            "job_id" : job_id,
        })
        return job_id
        
           
if __name__ == '__main__':
    rtn = wrap_task()
    print(f'******************************************************* Process Finished with code {rtn}')
    wandb.finish()
