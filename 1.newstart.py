import wandb
import yaml
import os
import time
from multiprocessing import Process
import subprocess
import myconfig


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


# define & save the yaml file for wandb sweep
def make_para_yaml(project_name, slurm_wrapper, scripts_path, sweep_config):
    # make the yaml file
    with open(scripts_path + 'sweep_params.yaml', 'w') as outfile:
        yaml.dump(sweep_config, outfile, default_flow_style=False)
        

def agent_wrapper(sweep_id, count):
    arg = {'sweep_id': sweep_id, 'count': count}
    return wandb.agent(**arg)


if __name__ == "__main__":
    # login to wandb
    wandb.login(key=myconfig.api_key)

    # start sweep
    # "Wandb sweep" 是指 Weights & Biases（wandb）
    # 平台提供的一种功能，用于进行超参数优化和实验调优。它允许你定义一个参数搜索空间，然后自动执行多次训练，并跟踪不同超参数组合的性能，以找到最佳的超参数配置。
    #    - if new_run, then prepare working folders, generate & load sweep config, start sweep
    if myconfig.new_run:
        # prepare working folders
        # build_folder_and_clean('./wandb/')
        build_folder_and_clean(myconfig.slurm_scripts_path)
        build_folder_and_clean(myconfig.log_path)
        build_folder_and_clean(myconfig.coffer_path)
        # generate & load sweep config
        make_para_yaml(myconfig.project_name, myconfig.slurm_wrapper_name, myconfig.slurm_scripts_path, myconfig.sweep_config)
        with open(myconfig.slurm_scripts_path + 'sweep_params.yaml') as file:
            config_dict = yaml.load(file, Loader=yaml.FullLoader)
        # start sweep
        sweep_id = wandb.sweep(sweep=config_dict)
        wandb.agent(sweep_id=sweep_id, count=myconfig.total_sweep)
    else:
        sweep_id = myconfig.sweep_id
        # direcutoray
        wandb.agent(sweep_id=sweep_id, entity=myconfig.entity_name, project=myconfig.project_name, count=myconfig.total_sweep)
    
