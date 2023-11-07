# wandb api key
# §§
api_key = '50d21c78c93763fe3ab0c50a896d798f9a8e7d0a'

# is it a new run or continue
new_run = False
sweep_id = 'h8lygcm6'

# where should the intermedia generated scripts be saved (automatically cleaned at the start of each run)
slurm_scripts_path = './slurm_scripts/'
# where should the outputs & logs be saved (automatically cleaned at the start of each run)
log_path = './logs/'
# where should calculation nodes save their important results (e.g. best model weights)
coffer_path = './coffer/'

# entity name (your wandb account name)
# §§
entity_name = 'chaofan'
# project name on wandb and HPC
project_name = 'PEGNN_sweep_ber'
# e-mail address to recieve notifications
e_mail = 'hpcdam.lcf@gmail.com'
# conda location
conda_env = '/home/kit/tm/lm6999/miniconda3/envs/new_env'
# file name of the slurm_wrapper, don't change this if you haven't write a new one
slurm_wrapper_name = './slurm_wrapper.py'
# file name of the training code
train_script_name = './Trainer.py'


# define custom sweep hyperparameters
#     - how many sweeps do you want to run in total
total_sweep = 200
#     - how many sweeps do you want to run parallelly
pool_size = 40


# define wandb sweep parameters
#     - project definition
sweep_config = {
    "project": project_name,
    'program': slurm_wrapper_name,
    "name": "offline-sweep",
    'method': 'random',
}
#     - metric definition
metric = {
    'name': 'best_err',
    'goal': 'minimize'   
}
sweep_config['metric'] = metric
#     - parameters search range definition
parameters_dict = {
    'full_batch': {
        'values': [128, 256, 512]
    },
    'lr': {
        'values': [1e-3, 1e-4, 1e-5]
    },
    'emb_dim': {
        'values': [16, 32, 64]
    },
    'k': {
        'values': [5, 10, 20]
    },
    'conv_dim': {
        'values': [128, 256, 512]
    },
}
sweep_config['parameters'] = parameters_dict
