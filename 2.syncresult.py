import wandb
import os
import json
import time
import myconfig
import subprocess

# get result from slurm and sync to wandb


def collect_slurm_id(path):
    files = os.listdir(path)
    for file in files:
        if '.rtn' in file:
            return file, file.split('.')[0]
    else:
        return 'empty', '0'


if __name__ == "__main__":
    # login to wandb
    wandb.login(key=myconfig.api_key)
    wandb_path = './wandb/'

    # 1.time syn?
    #
    first_start = True
    #
    processing = False
    
    if first_start:
        folder_list = os.listdir(wandb_path)
        synced_list = []
        with open('./sync_list.log', 'w') as fresult:
            fresult.write(json.dumps(synced_list))
    else:
        folder_list = os.listdir(wandb_path)
        with open('./sync_list.log', "r") as f:
            result = f.read()
            synced_list = json.loads(result)
    
    for folder in folder_list:
        if folder in synced_list:
            print(f'Already Synced, skipping: {folder}')
            continue
        if 'run-' not in folder:
            print(f'Not a run, skipping: {folder}')
            continue
        
        run_id = folder.split('-')[-1]
        output_folder = wandb_path + folder + '/files/'
        rtn_file, slurm_id = collect_slurm_id(output_folder)


        if rtn_file == 'empty':
            if not processing:
                subprocess.run(["rm", "-rf", f"{folder}"]) 
            print(f'Empty run, skipping: {folder}')
            continue
        
        # the rest is a run folder, not empty and not yet in synced_list
        try:
            wandb.init(project=myconfig.project_name, id=run_id, resume="allow")
        except Exception as e:
            print(f"Init failed, skipping: {run_id}")
            continue
            
        config = dict(wandb.config)
        with open(output_folder + rtn_file, "r") as f:
            result = f.read()
            rtn_dict = json.loads(result)

        # Print calculation result
        print(f"Calculation result: {rtn_dict['best_err']}, {rtn_dict['r_squared']}")

         # sync result to wandb
        data_loss_0 = [[epoch, loss] for epoch, loss in enumerate(rtn_dict["list_total_0"])]
        table_loss_0 = wandb.Table(data=data_loss_0, columns = ["Epoch", "Total_Loss_0"])

        data_err_0 = [[epoch, err] for epoch, err in enumerate(rtn_dict["list_err_0"])]
        table_err_0 = wandb.Table(data=data_err_0, columns = ["Epoch", "Test_Err_0"])

        for i in range(20):
            try:
                wandb.log({
                    "Total_Loss_Curve_0" : wandb.plot.line(table_loss_0, "Epoch", "Total_Loss", title="Total_loss Curve 0"),
                    "Total_Err_Curve_0" : wandb.plot.line(table_err_0, "Epoch", "Test_Err", title="Test_Err Curve 0"),
                    "best_err" : rtn_dict["best_err"],
                    "r_squared": rtn_dict['r_squared'], 
                })
                wandb.finish()
                
                synced_list.append(folder)
                with open('./sync_list.log', 'w') as fresult:
                    fresult.write(json.dumps(synced_list))
                
                break
            except Exception as e:
                print(f"Error occurred: {str(e)}")
                print(f"Retrying in 60 seconds...")
                time.sleep(60)
        else:
            print(f"Function failed after 20 attempts")
           
        
