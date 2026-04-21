# -*- coding: utf-8 -*-
"""
Created on 24.04.2025
@author: eschlager
Create hyperparameter search using Optuna
"""

import os
import sys
import glob
import logging

import numpy as np
import optuna
from optuna.storages import RDBStorage
from sqlalchemy.pool import NullPool
import torch.multiprocessing as mp
import pandas as pd

# import local modules
script_dir = os.path.abspath(os.path.dirname(__file__))
project_dir = os.path.sep.join([script_dir, '..'])
sys.path.append(os.path.sep.join([project_dir, 'src']))
import logging_config
import train_meltNN
import eval_meltNN
import read_yaml
from prepare_trainset import ZarrDataset


SPECS_FILE_NAME = "specs_optuna_melt_modularNN_wotas.yml"
MAIN_TARGET = "snmel"    # define a main target for scoring of trials, as loss for multi-target training depends on different weightings
MAIN_SCORE = "rmse"  # define main score to optimize, must be a return value from train_meltNN.perform_training()



def list_open_fds(): 
    pid = os.getpid() 
    fds = glob.glob(f"/proc/{pid}/fd/*") 
    print("open fds:", len(fds)) 
    

def objective(trial):
    print(f"Starting trial {trial.number}")
    config = read_yaml.read_yaml_file(os.path.sep.join([script_dir, "spec_files", SPECS_FILE_NAME]))
    config['directories']['out_dir'] = os.path.sep.join([config['directories']['out_dir'], f"trial_{trial.number}"])
    config['directories']['out_dir'], out_dir_abs = train_meltNN.mk_my_outdir(config)
    logging_config.define_root_logger(os.path.join(out_dir_abs, f'log.txt'))

    # Define hyperparameters to tune
    config['training']['lr'] = trial.suggest_float('lr', 1e-3, 1e-1, log=True)

    ## tune weights for loss function for multitarget training
    # config['training']['loss_weights'] = trial.suggest_categorical('loss_weights', [{'snmel':0.5, 'albedom':0.5},
                                                                #    {'snmel':0.7, 'albedom':0.3},
                                                                #    {'snmel':0.9, 'albedom':0.1},
                                                                #    {'snmel':0.3, 'albedom':0.7}])

    try:
        scores, train_loss, val_loss, epoch = train_meltNN.perform_training(config)

        trial.set_user_attr("train_loss", train_loss)
        trial.set_user_attr("val_loss", val_loss)
        trial.set_user_attr("epochs", epoch)

        if scores is not None:
            for k,s in scores.items():
                trial.set_user_attr(k, s)
            return scores[MAIN_TARGET+'_'+MAIN_SCORE]
        else:
            return None       
    
    except Exception as e:
        logging.error(f"Trial {trial.number} failed due to: {e}")
        epoch = np.nan
        train_loss = np.nan
        val_loss = np.nan
        trial.set_user_attr("train_loss", train_loss)
        trial.set_user_attr("val_loss", val_loss)
        trial.set_user_attr("epochs", epoch)
        return None
    
    finally:
        list_open_fds()     # for debugging; using num_workers>0 and persistent_workers=True accumulates open files over trials
    


if __name__ == "__main__":

    mp.set_start_method('spawn', force=True)   # needed for num_workers>0
    specs = read_yaml.read_yaml_file(os.path.sep.join([script_dir, "spec_files", SPECS_FILE_NAME]))
    out_dir_abs = os.path.abspath(os.path.sep.join([project_dir, specs['directories']['out_dir']]))
    os.makedirs(out_dir_abs, exist_ok=True)

    assert MAIN_TARGET in specs['data']['target'], f"MAIN_TARGET {MAIN_TARGET} not in target variables {specs['data']['target']}"

    # specs file needs to have most extensive definition of input/target variables, prec_days, and auto_reg_steps so that same zarr file can be used for all trials
    logging.info("Start dataset creation...")
    zarrset = ZarrDataset(specs)
    try:
        zarrset.check_file()
    except (FileNotFoundError, KeyError) as e:
        zarrset.make_file()  

    # create study
    study_name = SPECS_FILE_NAME.split("specs_optuna_")[1].split(".")[0]

    # ## IF WANT TO CONTINUE AN OLD STUDY:
    # study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{project_dir}/output/optuna_v_02_study.db")

    ## IF WANT TO DELETE EXISTING STUDY:
    # optuna.delete_study(study_name=study_name, storage=f"sqlite:///{project_dir}/output/optuna_v_02_study.db")
    
    # IF WANT TO CREATE NEW STUDY:
    db_path = os.path.join(project_dir, "output", "optuna_v_02_study.db")
    storage = RDBStorage(
        url=f"sqlite:///{db_path}",
        engine_kwargs={
            "connect_args": {"check_same_thread": False},
            "poolclass": NullPool,
        },
    )
    study = optuna.create_study(study_name=study_name, direction="minimize", storage=storage)
    
    n_trials = 33
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    print("Best trial:", study.best_trial)

    df = study.trials_dataframe()
    df.to_csv(os.path.sep.join([out_dir_abs, "optuna_trials.csv"]), index=False)

    # run after tuning to collect scores of all trials
    df = pd.DataFrame(columns=['trial', 'rmse', 'mae', 'mbe', 'r2'])
    for trialnr in range(n_trials):
        # take care to not make all the plots in eval_meltNN!
        rmse, mae, mbe, r2 = eval_meltNN.main(specs['directories']['out_dir']+f'/trial_{trialnr}', mode='val', reconstruct_coords=False)
        df = pd.concat([df, pd.DataFrame({'trial':[trialnr], 'rmse':[rmse], 'mae':[mae], 'mbe':[mbe], 'r2':[r2]})], ignore_index=True)
    df.sort_values('mae', inplace=True)
    df.to_csv(os.path.sep.join([out_dir_abs, 'scores_val.csv']), sep=',', index=False)