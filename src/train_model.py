# -*- coding: utf-8 -*-
"""
Created on 23.07.2024
@author: eschlager
training loop
"""

import torch
import torch.nn as nn
import logging
import os
import numpy as np
import csv
import pandas as pd
import time
import gc

from my_utils import shutdown_dataloader


class ModelTrainer():

    def __init__(self, model, dataloader_train, dataloader_val, train_specs, out_dir_abs, auto_mode=False, device='cpu'):
        """
        Args:
            model (nn.Module): the model to be trained
            dataloader_train (torch.utils.data.DataLoader): training data loader
            dataloader_val (torch.utils.data.DataLoader): validation data loader
            train_specs (dict): training specifications, e.g. number of epochs, learning rate, loss function, etc.
            out_dir_abs (str): absolute path to the output directory where the model and loss history
            auto_mode (bool): if True, use previous predictions for auto-regressive learning, otherwise use teacher-forced learning
            device (str): device to use for training, cpu or cuda
        """
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.train_specs = train_specs
        self.out_dir = out_dir_abs
        self.auto_mode = auto_mode
        self.device = device
        

        self.targets = model.get_targets()
            
        self.max_epochs = self.train_specs['epochs']   

        if 'lr_decay' in self.train_specs:
            self.lr_decay = self.train_specs['lr_decay']
        else:
            self.lr_decay = 1.
            
        if 'lr_decay_epochs' in self.train_specs:
            self.lr_decay_epochs = self.train_specs['lr_decay_epochs']
        else:
            self.lr_decay_epochs = self.max_epochs

        params_to_optimize = list(self.model.parameters())

        self.train_loss_components = {key: [] for key in self.targets}
        self.val_loss_components = {key: [] for key in self.targets}
        self.train_loss = []
        self.val_loss = []
        if self.auto_mode:
            self.val_loss_teacher = []
        self.best_val = np.inf
        self.best_train = np.inf
        self.start_epoch = 1
        
        # define which loss function to use for each target variable
        self.loss_fct_tags = {k: self.train_specs['loss_reg'] for k in self.targets}
        
        torch_losses = {
            'mse': torch.nn.MSELoss(),
            'l1': torch.nn.L1Loss(),
            'smoothl1': torch.nn.SmoothL1Loss(),
            'huber': torch.nn.HuberLoss(),
        }
        
        
        self.loss_fct = {k: torch_losses[v] for k,v in self.loss_fct_tags.items()}
        logging.info(f"Use loss functions: {self.loss_fct_tags}.")

        self.trainable_weights = False
        if len(self.targets) > 1:
            # define weights for each loss components if multiple target variables
            if 'loss_weights' in self.train_specs:
                if isinstance(self.train_specs['loss_weights'], dict):  # if having pre-defined weights
                    if set(self.train_specs['loss_weights'].keys()) == set(self.targets):
                        total_weight = sum(self.train_specs['loss_weights'].values())
                        self.loss_weights = {key: self.train_specs['loss_weights'][key]/total_weight for key in self.targets}
                        logging.info(f"Use loss weights {self.loss_weights}.")
                    else:
                        raise ValueError(f"Loss weight keys {set(self.train_specs['loss_weights'].keys())} do not match targets {set(self.targets)}.")
                elif self.train_specs['loss_weights'] == 'equal':  # each component has the same weighting
                    self.loss_weights = {key: 1/len(self.targets) for key in self.targets}
                    logging.info(f"Use equal loss weights {self.loss_weights}.")
                elif self.train_specs['loss_weights'] == 'trainable':  # trainable weights
                    self.loss_weights = {key: nn.Parameter(torch.tensor(1.0, requires_grad=True, device=self.device)) for key in self.targets}
                    logging.info(f"Use trainable loss weights for target variables {self.targets}.")
                    params_to_optimize =  params_to_optimize + list(self.loss_weights.values())  # add weights to parameters to be optimized
                    self.trainable_weights = True
            self.loss = Multitarget_loss(self.targets, self.loss_fct, self.loss_weights, trainable_weights=self.trainable_weights)
        else:
            self.loss_weights = {self.targets[0]: 1.0}
            self.loss = self.loss_fct[self.targets[0]]
            logging.info(f"Only one target variable, use {self.loss_fct_tags[self.targets[0]]} without any weighting.")
            

        if self.train_specs['optimizer'] == 'adam':
            self.optimizer = torch.optim.AdamW(params_to_optimize, lr=self.train_specs['lr'], eps=1e-6) 
        else:
            raise ValueError(f"Optimizer {self.train_specs['optimizer']} not implemented.")
        

        if 'continue_training' in self.train_specs:
            if self.train_specs['continue_training'] == True:
                PATH = os.path.sep.join([self.out_dir, 'latest_model.pth'])
                logging.info(f'Continue training from {PATH}')
                checkpoint = torch.load(PATH, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(
                    checkpoint['optimizer_state_dict'])
                for g in self.optimizer.param_groups:
                    g['lr'] = self.train_specs['lr']
                loss_history = pd.read_csv(os.path.sep.join(
                    [self.out_dir, 'loss.csv']), delimiter=';')
                
                self.train_loss = loss_history['train_loss'].tolist()
                self.val_loss = loss_history['val_loss'].tolist()
                if self.auto_mode:
                    if 'val_loss_teacher' in loss_history.columns: # was auto_mode before
                        self.val_loss_teacher = loss_history['val_loss_teacher'].tolist()
                    else: # was not in auto_mode before
                        self.val_loss_teacher = loss_history['val_loss'].tolist()
                        loss_history['val_loss_teacher'] = loss_history['val_loss']
                        loss_history.to_csv(os.path.sep.join([self.out_dir, 'loss.csv']), sep=';', index=False)
            
                
                self.best_val = np.min(self.val_loss)
                self.best_train = np.min(self.train_loss)
                self.start_epoch = checkpoint['epoch'] + 1


    def to_device(self, *args):
        return [x.to(self.device) for x in args]


    def calc_y_prev(self, _x_daily, _x_medrange, _x_spinup, _y_prev, _trunoff_map, _doy):
        """
        Calculate y_prev using auto-regressive predictions over the given sequence length.
        works for 1 autoregressive timestep only, but for longer rollout window 
        (i.e. make _sequ_len successive predictions, each time updating y_prev with previous prediction,
        to get a y_prev for the last day in the sequence to incorporate error propagation in the training process)
        """
        
        auto_vars = []
        for var in self.dataloader_train.dataset.target_vars:   # iterate through all target variables
            if var+'_d-1' in self.dataloader_train.dataset.auto_inputs:  # identify which is autoreg input variable
                #logging.info(f' ... auto-regressive variable: {var}')
                t_auto_vars = [t for t in self.dataloader_train.dataset.auto_inputs if t.startswith(var+'_d-')]
                #logging.info(f'   auto-regressive vars: {t_auto_vars}')
                auto_vars.append(var)
                nr_auto_reg_steps = len(t_auto_vars)      # get nr of autoreg steps  

        var_to_newcol = {}
        var_to_outcol = {}
        for var in auto_vars:
            out_col_idx = self.dataloader_train.dataset.target_vars.index(var)
            var_to_outcol[var] = out_col_idx
            for d in range(1, nr_auto_reg_steps+1):
                colname = f'{var}_d-{d}'
                if colname in self.dataloader_train.dataset.auto_inputs:
                    new_col_idx = self.dataloader_train.dataset.auto_inputs.index(colname)
                    var_to_newcol[colname] = new_col_idx

        # sequence lengths = rollout window for making making cascading predictions
        _sequ_len = _x_daily.size(0)

        # get y_prev for first (=oldest) element in sequence
        y_prev = _y_prev[0,:,:]

        for step in range(_sequ_len):
            x_daily_prev = _x_daily[step,:,:]
            x_medrange_prev = _x_medrange[step,:,:]
            x_spinup_prev = _x_spinup[step,:,:]
            trunoff_map = _trunoff_map[step,:,:]
            doy_prev = _doy[step,:]
            output = self.model(x_daily_prev, x_medrange_prev, x_spinup_prev, y_prev, trunoff_map, doy_prev)

            y_prev_new = y_prev.clone()

            for var in auto_vars:
                for d in range(nr_auto_reg_steps,0,-1):
                    colname = f'{var}_d-{d}'
                    if colname in var_to_newcol:
                        new_col_idx = var_to_newcol[colname]
                        if d == 1:
                            out_col_idx = var_to_outcol[var]
                            #logging.info(f'Insert latest prediction from {var} to {colname}.')
                            y_prev_new[:, new_col_idx] = output[:, out_col_idx]
                        else:
                            prev_colname = f'{var}_d-{d-1}'
                            prev_col_idx = var_to_newcol[prev_colname]
                            #logging.info(f'Shift previous prediction from {prev_colname} to {colname}.')
                            y_prev_new[:, new_col_idx] = y_prev[:, prev_col_idx]
            y_prev = y_prev_new
 
        return y_prev


    def train(self):            
        with open(os.path.sep.join([self.out_dir, 'loss.csv']), 'a', newline='') as lossfile:
            writer = csv.writer(lossfile, delimiter=';')
            if self.start_epoch == 1:
                if len(self.targets)>1:
                    column_headers = ['epoch', 'lr', 'train_loss', 'val_loss'] + [f"train_{v}_{k}" for k, v in self.loss_fct_tags.items()] + [f"val_{v}_{k}" for k, v in self.loss_fct_tags.items()]
                else:
                    column_headers = ['epoch', 'lr', 'train_loss', 'val_loss']
                if self.auto_mode:
                    column_headers = column_headers + ['val_loss_teacher', 'autoreg_rate']
                writer.writerow(column_headers)

            lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.lr_decay_epochs, gamma=self.lr_decay, last_epoch=-1)
            lr_old = lr_scheduler.get_last_lr()[0]
            early_stopper = EarlyStopper(patience=1000, min_delta=1.e-12) # early stopping if no improvement in validation loss for patience epochs


            logging.info(f'Start training with learning rate {lr_old}')
            epoch = 0
            autoreg_rate = 0   # use teacher forcing for all samples at beginning
            for epoch in range(self.start_epoch, self.max_epochs+self.start_epoch):       
                train_loss_epoch = 0.
                train_loss_epoch_vars = {k: 0. for k in self.targets}
                total_samples = 0
                self.model.train()

                # To use adapting roll-out window during training
                if epoch > 1: # increase rate of autoregressive samples every 30 epochs
                    train_iter = iter(self.dataloader_train)
                    _, _, _, y, _, _, _, _ = next(train_iter)
                    sequ_len = y.size(0)
                    if sequ_len >1 and self.auto_mode:
                        if epoch % 30 == 0 and autoreg_rate < 1:
                            autoreg_rate += 0.1
                            logging.info(f'Use full roll-out window for {autoreg_rate*100}% of samples.')
                            
                
                for x_daily, x_medrange, x_spinup, y, y_prev, trunoff_map, doy, idx in train_iter:
                    x_daily, x_medrange, x_spinup, y, y_prev, trunoff_map = self.to_device(
                                x_daily, x_medrange, x_spinup, y, y_prev, trunoff_map)
                    batch_size = y.size(1)
                    sequ_len = y.size(0)
                    total_samples += batch_size
                    self.optimizer.zero_grad(set_to_none=True)   # set_to_none to save memory: https://discuss.pytorch.org/t/the-location-of-zero-grad-at-the-training-loop/160206
                    
                    if sequ_len>1 and self.auto_mode and autoreg_rate>0:
                        autoreg_batch_size = int(autoreg_rate * batch_size)
                        # logging.info(f'Use {autoreg_batch_size} auto-reg samples')
                        # use full rollout window for first autoreg_batch_size samples
                        x_daily_prev = x_daily[:-1,:autoreg_batch_size,:]
                        x_medrange_prev = x_medrange[:-1,:autoreg_batch_size,:]
                        x_spinup_prev = x_spinup[:-1,:autoreg_batch_size,:]
                        y_prev_prev = y_prev[:-1,:autoreg_batch_size,:]
                        trunoff_map_prev = trunoff_map[:-1,:autoreg_batch_size,:]
                        doy_prev = doy[:-1,:autoreg_batch_size]
                        y_prev_auto = self.calc_y_prev(x_daily_prev, x_medrange_prev, x_spinup_prev, y_prev_prev, trunoff_map_prev, doy_prev)
                        
                        y_prev_teacher = y_prev[-1,autoreg_batch_size:,:]
                        
                        y_prev = torch.cat([y_prev_auto, y_prev_teacher], dim=0)
                    else:
                        # logging.info('Use teacher-forcing')
                        y_prev = y_prev[-1,:,:]

                    # calculate output for latest day
                    x_daily = x_daily[-1,:,:]
                    x_medrange = x_medrange[-1,:,:]
                    x_spinup = x_spinup[-1,:,:]
                    trunoff_map = trunoff_map[-1,:,:]
                    doy = doy[-1,:]
                    output = self.model(x_daily, x_medrange, x_spinup, y_prev, trunoff_map, doy)

                    ## run diagnosis after first epoch
                    # self.model.diagnose_training(x_daily, x_medrange, x_spinup, y_prev, trunoff_map, doy, y[-1,:,:], epoch)

                    # calculate loss
                    if len(self.targets)==1: 
                        loss_batch = self.loss(output, y[-1,:,:])
                    else:
                        loss_batch, loss_per_var = self.loss(output, y[-1,:,:])
                        for i,t in enumerate(self.targets):
                            train_loss_epoch_vars[t] += loss_per_var[i].detach().item() * batch_size
                    loss_batch.backward()

                    # apply gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) 
                    
                    self.optimizer.step()
                    
                    train_loss_epoch += loss_batch.detach().item() * batch_size
                    del output, y, loss_batch
                    
                total_train_loss = train_loss_epoch / total_samples
                
                self.train_loss.append(total_train_loss)
                logging.info(f'E# {epoch} - Training loss: {total_train_loss:.6f}')
                if len(self.targets)>1:
                    for t in self.targets:
                        train_loss_epoch_vars[t] = train_loss_epoch_vars[t] / total_samples
                        self.train_loss_components[t].append(train_loss_epoch_vars[t])
                        logging.info(f'             with {self.loss_fct_tags[t]} for {t}: {train_loss_epoch_vars[t]:.6f}')



                # perform validation of epoch
                val_loss_epoch = 0.
                val_loss_epoch_teacher = 0.
                val_loss_epoch_vars = {k: 0. for k in self.targets}
                total_samples = 0
                self.model.eval()
                with torch.inference_mode():
                    val_iter = iter(self.dataloader_val)
                    for x_daily, x_medrange, x_spinup, y, y_prev, trunoff_map, doy, idx in val_iter:
                        x_daily, x_medrange, x_spinup, y, y_prev, trunoff_map  = self.to_device(
                                x_daily, x_medrange, x_spinup, y, y_prev, trunoff_map)
                        batch_size = y.size(1)
                        total_samples += batch_size

                        if sequ_len >1 and self.auto_mode:
                            # loss in teacher-forced mode
                            output_teacher = self.model(x_daily[-1,:,:], x_medrange[-1,:,:], x_spinup[-1,:,:], y_prev[-1,:,:], trunoff_map[-1,:,:], doy[-1,:])
                            loss_batch_teacher = self.loss(output_teacher, y[-1,:,:])
                            val_loss_epoch_teacher += loss_batch_teacher.item() * batch_size
                            
                            # if input for multiple days available: calculate y_prev using this full rollout window
                            x_daily_prev = x_daily[:-1,:,:]
                            x_medrange_prev = x_medrange[:-1,:,:]
                            x_spinup_prev = x_spinup[:-1,:,:]
                            y_prev_prev = y_prev[:-1,:,:]
                            trunoff_map_prev = trunoff_map[:-1,:,:]
                            doy_prev = doy[:-1,:]
                            y_prev = self.calc_y_prev(x_daily_prev, x_medrange_prev, x_spinup_prev, y_prev_prev, trunoff_map_prev, doy_prev)
                        else:
                            y_prev = y_prev[-1,:,:]

                        # calculate output for latest day
                        x_daily = x_daily[-1,:,:]
                        x_medrange = x_medrange[-1,:,:]
                        x_spinup = x_spinup[-1,:,:]
                        trunoff_map = trunoff_map[-1,:,:]
                        doy = doy[-1,:]
                        output = self.model(x_daily, x_medrange, x_spinup, y_prev, trunoff_map, doy)

                        if len(self.targets)==1: 
                            loss_batch = self.loss(output, y[-1,:,:])
                        else:
                            loss_batch, loss_per_var = self.loss(output, y[-1,:,:])
                            for i,t in enumerate(self.targets):
                                val_loss_epoch_vars[t] += loss_per_var[i].item() * batch_size
                        
                        val_loss_epoch += loss_batch.item() * batch_size



                total_val_loss = val_loss_epoch / total_samples

                if self.auto_mode:
                    total_val_loss_teacher = val_loss_epoch_teacher / total_samples
                    logging.info(f'       Validation loss (auto-regressive): {total_val_loss:.6f}, (teacher-forced): {total_val_loss_teacher:.6f}')
                    self.val_loss_teacher.append(total_val_loss_teacher)
                else:
                    logging.info(f'       Validation loss: {total_val_loss:.6f}')
                
                self.val_loss.append(total_val_loss)
                if len(self.targets)>1:
                    for t in self.targets:
                        val_loss_epoch_vars[t] = val_loss_epoch_vars[t] / total_samples
                        self.val_loss_components[t].append(val_loss_epoch_vars[t])
                        logging.info(f'             with {self.loss_fct_tags[t]} of {t}: {val_loss_epoch_vars[t]:.6f}')
                
                # write to loss file
                if len(self.targets)>1:
                    results_row = [epoch, lr_old, total_train_loss, total_val_loss] + [v[-1] for t,v in self.train_loss_components.items()] + [v[-1] for t,v in self.val_loss_components.items()]
                else:
                    results_row = [epoch, lr_old, total_train_loss, total_val_loss]
                if self.auto_mode:
                    results_row.append(total_val_loss_teacher)
                    results_row.append(autoreg_rate)
                writer.writerow(results_row)
                lossfile.flush()

                if self.trainable_weights:
                    for t,w in self.loss_weights.items():
                        logging.info(f"Learned std of {t}: {w}")

                # save model every 10 epochs
                if epoch % 10 == 0:
                    logging.info(f'Save latest model at epoch {epoch} with validation loss: {total_val_loss:.6f}')
                    self.checkpoint(epoch, os.path.sep.join([self.out_dir, 'latest_model.pth']))
                
                # save best model (w.r.t. training loss)
                if total_train_loss < self.best_train:
                    self.best_train = total_train_loss
                    logging.info(f'Safe new best model at epoch {epoch} with training loss: {self.best_train:.6f}')
                    self.checkpoint(epoch, os.path.sep.join([self.out_dir, 'best_train_model.pth']))

                # save best model (w.r.t. validation loss)
                if total_val_loss < self.best_val:
                    self.best_val = total_val_loss
                    logging.info(f'Safe new best model at epoch {epoch} with validation loss: {self.best_val:.6f}')
                    self.checkpoint(epoch, os.path.sep.join([self.out_dir, 'best_model.pth']))

                if early_stopper.stop(total_val_loss):
                    logging.info(f"Early stopping applied with patience {early_stopper.patience} and min_delta {early_stopper.min_delta}") 
                    logging.info(f'Save latest model at epoch {epoch} with validation loss: {total_val_loss:.6f}')
                    self.checkpoint(epoch, os.path.sep.join([self.out_dir, 'latest_model.pth']))
                else:
                    # update learning rate
                    lr_scheduler.step()  
                    lr_now = lr_scheduler.get_last_lr()[0]
                    if lr_old != lr_now:
                        logging.info(f'Learning rate changed from {lr_old} to {lr_now}')
                        lr_old = lr_now


                
   
        # finally: save latest model       
        logging.info(f'Save latest model at epoch {epoch} with validation loss: {total_val_loss:.6f}')
        self.checkpoint(epoch, os.path.sep.join([self.out_dir, 'latest_model.pth']))
        
        # finally: clean up
        del output, self.optimizer, lr_scheduler

        self.close_iter(train_iter)
        del train_iter
        self.close_iter(val_iter)
        del val_iter

        return self.best_train, self.best_val, epoch, True


    def close_iter(self, iterator):
        try:
            if hasattr(iterator, "_shutdown_workers"): 
                iterator._shutdown_workers() 
        except Exception: 
            logging.exception("iterator._shutdown_workers() failed") # remove references so object can be garbage-collected 
        


    def close(self):
        # Close/flush any writers owned by trainer
        try:
            if hasattr(self, "tb_writer") and self.tb_writer is not None:
                try:
                    self.tb_writer.flush()
                    self.tb_writer.close()
                except Exception:
                    logging.exception("Failed to close tb_writer")
                finally:
                    self.tb_writer = None
        except Exception:
            logging.exception("Error closing tb_writer")


        # Ask any dataloaders to shutdown their workers
        try:
            if getattr(self, "dataloader_train", None) is not None:
                try:
                    shutdown_dataloader(self.dataloader_train)
                except Exception:
                    logging.exception("shutdown_dataloader(train) failed")
                finally:
                    # break reference
                    self.dataloader_train = None
        except Exception:
            logging.exception("Error shutting down train dataloader")

        try:
            if getattr(self, "dataloader_val", None) is not None:
                try:
                    shutdown_dataloader(self.dataloader_val)
                except Exception:
                    logging.exception("shutdown_dataloader(val) failed")
                finally:
                    self.dataloader_val = None
        except Exception:
            logging.exception("Error shutting down val dataloader")

        # Clear other possibly large attributes
        try:
            self.model = None
        except Exception:
            pass

        # Final garbage collection and small pause
        try:
            gc.collect()
            time.sleep(0.05)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception:
            pass


    def checkpoint(self, epoch, filename):
        '''
        https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
        https://machinelearningmastery.com/managing-a-pytorch-training-process-with-checkpoints-and-early-stopping/
        '''
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'tain_loss': self.train_loss[-1],
            'val_loss': self.val_loss[-1],
        }, filename)


class EarlyStopper:
    def __init__(self, patience=10, min_delta=1.e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def stop(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= self.min_validation_loss :
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



class Multitarget_loss(nn.Module):
    def __init__(self, target_names, loss_fct, weights, trainable_weights=False):
        super().__init__()
        self.weights = weights
        self.loss_fct = loss_fct
        self.target_names = target_names
        self.trainable_weights = trainable_weights

    def forward(self, outputs, targets):
        loss_per_var = []
        loss_weighted = 0.
        for i,t in enumerate(self.target_names):
            loss_per_var.append(self.loss_fct[t](outputs[:,i], targets[:,i]))

            if self.trainable_weights:  # if weights are trainable, use uncertainty weighting (https://arxiv.org/pdf/1705.07115)
                precision = (self.weights[t]**(-2))*0.5
                loss_weighted += loss_per_var[-1]*(precision) + torch.log(1+self.weights[t])
            else:
                loss_weighted += self.weights[t] * loss_per_var[-1]

        return loss_weighted, loss_per_var