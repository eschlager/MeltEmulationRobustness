# -*- coding: utf-8 -*-
"""
Created on 29.01.2025
@author: eschlager
NN for emulating 1-dimensional firn models
"""

import logging
import time
import torch
import torch.nn as nn


class NN(nn.Module):
    """
    Neural network model for firn process emulation, supporting multiple input data types and flexible architecture.
    Args:
        targets (list of str): List of target variable names to predict.
        layers_daily (list of int): Layer sizes for the daily feature extractor block.
        layers_regressor (list of int): Layer sizes for the final regression block.
        hidden_activation (str): Activation function to use in hidden layers (e.g., 'ReLU', 'LeakyReLU').
        output_activation (str or dict): Activation function(s) for output heads. If str, applies to all continouts targets while 
                                         uses 'Sigmoid' for binary mask targets; if dict, maps target names to activations.
        layers_medrange (list of int, optional): Layer sizes for the medium-range feature extractor block. Default is None.
        layers_spinup (list of int, optional): Layer sizes for the spin-up feature extractor block. Default is None.
        use_season (bool, optional): Whether to include seasonality features (day-of-year sine/cosine). Default is True.
        trunoff (bool, optional): Whether to include runoff time scale as input. Default is False.
        n_auto (int, optional): Number of auto-regressive variables to include in final regressor. Default is 0 (no auto-regression).
    """

    def __init__(
        self,
        targets,
        layers_daily,
        layers_regressor,
        hidden_activation,
        output_activation,
        layers_medrange=None,
        layers_spinup=None,
        use_season=True,
        trunoff=False,
        n_auto=0
    ):
        super().__init__()
        self.targets = targets
        self.layers_daily = layers_daily
        self.layers_regressor = layers_regressor
        self.hidden_activation = hidden_activation
        self.output_activation = self._get_output_activation(output_activation)
        self.layers_medrange = layers_medrange
        self.layers_spinup = layers_spinup
        self.use_season = use_season
        self.trunoff = trunoff
        self.n_auto = n_auto

        # specify seasonality look-up table
        if self.use_season:
            self._init_seasonality()

        logging.info(f"Set up NN with target(s): {self.targets} ...")

        # init short-term block
        logging.info(f"Init daily block ...")     
        if self.n_auto > 0:
            logging.info(f"   ... add {self.n_auto} input node(s) for auto-regressive approach to daily block.")
            self.layers_daily[0] += self.n_auto
        if self.use_season:
            logging.info(f"   ... add two input nodes for seasonal information to daily block.")
            self.layers_daily[0] += 2
        self.NNdaily = NNBlock(self.layers_daily, hidden_act=self.hidden_activation)    

        # init block for medium-range data
        if self.layers_medrange is not None:
            logging.debug(f"Init medium-range block ...")
            # if self.use_season:
            #     logging.info(f"   ... add two input nodes for seasonal information to medium-range block.")
            #     self.layers_medrange[0] += 2
            self.NNmedrange = NNBlock(self.layers_medrange, hidden_act=self.hidden_activation)
            
        # init spin-up block
        if self.layers_spinup is not None:
            logging.debug(f"Init spin-up block with layers ...")
            self.NNspinup = NNBlock(self.layers_spinup, hidden_act=self.hidden_activation)

        # init final regression block, where all outputs from above blocks are put in to get the final prediction
        logging.info(f"Init final regression block ...")
        self._init_final_regr()

    

    def _init_seasonality(self):
        doy_lookup = torch.arange(1, 366, dtype=torch.float32)
        radians = 2 * torch.pi * doy_lookup / 365
        self.sin_lookup = torch.sin(radians)
        self.cos_lookup = torch.cos(radians)


    def _get_output_activation(self, _output_activation):
        """
        define output activation function respective to target
        """
        if isinstance(_output_activation, str):
            output_activation = {}
            for key in self.targets:
                if 'mask' in key.lower(): # use Simgoid for masks
                    logging.info(f"Set output activation for {key} to \'Sigmoid\'.")
                    logging.info(f"   do not use Sigmoid activation during training because of BCEwithLogits.")
                    output_activation[key] = 'Sigmoid'
                elif _output_activation.lower() == 'linear': # use linear activation for continuous targets if not specified
                    logging.info(f"Set no output activation for {key}.")
                    output_activation[key] = ''
                else: # use specified activation for continuous targets
                    logging.info(f"Set output activation for {key} to \'{_output_activation}\'.")
                    output_activation[key] = _output_activation
            return output_activation
        elif isinstance(_output_activation, dict):
            if set(_output_activation.keys()) == set(self.targets):
                output_activation = {key: _output_activation[key] for key in self.targets}
                return output_activation
            else:
                raise ValueError(f"Output activation dict keys {set(_output_activation.keys())} do not match targets {set(self.targets)}.")
        else:
            raise ValueError(f"Output activation must be a string or a dict, got {type(_output_activation)} instead.")


    def inference(self, x_daily, x_medrange=None, x_spinup=None, y_prev=None, trunoff_map=None, doy=None):
        """ Perform model inference with forward method in eval mode.
        """
        self.eval()
        with torch.no_grad():
            y = self.forward(x_daily, x_medrange, x_spinup, y_prev, trunoff_map, doy)
        
        y_split = y.split(1, dim=1)
        y_processed = []


        for i, (name, head) in enumerate(self.reg_heads.items()):
            out_i = y_split[i]
            if 'mask' in name.lower():   # apply sigmoid activation for mask targets
                out_i = torch.sigmoid(out_i)
                out_i = (out_i > 0.5).float()
            y_processed.append(out_i)

        return torch.cat(y_processed, dim=1)


    def forward(self, x_daily, x_medrange=None, x_spinup=None, y_prev=None, trunoff_map=None, doy=None):
        """ Forward pass through the model.
        Args:
            x_daily (torch.Tensor): Daily input features.
            x_medrange (torch.Tensor, optional): Temporal aggregated input features over a medium-range window (recomm.: 7-30 days). Default is None.
            x_spinup (torch.Tensor, optional): Spin-up input features. Default is None.
            y_prev (torch.Tensor, optional): Previous target values for auto-regression. Default is None.
            trunoff_map (torch.Tensor, optional): Runoff time scale input. Default is None.
            doy (torch.Tensor, optional): Day of year for seasonality features. Default is None.
        Returns:
            outputs (torch.Tensor): Model predictions.
        """

        y = x_daily
        if self.use_season:
            y_season = self._get_seasonality(doy).to(y.device)
            y = torch.cat([y, y_season], dim=1)
        if self.n_auto > 0:  # autoregressive inputs
            y = torch.cat([y, y_prev], dim=1)
        y = self.NNdaily(y)

        if self.layers_medrange is not None:
            y = torch.cat([self.NNmedrange(x_medrange), y], dim=1)

        if self.layers_spinup is not None:
            y = torch.cat([self.NNspinup(x_spinup), y], dim=1)

        if self.trunoff:
            y_trunoff = trunoff_map.to(y.device)
            y = torch.cat([y_trunoff, y], dim=1)
        
        if self.regressor is not None:
            y = self.regressor(y)

        outputs_reg = torch.empty((y.shape[0], len(self.targets)), dtype=torch.float32).to(y.device)
        for i, (_, head) in enumerate(self.reg_heads.items()):
            outputs_reg[:,i] = head(y).squeeze()

        return outputs_reg


    def get_targets(self):
        """ Returns the target variable names.
        """
        return self.targets
    

    def _init_final_regr(self):
        nr_inputs = self.layers_daily[-1]
        if self.layers_spinup is not None:
            nr_inputs += self.layers_spinup[-1]
        if self.layers_medrange is not None:
            nr_inputs += self.layers_medrange[-1]
        if self.trunoff:
            nr_inputs += 1
        
        layers_regressor = [nr_inputs] + self.layers_regressor
        self.regressor = NNBlock(layers_regressor, hidden_act=self.hidden_activation, use_batchnorm=False)
        
        # define final heads for each target
        self.reg_heads = nn.ModuleDict()
        for i,target in enumerate(self.targets):  
            # add one hidden layer per target
            linear1 = nn.Linear(layers_regressor[-1], layers_regressor[-1])
            nn.init.kaiming_normal_(linear1.weight, a=0.01, nonlinearity='leaky_relu')
            activation1 = getattr(nn, self.hidden_activation)()
            
            # add output layer for each target
            linear2 = nn.Linear(layers_regressor[-1], 1)
            if self.output_activation[target] == 'ReLU':
                nn.init.kaiming_normal_(linear2.weight, nonlinearity='relu')
            elif self.output_activation[target] == 'Softplus':
                #nn.init.kaiming_normal_(linear2.weight, nonlinearity='linear')
                nn.init.xavier_normal_(linear2.weight)
                nn.init.constant_(linear2.bias, 0.1) 
            elif self.output_activation[target] == '':
                    nn.init.kaiming_normal_(linear2.weight, nonlinearity='linear')
            elif self.output_activation[target] == 'Sigmoid':  # use linear activation for classification and BCEwithLogitsLoss, i.e. actually Sigmoid activation!
                nn.init.xavier_normal_(linear2.weight)   
            else: 
                nn.init.kaiming_normal_(linear2.weight, a=0.01, nonlinearity='leaky_relu')       
            
            if (self.output_activation[target] == 'Sigmoid') or (self.output_activation[target] == ''):
                # do not use output activation for BCEwithLogits
                self.reg_heads[target] = nn.Sequential(
                                        linear1,
                                        activation1,
                                        linear2)
            else:
                self.reg_heads[target] = nn.Sequential(
                                        linear1,
                                        activation1,
                                        linear2,
                                        getattr(nn, self.output_activation[target])()
                )

        
    def _get_seasonality(self, doy):
        doy = torch.clamp(doy, max=365)
        sin_doy = self.sin_lookup[doy - 1]
        cos_doy = self.cos_lookup[doy - 1]
        x = torch.stack([sin_doy, cos_doy], dim=1)
        return x
        
        
class NNBlock(nn.Module): 
    def __init__(self, layers, hidden_act, use_batchnorm=False):
        super().__init__()
        
        hidden_activation = getattr(nn, hidden_act)()
        self.linears = nn.ModuleList()
        logging.info(f"... init NNBlock with layers: {layers}")
        if len(layers) > 1:
            for i in range(len(layers)-1):
                self.linears.append(nn.Linear(layers[i], layers[i+1]))
                if use_batchnorm:
                    self.linears.append(nn.BatchNorm1d(layers[i+1]))
                self.linears.append(hidden_activation)

        self.apply(self._initialize_weights)  
         
            
    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')

                    
    def forward(self, x):
        out = x
        for module in self.linears:
            out = module(out)
        return out

def _get_block_layers(input_vars, hidden_layer_name, specs):
    hidden_layers = specs['model'].get(hidden_layer_name)
    input_layer = len(input_vars)
    if input_layer == 0:
        if hidden_layer_name == 'layers_daily_feat_extractor':
            logging.info(f"Input variables for {hidden_layer_name}: seasonality only!")
            if specs['model']['use_season']:
                return [0] + hidden_layers
        logging.info(f"No input variables specified for {hidden_layer_name}, skip block.")
        return None
    elif hidden_layers is None or len(hidden_layers) == 0:
        logging.info(f"No hidden layers specified for {hidden_layer_name}, skip block.")
        return None
    else:
        return [input_layer] + hidden_layers


def init_model(var_names_dict, specs):
    n_auto_vars = len(var_names_dict['auto'])
    targets = var_names_dict['target']
    use_season = specs['model']['use_season']
    hidden_activation = specs['model']['hidden_activation']
    output_activation = specs['model']['output_activation']

    layers_daily = _get_block_layers(var_names_dict['daily'], 'layers_daily_feat_extractor', specs)
    layers_medrange = _get_block_layers(var_names_dict['medrange'], 'layers_medrange_feat_extractor', specs)
    layers_spinup = _get_block_layers(var_names_dict['spinup'], 'layers_spinup_feat_extractor', specs)

    layers_regressor = specs['model']['layers_regressor']

    trunoff = True if 'trunoff_file' in specs['directories'] else False
    model = NN(targets=targets, layers_daily=layers_daily, layers_regressor=layers_regressor, 
                    hidden_activation=hidden_activation, output_activation=output_activation, 
                    layers_medrange=layers_medrange, layers_spinup=layers_spinup, 
                    use_season=use_season, trunoff=trunoff, n_auto=n_auto_vars)
    return model


if __name__ == "__main__":
    
    logging.getLogger().setLevel(logging.DEBUG)

    logging.info("Testing model for melt emulation...")
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    layers_daily = [6,16,20]
    layers_spinup = [1,4]
    use_season = True
    layers_regressor = [12,8]
    targets = ['snmel', 'albedom']
    n_auto_reg = 1
    
    n_samples = 5
    x_daily = torch.rand((n_samples, 6)).to(device)
    x_spinup = torch.rand((n_samples, layers_spinup[0])).to(device)
  
    model = NN(targets=targets, layers_daily=layers_daily, layers_regressor=layers_regressor,   
                   hidden_activation='LeakyReLU', output_activation='', 
                   layers_spinup=layers_spinup, 
                   use_season=True, n_auto=n_auto_reg).to(device)
    
    logging.info(model)
    
    y = torch.rand((n_samples, len(targets))).to(device)
    y_prev = torch.rand((n_samples, n_auto_reg)).to(device)
    doy = torch.ones((n_samples,), dtype=torch.int)

    start_time = time.time()
    output = model(x_daily, x_spinup=x_spinup, y_prev=y_prev, doy=doy)
    logging.info(f'Output shape for {n_samples} samples, {len(targets)} targets: {output.shape}')

    

