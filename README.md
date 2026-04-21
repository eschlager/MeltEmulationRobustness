# Emulating Greenland Ice Sheet Surface Melt from Polar RCMs

This repository is part of a PhD project, which aims to emulate components from firn models of polar RCMs using Machine Learning (ML).
SMB estimation with polar RCMs includes (1) dynamic downscaling of the atmospheric variables, and (2) a firn model to infer SMB components based on the atmospheric data. While there are many efforts in statistical downscaling using ML to replace the computationally expensive RCMs, this work focuses on the second part: emulating the firn model.
The proposed ML model is a modular Neural Network, trained on HIRHAM5 data; it takes daily atmospheric variables as inputs, and yields daily surface melt as outputs.

<img src="https://github.com/eschlager/MeltEmulation/blob/main/modeling_overview.png" title="SMB estimation with polar RCMs" height="270">

## Network Architecture
<img src="https://github.com/eschlager/MeltEmulation/blob/main/NNmelt_architecture.png" height="300">


## Data Availability
Data for model training is available upon request.
Output data is available at Zenodo: [DOI: 10.5281/zenodo.19627367](https://doi.org/10.5281/zenodo.19627367)


## Limitations
* The model is trained on HIRHAM5 data which was forced by ERA-Interim re-analysis data. Validity on HIRHAM5 runs forced by GCMs or other atmospheric forcings, and for future time periods must be assessed!
* Since different polar RCMs and firn models have different parameterizations and physical schemes, this emulator cannot be assumed to be valid when being applied to atmospheric forcings from other RCMs such as MAR or RACMO!


## Repository Overview
```
MeltEmulation
├── data (see Data Availability)
├── evaluations: jupyter notebooks for evaluating and comparing models
│   ├── calc_climatology: calculate climatology of true and predicted melt values (Table A1)
│   ├── evaluate_best_models: computes scores for best model for each configuration (Tables 3 and 4)
│   ├── evaluate_modularNN: evaluate Modular NN basin-wise (Table 5, Fig. 8)
│   ├── evaluate_tuning_val: evaluate the tuning results of all configurations (Tables C1 and C2, Fig. C1)
│   └── plot_GRL: plot zones and basin maps
├── figures: resulting fiugres from evaluations scripts
├── modeling
│   ├── models: contains model architecture meltNN.py
│   ├── specs_files: yaml files with model specifications used in train_meltNN and eval_meltNN.py
│   ├── eval_meltNN.py: main file for evaluating a model, producing hexbin plots and melt maps (Figures 3, 4, 5)
│   ├── prepare_trainset.py: prepares a zarr file ready for efficient training based on specifications file
│   ├── train_loop_optuna.py: Hyperparameter optimization for a specific configuration using Optuna library
│   └── train_meltNN.py: main script for model training
├── output (see Data Availability)
├── preprocessing
│   └── HIRHAM5_reanalysis:
│       ├── create_basefile.py: data cleaning and transformation to zarr file, which is then used for pepare_trainset.py
│       ├── create_data_splits.py: create temporal split (train, val, test) and temporal sub-sampling
│       └── create_spatial_subsampling.py: calculate GrIS zones and spatial sub-sampling
└── src [reusable code parts, e.g., training pipeline]
	├── create_dataset.py: creation of a Pytorch Dataset for efficient dataloading during training
	├── eval_model.py: evaluations routines used in eval_meltNN.py
	├── GRL_plotter.py: routines for plotting maps in lat/lon projection
	├── predictor.py: ModelPredictor class to make predictions from preprocessed file. Operates lazily and saves predicted data to zarr files.
	└── train_model.py: training loop used in train_meltNN.py
				
```


## Citation
This repository accompanies a paper that is currently in the submission process
[DOI: 10.5194/egusphere-2026-7](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-7/)



Code: licensed under MIT — see `LICENSE` file .

