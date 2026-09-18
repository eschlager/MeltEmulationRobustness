# Emulating Greenland Ice Sheet Surface Melt from Polar RCMs -- a transferability study

This repository is part of a PhD project, which aims to emulate components from firn models of polar RCMs using Machine Learning (ML).
SMB estimation with polar RCMs includes (1) dynamic downscaling of the atmospheric variables, and (2) a firn model to infer SMB components based on the atmospheric data. While there are many efforts in statistical downscaling using ML to replace the computationally expensive RCMs, this work focuses on the second part: emulating the firn model.
The ML model is a modular Neural Network, trained on HIRHAM5 data; it takes daily atmospheric variables as inputs, and yields daily surface melt as outputs.

This code is used to investigate the transferability of the emulator to HIRHAM5 simulation data forced by a different GCM than it was trained on. This study includes
* a crossvalidation study to assess interannual variabiliy of the emulator
* repeated training runs to account for training variability due to random layer initialization and training data shuffling
* training and testing on simulation data forced by ERA-Interim (1990-2016), and CESM2 and EC-Earth (1990-2016 and 2075-2100)
* a perturbation study to investigate the physical consistency of the emulator
* additional emulator training with physics-informed loss terms
* additional emulator training including albedo, excluding snow/rain, or excluding the long-term module to investigate influences on transferability


<img src="https://github.com/eschlager/MeltEmulation/blob/main/modeling_overview.png" title="SMB estimation with polar RCMs" height="270">

## Network Architecture
<img src="https://github.com/eschlager/MeltEmulation/blob/main/NNmelt_architecture.png" height="300">


## Data Availability
Data for model training is available upon request.
The trained models and generated outputs can be downloaded here: [https://anon.erda.au.dk/sharelink/CIWfIqtI0i](https://anon.erda.au.dk/sharelink/CIWfIqtI0i)



## Citation
This repository accompanies a paper that is currently in the submission process: "Transferability of ML emulators for polar RCM surface melt under different forcings: Limitations and recommendations".



Code: licensed under MIT — see `LICENSE` file .

