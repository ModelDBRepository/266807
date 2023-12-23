# Model code from the paper: Prager, E. M., Dorman, D. B., Hobel, Z. B., Malgady, J. M., Blackwell, K. T., & Plotkin, J. L. (2020). Dopamine oppositely modulates state transitions in striosome and matrix direct pathway striatal spiny neurons. Neuron, 108(6), 1091-1102.

The code in this directory implements the model simulations in the paper: Prager, E. M., Dorman, D. B., Hobel, Z. B., Malgady, J. M., Blackwell, K. T., & Plotkin, J. L. (2020). Dopamine oppositely modulates state transitions in striosome and matrix direct pathway striatal spiny neurons. Neuron, 108(6), 1091-1102.

Simulations were run on the neuroscience gateway portal using the script `sim_upstate.py`
This script by default runs parallel simulations using MPI on the neuroscience gateway,
and code for all simulation parameter combinations are included, though some parameter sets are commented out and may be uncommented to run.

However, in addition to running many parallel simulations on the NSG portal, an example single simulation can be run on a single using workstation using:

`python sim_upstate.py --single`

This script with the --single option will plot somatic voltage traces for a single parameter set, simulating the combination of dispersed synaptic activity with a dendritic plateau potential evoked by clustered synaptic stimulation.

**Note:** To ensure the correct code dependencies and versions are installed, an `environment.yml` file is included for creating a conda python environment with the necessary packages.
To create the correct environment, run `conda env create -f environment.yml`  in a terminal, then to activate the environment run `conda activate py36`. The sim_upstate script can then be run.
The version of moose included requires that the correct python/numpy version be used.

When run on the NSG Portal, it should be run using python 3.6.

This directory includes a pre-compiled version of moose in the moose subdirectory, so a separate installation of moose is not required.

The actual model parameters and model code are in the directory `moose_nerp`, and the parameters for the neurons in this specific publication are in /moose_nerp/D!PatchSample5/ and /moose_nerp/D1MatrixSample2/

Models were developed using the optimization code included in the subdirectory NSGOpt, which itself can be uploaded to the neuroscience gateway portal, where optimizations were run using the scripts such as `D1PatchSample5opt_varyChans.py` for the D1 Patch sample 5 neuron data and model.
The NSGOpt directory contains its own standalone code dependencies including moose, moose_nerp, and ajustador as subdirectories, as it can be uploaded separately to run on the NSG.

Code used to compute Random Forest analysis in the paper is included in `analysis/plot_sim_upstate_vary_params.py`


