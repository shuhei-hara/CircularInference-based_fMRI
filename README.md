# CircularInference-based_fMRI

This repository provides code for performing model-based regression analysis of fMRI data in schizophrenia patients, using the Circular Inference framework.

# behavior
This directory contains scripts and notebooks related to behavioral analysis, including both simulations and analyses of actual subject data:
simulation/ – Simulates behavior to explore how the Circular Inference model behaves under different parameters.
model-free/ – Analyzes subject performance without applying the Circular Inference model.
parameter_recovery/ – Evaluates how accurately the model can recover known parameters from simulated data.
model_recovery/ – Assesses the model's ability to distinguish between different competing models through fitting.
model_fits_subjects/ – Fits the Circular Inference model to real behavioral data to investigate top-down and bottom-up processes in schizophrenia.

# fMRI
This directory contains scripts for fMRI analysis using the Circular Inference model:
model_based_regression.py – Performs voxel-wise regression to identify brain regions encoding model-related processes.
submit_subject_level.sh – Shell script to run model_based_regression.py in parallel for each subject.

# online_task
This directory contains scripts for running the online behavioral experiment used in this project.


Data
______________________________________________
